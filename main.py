from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Plain
from astrbot.api.star import Context, Star, register
from astrbot.core.platform.message_type import MessageType
from astrbot.core.star.filter.command import GreedyStr

from .config import PluginConfig
from .extractor import TargetExtractor, TargetResult
from .formatter import format_punchline_response
from .history_manager import GroupHistoryManager
from .parser_bridge import ParserBridge, RequestContext
from .prompt import SYSTEM_PROMPT, build_user_prompt

# 中文数字映射表与历史模式正则（支持“最近/前”“阿拉伯数字/中文数字”如“前三十条”、“最近30条”）
CN_NUM_MAP = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
    "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20,
    "二十一": 21, "二十二": 22, "二十三": 23, "二十四": 24, "二十五": 25,
    "二十六": 26, "二十七": 27, "二十八": 28, "二十九": 29, "三十": 30,
    "三十一": 31, "三十二": 32, "三十三": 33, "三十四": 34, "三十五": 35,
    "三十六": 36, "三十七": 37, "三十八": 38, "三十九": 39, "四十": 40,
    "四十五": 45, "五十": 50,
}

RECENT_CMD_PATTERN = re.compile(
    r"(?:最近|前)\s*(\d+|[一二两三四五六七八九十]{1,4})\s*条?",
    re.IGNORECASE
)


def _get_platform_id(event: AstrMessageEvent) -> str:
    if hasattr(event, "get_platform_id") and callable(event.get_platform_id):
        try:
            v = event.get_platform_id()
            if v:
                return str(v).strip()
        except Exception:
            pass
    if hasattr(event, "get_platform_name") and callable(event.get_platform_name):
        try:
            v = event.get_platform_name()
            if v:
                return str(v).strip()
        except Exception:
            pass
    umo = getattr(event, "unified_msg_origin", "") or ""
    if ":" in umo:
        return umo.split(":")[0].strip()
    return "default"


def _get_bot_id(event: AstrMessageEvent) -> str:
    if hasattr(event, "get_self_id") and callable(event.get_self_id):
        try:
            v = event.get_self_id()
            if v is not None:
                return str(v).strip()
        except Exception:
            pass
    return "default"


def _get_group_id(event: AstrMessageEvent) -> str:
    if hasattr(event, "get_group_id") and callable(event.get_group_id):
        try:
            v = event.get_group_id()
            if v is not None:
                return str(v).strip()
        except Exception:
            pass
    return ""


def _get_sender_id(event: AstrMessageEvent) -> str:
    if hasattr(event, "get_sender_id") and callable(event.get_sender_id):
        try:
            v = event.get_sender_id()
            if v is not None:
                return str(v).strip()
        except Exception:
            pass
    return ""


def _get_sender_nickname(event: AstrMessageEvent) -> str:
    """获取发送者在群内的显示昵称（优先群名片 > 用户昵称 > 账号ID）"""
    # 1. 优先调用 event.get_sender_name()（AstrBot aiocqhttp 适配器会自动填入 card or nickname）
    if hasattr(event, "get_sender_name") and callable(event.get_sender_name):
        try:
            name = event.get_sender_name()
            if name and str(name).strip() and str(name).strip() != "N/A":
                return str(name).strip()
        except Exception:
            pass

    # 2. 从 message_obj.sender 中提取 card 或 nickname
    msg_obj = getattr(event, "message_obj", None)
    if msg_obj:
        sender = getattr(msg_obj, "sender", None)
        if sender:
            card = getattr(sender, "card", None)
            if card and str(card).strip():
                return str(card).strip()
            nick = getattr(sender, "nickname", None)
            if nick and str(nick).strip() and str(nick).strip() != "N/A":
                return str(nick).strip()

        # 3. 检查 raw_message (OneBot dict) 内部 sender.card
        raw = getattr(msg_obj, "raw_message", None)
        if isinstance(raw, dict):
            raw_sender = raw.get("sender")
            if isinstance(raw_sender, dict):
                card = raw_sender.get("card")
                if card and str(card).strip():
                    return str(card).strip()
                nick = raw_sender.get("nickname")
                if nick and str(nick).strip() and str(nick).strip() != "N/A":
                    return str(nick).strip()

    # 4. 兜底返回 sender_id
    sid = _get_sender_id(event)
    return sid or ""


class CooldownRecord:
    __slots__ = ("timestamp", "request_id", "prev_time", "prev_request_id")

    def __init__(
        self,
        timestamp: float,
        request_id: str,
        prev_time: float | None = None,
        prev_request_id: str | None = None,
    ) -> None:
        self.timestamp = timestamp
        self.request_id = request_id
        self.prev_time = prev_time
        self.prev_request_id = prev_request_id


class CooldownTicket(tuple):
    """
    预接纳凭据，继承 tuple 以保持对 (old_user_time, old_group_time) 的完全解包兼容。
    同时携带 request_id 及上下文标识，用于归属校验与精确回滚。
    """
    def __new__(
        cls,
        old_user_time: float | None,
        old_group_time: float | None,
        request_id: str,
        platform_id: str = "",
        bot_id: str = "",
        group_id: str = "",
        user_id: str = "",
    ):
        instance = super().__new__(cls, (old_user_time, old_group_time))
        instance.old_user_time = old_user_time
        instance.old_group_time = old_group_time
        instance.request_id = request_id
        instance.platform_id = platform_id
        instance.bot_id = bot_id
        instance.group_id = group_id
        instance.user_id = user_id
        return instance


class CooldownManager:
    """
    笑点解析冷却管理器
    1. 基于 time.monotonic() 单调时钟。
    2. 用户冷却：按 (platform, bot, group, user) 隔离，私聊则 group 为空字符串。
    3. 群组冷却：按 (platform, bot, group) 隔离，私聊不串群、不参与群冷却。
    4. 检查与接纳状态更新具备并发一致性（基于 asyncio.Lock）。
    5. 无权限、不符合触发条件、队列已满等未接纳请求不消耗冷却。
    6. 不使用后台定时器或延迟等待任务。
    7. 预接纳携带归属 request_id；回滚时分别独立检查用户、群记录仍属于本请求，严禁覆盖后来请求。
    """

    def __init__(self) -> None:
        self._user_records: dict[tuple[str, str, str, str], CooldownRecord] = {}
        self._group_records: dict[tuple[str, str, str], CooldownRecord] = {}
        self._lock = asyncio.Lock()

    async def check(
        self,
        platform_id: str,
        bot_id: str,
        group_id: str,
        user_id: str,
        user_cooldown_sec: float,
        group_cooldown_sec: float,
    ) -> tuple[bool, float]:
        """
        只读检查当前是否处于冷却中（并发安全）。
        返回 (是否可接纳, 剩余冷却秒数)。
        若不可接纳，剩余冷却秒数为两项中最大的剩余时间。
        """
        async with self._lock:
            now = time.monotonic()
            rem = 0.0

            # 检查用户冷却
            if user_cooldown_sec > 0:
                user_key = (platform_id, bot_id, group_id, user_id)
                rec_u = self._user_records.get(user_key)
                if rec_u is not None:
                    elapsed = now - rec_u.timestamp
                    if elapsed < user_cooldown_sec:
                        rem = max(rem, user_cooldown_sec - elapsed)

            # 检查群冷却（仅在群聊且配置冷却大于 0 时生效）
            if group_cooldown_sec > 0 and group_id:
                group_key = (platform_id, bot_id, group_id)
                rec_g = self._group_records.get(group_key)
                if rec_g is not None:
                    elapsed = now - rec_g.timestamp
                    if elapsed < group_cooldown_sec:
                        rem = max(rem, group_cooldown_sec - elapsed)

            if rem > 0:
                return False, rem
            return True, 0.0

    async def acquire_atomic(
        self,
        platform_id: str,
        bot_id: str,
        group_id: str,
        user_id: str,
        user_cooldown_sec: float,
        group_cooldown_sec: float,
        request_id: str | None = None,
    ) -> tuple[bool, float, CooldownTicket]:
        """
        原子检查并预接纳：
        两项均满足才接纳，并原子更新时间戳与归属 request_id；
        返回 (True, 0.0, CooldownTicket(old_u, old_g, req_id, ...))；
        若被冷却拒绝，返回 (False, 剩余秒数, CooldownTicket(None, None, ""))。
        """
        async with self._lock:
            now = time.monotonic()
            rem = 0.0
            user_key = (platform_id, bot_id, group_id, user_id)
            group_key = (platform_id, bot_id, group_id)

            old_u = self._user_records.get(user_key)
            if user_cooldown_sec > 0 and old_u is not None:
                elapsed = now - old_u.timestamp
                if elapsed < user_cooldown_sec:
                    rem = max(rem, user_cooldown_sec - elapsed)

            old_g = self._group_records.get(group_key)
            if group_cooldown_sec > 0 and group_id and old_g is not None:
                elapsed = now - old_g.timestamp
                if elapsed < group_cooldown_sec:
                    rem = max(rem, group_cooldown_sec - elapsed)

            if rem > 0:
                return False, rem, CooldownTicket(None, None, "")

            req_id = str(request_id).strip() if request_id else uuid.uuid4().hex
            old_u_time = old_u.timestamp if old_u is not None else None
            old_g_time = old_g.timestamp if old_g is not None else None

            # 两项均满足，原子更新时间戳并登记归属 request_id 与前驱版本
            if user_cooldown_sec > 0:
                self._user_records[user_key] = CooldownRecord(
                    timestamp=now,
                    request_id=req_id,
                    prev_time=old_u_time,
                    prev_request_id=old_u.request_id if old_u is not None else None,
                )
            if group_cooldown_sec > 0 and group_id:
                self._group_records[group_key] = CooldownRecord(
                    timestamp=now,
                    request_id=req_id,
                    prev_time=old_g_time,
                    prev_request_id=old_g.request_id if old_g is not None else None,
                )

            ticket = CooldownTicket(
                old_user_time=old_u_time,
                old_group_time=old_g_time,
                request_id=req_id,
                platform_id=platform_id,
                bot_id=bot_id,
                group_id=group_id,
                user_id=user_id,
            )
            return True, 0.0, ticket

    async def revert(
        self,
        platform_id: str | CooldownTicket = "",
        bot_id: str = "",
        group_id: str = "",
        user_id: str = "",
        old_user_time: float | None = None,
        old_group_time: float | None = None,
        request_id: str | None = None,
        ticket: CooldownTicket | tuple | None = None,
    ) -> None:
        """
        回滚接纳状态（当因排队超时或尚未取得执行锁时取消调用）：
        分别独立检查当前用户、群记录是否仍属于本请求（request_id 一致），才允许撤销；
        若当前记录已归属于后来请求，则严禁覆盖或删除后来请求建立的记录。
        """
        # 支持以 ticket 对象形式直接调用
        if isinstance(platform_id, CooldownTicket) or (isinstance(platform_id, tuple) and hasattr(platform_id, "request_id")):
            ticket = platform_id
            platform_id = ""

        if ticket is not None:
            if not platform_id and hasattr(ticket, "platform_id"):
                platform_id = getattr(ticket, "platform_id", "")
            if not bot_id and hasattr(ticket, "bot_id"):
                bot_id = getattr(ticket, "bot_id", "")
            if not group_id and hasattr(ticket, "group_id"):
                group_id = getattr(ticket, "group_id", "")
            if not user_id and hasattr(ticket, "user_id"):
                user_id = getattr(ticket, "user_id", "")
            if not request_id and hasattr(ticket, "request_id"):
                request_id = getattr(ticket, "request_id", "")

        req_id = request_id
        if not req_id and hasattr(old_user_time, "request_id"):
            req_id = getattr(old_user_time, "request_id", None)

        async with self._lock:
            user_key = (platform_id, bot_id, group_id, user_id)
            group_key = (platform_id, bot_id, group_id)

            # 1. 独立检查用户记录归属
            cur_u = self._user_records.get(user_key)
            if cur_u is not None:
                if req_id is None or cur_u.request_id == req_id:
                    prev_t = cur_u.prev_time if cur_u.prev_time is not None else (
                        old_user_time if isinstance(old_user_time, (int, float)) else None
                    )
                    prev_rid = cur_u.prev_request_id
                    if prev_t is None:
                        self._user_records.pop(user_key, None)
                    else:
                        self._user_records[user_key] = CooldownRecord(
                            timestamp=prev_t,
                            request_id=prev_rid or "",
                            prev_time=None,
                            prev_request_id=None,
                        )

            # 2. 独立检查群记录归属
            if group_id:
                cur_g = self._group_records.get(group_key)
                if cur_g is not None:
                    if req_id is None or cur_g.request_id == req_id:
                        prev_t = cur_g.prev_time if cur_g.prev_time is not None else (
                            old_group_time if isinstance(old_group_time, (int, float)) else None
                        )
                        prev_rid = cur_g.prev_request_id
                        if prev_t is None:
                            self._group_records.pop(group_key, None)
                        else:
                            self._group_records[group_key] = CooldownRecord(
                                timestamp=prev_t,
                                request_id=prev_rid or "",
                                prev_time=None,
                                prev_request_id=None,
                            )


@register(
    "astrbot_plugin_punchline_analyzer",
    "Antigravity",
    "智能解析群聊发言、图片表情包、短视频、合并转发聊天记录及B站等分享中的幽默笑点与梗。",
    "1.0.0",
    "",
)
class PunchlineAnalyzerPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | dict | None = None) -> None:
        super().__init__(context)
        raw_conf = config if isinstance(config, dict) else (config.to_dict() if config else {})
        self.cfg = PluginConfig(raw_conf)

        # 缓存与桥梁
        self.history_manager = GroupHistoryManager(max_buffer_per_group=100)
        self.parser_bridge = ParserBridge(context)

        # 临时工作目录
        self.temp_dir = Path(tempfile.gettempdir()) / "punchline_analyzer"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.extractor = TargetExtractor(
            config=self.cfg,
            parser_bridge=self.parser_bridge,
            history_manager=self.history_manager,
            temp_dir=self.temp_dir,
        )

        # 单任务并发锁与排队计数器（保护 2 核 1.7G 机器不被瞬时并发拖垮）
        self._lock = asyncio.Lock()
        self._waiting_count = 0

        # 并发一致性冷却管理器
        self.cooldown_manager = CooldownManager()

    async def initialize(self) -> None:
        logger.info("[笑点解析] 插件已初始化加载完成")

    async def terminate(self) -> None:
        # 清理临时目录
        try:
            if self.temp_dir.exists():
                for f in self.temp_dir.glob("*"):
                    try:
                        if f.is_file():
                            f.unlink()
                    except Exception:
                        pass
        except Exception:
            pass
        logger.info("[笑点解析] 插件已卸载")

    # ================= 触发入口 1：独立指令 /笑点解析 =================
    @filter.command("笑点解析", alias={"笑点分析"})
    async def cmd_punchline(self, event: AstrMessageEvent, arg: GreedyStr = ""):
        """独立指令入口：/笑点解析 [最近30条]"""
        event.stop_event()

        # 白名单群过滤：与自动触发入口保持统一权限检查
        group_id = event.get_group_id()
        if not self.cfg.is_group_allowed(group_id):
            return

        arg_str = (arg or "").strip()

        # 判断是否为“最近N条”
        is_history_mode, count = self._parse_history_args(arg_str)
        async for res in self._process_flow(
            event=event,
            user_text=arg_str,
            is_history_mode=is_history_mode,
            history_count=count,
        ):
            yield res

    # ================= 触发入口 2：@当前机器人 + 关键词 =================
    @filter.event_message_type(filter.EventMessageType.ALL, priority=50)
    async def on_message(self, event: AstrMessageEvent):
        """统一监听：维护群历史缓存，并在精准命中 @当前机器人 + “笑点解析” 时触发"""
        # 记录到群聊滑动窗口历史（即使不触发也要记，以备后续溯源）
        self.history_manager.record_event(event)

        # 1. 自动触发总开关校验：若关闭自动入口，直接退出，不进入后续识别，但历史记录已正常维护
        if not self.cfg.enable_auto_trigger:
            return

        # 仅处理群消息
        mtype = event.get_message_type()
        is_group = (
            mtype == MessageType.GROUP_MESSAGE
            or str(mtype).lower() in ("groupmessage", "group_message", "messagetype.group_message")
            or (hasattr(mtype, "name") and "GROUP" in mtype.name)
        )
        if not is_group:
            return

        # 仅处理收到的用户消息，避免机器人自发自收导致循环
        self_id = str(event.get_self_id() or "").strip()
        sender_id = str(event.get_sender_id() or "").strip()
        if sender_id and self_id and sender_id == self_id:
            return

        # 白名单群过滤
        group_id = event.get_group_id()
        if not self.cfg.is_group_allowed(group_id):
            return

        chain = event.get_messages() or []
        if not chain:
            return

        # 提取当前发送的顶层纯文字
        top_level_plain = "".join(
            getattr(seg, "text", "") for seg in chain if isinstance(seg, Plain) or type(seg).__name__ in ("Plain", "MockPlain")
        ).strip()

        # 检查是否包含核心关键词（支持“笑点解析”或“笑点分析”）
        has_keyword = "笑点解析" in top_level_plain or "笑点分析" in top_level_plain
        if not has_keyword:
            return

        # 触发判定（支持无需 @bot 直接使用，同时保留防误触机制）：
        # 1. 显式 @当前机器人
        at_self = any(
            (isinstance(seg, At) or getattr(seg, "type", None) in ("at", "At") or type(seg).__name__ in ("At", "MockAt"))
            and str(getattr(seg, "qq", "")).strip() == self_id
            for seg in chain
        )
        # 防误触：若消息显式 @ 了其他人且未 @ 机器人本身，属于群友之间对话，不应误抢答
        at_others = any(
            (isinstance(seg, At) or getattr(seg, "type", None) in ("at", "At") or type(seg).__name__ in ("At", "MockAt"))
            and str(getattr(seg, "qq", "")).strip() != self_id
            for seg in chain
        )
        if at_others and not at_self:
            return
        # 2. 以“笑点解析”或“笑点分析”作为前缀指令（如：笑点解析 最近30条、笑点分析前三十条、/笑点解析、。笑点分析）
        cleaned_plain = top_level_plain.lstrip("/。!！# ")
        is_cmd_prefix = cleaned_plain.startswith("笑点解析") or cleaned_plain.startswith("笑点分析")

        # 3. 附带了引用回复 (Reply) 或富媒体 (Image/Video) 且文字包含关键词
        has_reply_or_media = any(
            getattr(seg, "type", "").lower() in ("reply", "image", "video")
            or type(seg).__name__.lower() in ("reply", "mockreply", "image", "mockimage", "video", "mockvideo")
            for seg in chain
        )

        if not (at_self or is_cmd_prefix or has_reply_or_media):
            return

        # 命中触发！调用 stop_event() 避免 AstrBot 默认聊天回复产生双份回答
        event.stop_event()

        # 判断是否指定了最近条数（例如 @机器人 笑点解析 最近30条）
        is_history_mode, count = self._parse_history_args(top_level_plain)

        async for res in self._process_flow(
            event=event,
            user_text=top_level_plain,
            is_history_mode=is_history_mode,
            history_count=count,
        ):
            yield res

    # ================= 核心执行链路 =================
    async def _process_flow(
        self,
        event: AstrMessageEvent,
        user_text: str,
        is_history_mode: bool = False,
        history_count: int = 30,
    ):
        # 白名单群过滤防御（未通过权限的不消耗冷却）
        group_id_val = _get_group_id(event)
        if not self.cfg.is_group_allowed(group_id_val):
            return

        platform_id = _get_platform_id(event)
        bot_id = _get_bot_id(event)
        sender_id = _get_sender_id(event)
        sender_nick = _get_sender_nickname(event)

        # 1. 检查冷却状态（只读检查，此时未接纳不消耗冷却）
        is_ok, rem = await self.cooldown_manager.check(
            platform_id=platform_id,
            bot_id=bot_id,
            group_id=group_id_val,
            user_id=sender_id,
            user_cooldown_sec=self.cfg.user_cooldown_seconds,
            group_cooldown_sec=self.cfg.group_cooldown_seconds,
        )
        if not is_ok:
            event.stop_event()
            rem_str = f"{rem:.1f}" if rem < 1.0 else f"{int(round(rem))}"
            yield event.plain_result(f"笑点解析冷却中，请等待 {rem_str} 秒后再试~")
            return

        # 2. 有限排队控制：保护服务器资源，超额排队直接快速拒绝（队列已满不消耗冷却）
        if self._waiting_count >= self.cfg.max_wait_queue:
            yield event.plain_result("当前笑点解析排队人数较多（保护服务器资源），请稍后再试~")
            return

        # 3. 原子检查并预接纳（并发一致性保障，两项均满足才接纳并更新时间戳与归属 request_id）
        req_id = uuid.uuid4().hex
        acquired_cd, rem, old_cd = await self.cooldown_manager.acquire_atomic(
            platform_id=platform_id,
            bot_id=bot_id,
            group_id=group_id_val,
            user_id=sender_id,
            user_cooldown_sec=self.cfg.user_cooldown_seconds,
            group_cooldown_sec=self.cfg.group_cooldown_seconds,
            request_id=req_id,
        )
        if not acquired_cd:
            event.stop_event()
            rem_str = f"{rem:.1f}" if rem < 1.0 else f"{int(round(rem))}"
            yield event.plain_result(f"笑点解析冷却中，请等待 {rem_str} 秒后再试~")
            return

        self._waiting_count += 1
        acquired = False
        try:
            try:
                async with asyncio.timeout(self.cfg.queue_wait_seconds):
                    await self._lock.acquire()
                    acquired = True
            except asyncio.TimeoutError:
                # 队列等待超时，请求未能正式被接纳执行，按归属撤销本请求冷却预占！
                await self.cooldown_manager.revert(
                    platform_id=platform_id,
                    bot_id=bot_id,
                    group_id=group_id_val,
                    user_id=sender_id,
                    old_user_time=old_cd[0],
                    old_group_time=old_cd[1],
                    request_id=req_id,
                    ticket=old_cd,
                )
                yield event.plain_result(
                    f"笑点解析排队等待超过 {self.cfg.queue_wait_seconds:.0f} 秒，当前任务繁忙，请稍后再试~"
                )
                return
            except asyncio.CancelledError:
                # 排队等待全局锁期间收到 CancelledError，尚未取得执行锁时安全撤销本请求预占，再透传取消
                if not acquired:
                    await self.cooldown_manager.revert(
                        platform_id=platform_id,
                        bot_id=bot_id,
                        group_id=group_id_val,
                        user_id=sender_id,
                        old_user_time=old_cd[0],
                        old_group_time=old_cd[1],
                        request_id=req_id,
                        ticket=old_cd,
                    )
                raise
            finally:
                self._waiting_count = max(0, self._waiting_count - 1)

            # 已取得锁并开始执行的请求，保留现有冷却语义
            task_dir = self.temp_dir / f"task_{uuid.uuid4().hex}"
            task_dir.mkdir(parents=True, exist_ok=True)
            req_ctx = RequestContext(task_dir=task_dir)
            target: TargetResult | None = None
            try:
                # 4. 整体执行超时保护（包含素材准备、抽帧与模型生成全流程）
                async with asyncio.timeout(self.cfg.timeout_seconds):
                    # 目标提取与多模态素材准备
                    target = await self.extractor.extract_target(
                        event=event,
                        user_plain_text=user_text,
                        is_history_mode=is_history_mode,
                        requested_count=history_count,
                        task_dir=task_dir,
                        req_ctx=req_ctx,
                    )

                    if target.target_type == "none":
                        yield event.plain_result(
                            "未能识别到要解析的笑点目标。\n"
                            "💡 请直接引用要解析的消息，或在发送图片/视频/分享卡片时 @我 并附带“笑点解析”~"
                        )
                        await req_ctx.close_normal()
                        return

                    # 模型匹配与视觉降级判定
                    provider_id, active_images, degrade_note = await self._select_model_and_prepare_vision(
                        event=event,
                        candidate_images=target.images,
                    )

                    if degrade_note:
                        target.degrade_note = (target.degrade_note + "\n" + degrade_note).strip()

                    # 构造 Prompt（传入 max_punchlines 与 analysis_detail）
                    prompt_text = build_user_prompt(
                        target_description=target.description,
                        text_content=target.text_content,
                        supplementary_text=target.supplementary_text,
                        degrade_note=target.degrade_note,
                        max_punchlines=self.cfg.max_punchlines,
                        analysis_detail=self.cfg.analysis_detail,
                    )

                    # 调用框架现有 LLM 接口
                    llm_response = await self.context.llm_generate(
                        chat_provider_id=provider_id,
                        prompt=prompt_text,
                        image_urls=active_images if active_images else None,
                        system_prompt=SYSTEM_PROMPT,
                    )

                    reply_text = (getattr(llm_response, "completion_text", "") or "").strip()
                    if not reply_text:
                        yield event.plain_result("笑点解析未能生成有效结果，请稍后再试。")
                        await req_ctx.close_normal()
                        return

                    has_media_text_content = any(
                        k in target.text_content
                        for k in ("标题:", "作者/UP主:", "AI视频摘要:", "【视频理解与转述】", "UP主简介")
                    ) or any(
                        kw in target.supplementary_text
                        for kw in ("台词", "字幕", "对话内容")
                    )

                    is_degraded_video = (
                        target.target_type == "media"
                        and not target.temp_files
                        and not has_media_text_content
                        and (
                            "未能成功读取视频画面" in target.text_content
                            or "未能读取或抽取视频内部动态画面" in target.degrade_note
                            or "视频读取失败" in target.text_content
                            or "系统失败原因" in target.degrade_note
                        )
                    )
                    formatted_reply = format_punchline_response(
                        reply_text,
                        is_degraded_video=is_degraded_video,
                        max_punchlines=self.cfg.max_punchlines,
                        user_id=sender_nick,
                    )

                    # 输出解析结果
                    yield event.plain_result(formatted_reply)

                    # 正常流程生命周期：必须等待模型（LLM）完成图片/多模态输入读取并生成回复后，方可删除任务目录与临时文件
                    if target and target.temp_files:
                        for tf in target.temp_files:
                            req_ctx.add_temp_file(tf)
                    await req_ctx.close_normal()

            except asyncio.CancelledError:
                logger.info("[笑点解析] 任务被外部取消，开始取消后台任务并清理临时资源")
                if target and target.temp_files:
                    for tf in target.temp_files:
                        req_ctx.add_temp_file(tf)
                await req_ctx.cancel_and_cleanup()
                raise
            except asyncio.TimeoutError:
                if target and target.temp_files:
                    for tf in target.temp_files:
                        req_ctx.add_temp_file(tf)
                await req_ctx.cancel_and_cleanup()
                yield event.plain_result(f"笑点解析耗时已超过 {self.cfg.timeout_seconds} 秒上限，任务已自动终止。")
            except Exception as e:
                logger.error(f"[笑点解析] 执行失败: {e}", exc_info=True)
                if target and target.temp_files:
                    for tf in target.temp_files:
                        req_ctx.add_temp_file(tf)
                await req_ctx.cancel_and_cleanup()
                yield event.plain_result(f"笑点解析遇到错误: {e}")
            finally:
                # 兜底确保请求级资源与目录清理
                if target and target.temp_files:
                    for tf in target.temp_files:
                        req_ctx.add_temp_file(tf)
                await req_ctx.ensure_cleanup()
        finally:
            # 确保全局执行锁被正确释放
            if acquired and self._lock.locked():
                self._lock.release()

    # ================= 辅助工具方法 =================
    def _parse_history_args(self, text: str) -> tuple[bool, int]:
        """判定文本中是否包含「最近N条/前N条」的意图与数量（支持阿拉伯数字与中文数字）"""
        m = RECENT_CMD_PATTERN.search(text)
        if m:
            val_str = m.group(1).strip()
            if val_str.isdigit():
                c = int(val_str)
                return True, max(2, min(50, c))
            elif val_str in CN_NUM_MAP:
                c = CN_NUM_MAP[val_str]
                return True, max(2, min(50, c))

        # 宽松兜底：包含“最近/前/历史/近期”且包含“条”
        if ("最近" in text or "前" in text or "历史" in text or "近期" in text) and "条" in text:
            return True, self.cfg.max_history_count
        if "历史" in text or "近期" in text:
            return True, self.cfg.max_history_count

        return False, self.cfg.max_history_count

    async def _select_model_and_prepare_vision(
        self,
        event: AstrMessageEvent,
        candidate_images: list[str],
    ) -> tuple[str, list[str], str]:
        """
        确定调用的 provider_id，并根据是否具备视觉能力做出明确降级
        返回: (provider_id, 传给模型的图片列表, 降级提示说明)
        """
        umo = event.unified_msg_origin
        target_provider_id = self.cfg.model_provider_id

        # 若配置了指定 provider，校验是否存在
        if target_provider_id and target_provider_id.lower() != "auto":
            prov = self.context.get_provider_by_id(target_provider_id)
            if not prov:
                all_provs = self.context.get_all_providers()
                prov = next(
                    (
                        p
                        for p in all_provs
                        if (getattr(p, "meta", lambda: None)() and p.meta().id == target_provider_id)
                        or (getattr(p, "id", None) == target_provider_id)
                    ),
                    None,
                )
            if prov:
                if candidate_images and not self._is_visual_provider(prov):
                    return (
                        target_provider_id,
                        [],
                        "（注意：指定模型不具备视觉多模态能力，已降级为纯文本分析，未传入图片画面）",
                    )
                return target_provider_id, candidate_images, ""
            else:
                logger.warning(
                    f"[笑点解析] 指定的模型提供商 {target_provider_id} 未在系统中找到或未启用，已自动切换为自动选择"
                )

        # auto 模式：优先取当前会话绑定的 provider
        session_prov = await self.context.get_using_provider_async(umo)
        if session_prov and not candidate_images:
            return session_prov.meta().id, [], ""

        if session_prov and candidate_images:
            if self._is_visual_provider(session_prov):
                return session_prov.meta().id, candidate_images, ""

            # 当前会话 provider 无视觉能力，寻找系统已配置的其他多模态模型 (如 Gemini)
            all_provs = self.context.get_all_providers()
            visual_prov = next((p for p in all_provs if self._is_visual_provider(p)), None)
            if visual_prov:
                logger.info(
                    f"[笑点解析] 当前会话模型 {session_prov.meta().id} 不支持视觉，已自动切换至视觉模型 {visual_prov.meta().id}"
                )
                return visual_prov.meta().id, candidate_images, ""

            # 全局未找到视觉模型，明确降级
            logger.warning("[笑点解析] 系统中未配置任何多模态视觉模型，已降级为纯文本分析")
            return (
                session_prov.meta().id,
                [],
                "（注意：当前系统未配置支持看图的视觉模型，已降级为纯文本分析模式，严禁编造未看见的画面）",
            )

        # 兜底
        all_provs = self.context.get_all_providers()
        if all_provs:
            p = all_provs[0]
            if candidate_images and not self._is_visual_provider(p):
                return p.meta().id, [], "（注意：默认模型不具备视觉能力，已降级为纯文本分析）"
            return p.meta().id, candidate_images, ""

        raise RuntimeError("系统中未找到任何可用的模型提供商 (Provider)")

    def _is_visual_provider(self, provider: Any) -> bool:
        """判定模型提供商是否支持视觉多模态输入（优先根据 modalities 是否包含 image 判定）"""
        if not provider:
            return False

        # 1. 优先提取 Provider 的 modalities 配置
        modalities = None
        # 1.1 从 provider.provider_config 字典获取（AstrBot 核心 Provider 标准配置，如 deepseek-flash 默认声明带 image）
        if hasattr(provider, "provider_config") and isinstance(provider.provider_config, dict):
            modalities = provider.provider_config.get("modalities")

        # 1.2 从 provider 直接属性获取
        if modalities is None:
            modalities = getattr(provider, "modalities", None)

        # 1.3 从 provider.meta() 获取
        if modalities is None and hasattr(provider, "meta") and callable(provider.meta):
            try:
                meta = provider.meta()
                if meta:
                    modalities = getattr(meta, "modalities", None)
                    if modalities is None and hasattr(meta, "provider_config") and isinstance(meta.provider_config, dict):
                        modalities = meta.provider_config.get("modalities")
            except Exception:
                pass

        # 1.4 若 provider 自身是 dict（如直接传入配置项）
        if modalities is None and isinstance(provider, dict):
            modalities = provider.get("modalities")

        # 若提取到 modalities，优先按其是否包含 'image'（不区分大小写）判定
        if modalities is not None and isinstance(modalities, (list, tuple, set)):
            if any(str(m).strip().lower() == "image" for m in modalities):
                return True
            # 显式配置了模态列表但不含 image，确认为非视觉模型
            return False

        # 2. 兜底：未配置 modalities 时，根据模型名称与 Provider ID 关键词判定
        model = (getattr(provider, "get_model", lambda: "")() or "").lower()
        if not model and isinstance(provider, dict):
            model = str(provider.get("model", "")).lower()

        pid = ""
        if hasattr(provider, "meta") and callable(provider.meta):
            try:
                meta = provider.meta()
                if meta:
                    pid = getattr(meta, "id", "") or ""
            except Exception:
                pass
        if not pid:
            pid = getattr(provider, "id", "") or ""
        if not pid and isinstance(provider, dict):
            pid = str(provider.get("id", "") or "")
        pid = str(pid).lower()

        visual_keywords = ("gemini", "vision", "vl", "gpt-4o", "4o-mini", "claude-3", "qwen-vl")
        return any(k in model or k in pid for k in visual_keywords)
