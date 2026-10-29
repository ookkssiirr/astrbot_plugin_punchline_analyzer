from __future__ import annotations

import collections
import datetime
from dataclasses import dataclass, field
from typing import Any, Callable

from astrbot.api import logger
from astrbot.core.message.components import At, BaseMessageComponent, Forward, Image, Json, Plain, Reply, Video
from astrbot.core.platform.astr_message_event import AstrMessageEvent


@dataclass
class CachedMessage:
    message_id: str
    user_id: str
    nickname: str
    timestamp: int
    text: str
    images: list[str] = field(default_factory=list)
    reply_quote: str = ""
    has_rich_media: bool = False


class GroupHistoryManager:
    """按群隔离的滑动窗口消息缓存与协议端历史拉取器"""

    def __init__(self, max_buffer_per_group: int = 100) -> None:
        self.max_buffer_per_group = max_buffer_per_group
        # group_id -> deque of CachedMessage
        self._cache: dict[str, collections.deque[CachedMessage]] = {}

    def record_event(self, event: AstrMessageEvent) -> None:
        """记录群内收到的用户发言"""
        try:
            group_id = str(event.get_group_id() or "").strip()
            if not group_id:
                return

            self_id = str(event.get_self_id() or "").strip()
            sender_id = str(event.get_sender_id() or "").strip()
            if sender_id and self_id and sender_id == self_id:
                # 排除机器人自己的发言
                return

            sender_name = event.get_sender_name() or sender_id or "某群友"
            timestamp = getattr(event.message_obj, "timestamp", 0) or int(datetime.datetime.now().timestamp())
            msg_id = str(getattr(event.message_obj, "message_id", "") or "")

            text_parts: list[str] = []
            images: list[str] = []
            reply_quote = ""
            has_rich = False

            chain = event.get_messages() or []
            for seg in chain:
                stype = type(seg).__name__
                if isinstance(seg, Plain) or stype in ("Plain", "MockPlain"):
                    text_parts.append(getattr(seg, "text", ""))
                elif isinstance(seg, Image) or stype in ("Image", "MockImage"):
                    img_ref = getattr(seg, "url", None) or getattr(seg, "file", None) or getattr(seg, "path", None)
                    if img_ref:
                        images.append(str(img_ref))
                        has_rich = True
                elif isinstance(seg, At) or stype in ("At", "MockAt"):
                    name = getattr(seg, "name", "") or getattr(seg, "qq", "")
                    text_parts.append(f"@{name} ")
                elif isinstance(seg, Reply) or stype in ("Reply", "MockReply"):
                    quote_text = getattr(seg, "message_str", "") or ""
                    sender_nick = getattr(seg, "sender_nickname", "") or str(getattr(seg, "sender_id", "") or "")
                    reply_quote = f"(引用 {sender_nick}: {quote_text[:60]})" if quote_text else f"(引用 {sender_nick})"
                elif isinstance(seg, (Video, Forward, Json)) or stype in ("Video", "MockVideo", "Forward", "MockForward", "Json", "MockJson"):
                    has_rich = True

            full_text = "".join(text_parts).strip()
            if not full_text and images:
                full_text = "[图片]"

            item = CachedMessage(
                message_id=msg_id,
                user_id=sender_id,
                nickname=sender_name,
                timestamp=timestamp,
                text=full_text,
                images=images,
                reply_quote=reply_quote,
                has_rich_media=has_rich,
            )

            if group_id not in self._cache:
                self._cache[group_id] = collections.deque(maxlen=self.max_buffer_per_group)
            self._cache[group_id].append(item)
        except Exception as e:
            logger.debug(f"[笑点解析] 缓存群消息异常: {e}")

    async def get_recent_history(
        self,
        event: AstrMessageEvent,
        count: int = 30,
        max_images: int = 10,
    ) -> tuple[str, list[str], int]:
        """
        获取当前群最近的有效消息（排除当前命令和机器人自身回复）
        返回: (格式化文本, 提取的图片列表, 实际消息条数)
        """
        group_id = str(event.get_group_id() or "").strip()
        if not group_id:
            return "（非群聊环境，无法读取群历史记录）", [], 0

        self_id = str(event.get_self_id() or "").strip()
        current_msg_id = str(getattr(event.message_obj, "message_id", "") or "")

        messages_raw: list[CachedMessage] = []

        # 优先尝试从协议端拉取 (SnowLuma / OneBot v11)
        call_action = self._get_call_action(event)
        if call_action:
            try:
                # SnowLuma 驱动规范：只传 group_id 与 count，不传 reverseOrder
                fetch_count = min(count + 10, 50)
                res = await call_action("get_group_msg_history", group_id=int(group_id), count=fetch_count)
                if isinstance(res, dict) and "messages" in res:
                    raw_list = res.get("messages", [])
                    if isinstance(raw_list, list) and raw_list:
                        for m in raw_list:
                            parsed = self._convert_onebot_raw_message(m, self_id)
                            if parsed:
                                messages_raw.append(parsed)
            except Exception as e:
                logger.debug(f"[笑点解析] 协议端 get_group_msg_history 调用失败或不支持，将使用本地内存缓存: {e}")

        # 若协议端拉取为空，使用本插件内存滑动窗口
        if not messages_raw and group_id in self._cache:
            messages_raw = list(self._cache[group_id])

        # 过滤与去重：排除机器人自己、排除当前命令、排除命令本身
        filtered: list[CachedMessage] = []
        for m in messages_raw:
            if m.user_id and self_id and m.user_id == self_id:
                continue
            if current_msg_id and m.message_id and m.message_id == current_msg_id:
                continue
            # 排除笑点解析指令本身
            if "笑点解析" in m.text or "笑点分析" in m.text:
                continue
            filtered.append(m)

        # 截取最近 count 条
        selected = filtered[-count:] if len(filtered) > count else filtered
        actual_count = len(selected)

        if actual_count == 0:
            return "（当前群近期尚未记录到其他群友发言）", [], 0

        transcript_lines: list[str] = [
            f"=== 当前群聊最近 {actual_count} 条发言（按时间正序）==="
        ]
        collected_images: list[str] = []

        for idx, m in enumerate(selected, 1):
            time_str = ""
            if m.timestamp > 0:
                try:
                    time_str = datetime.datetime.fromtimestamp(m.timestamp).strftime("%H:%M:%S")
                except Exception:
                    pass
            time_prefix = f"[{time_str}] " if time_str else ""
            
            line_content = m.text
            # 关联图片
            if m.images:
                img_tags = []
                for img_url in m.images:
                    if len(collected_images) < max_images:
                        collected_images.append(img_url)
                        img_tags.append(f"[图片{len(collected_images)}]")
                    else:
                        img_tags.append("[图片]")
                if img_tags:
                    line_content = (line_content + " " + " ".join(img_tags)).strip()

            reply_part = f" {m.reply_quote}" if m.reply_quote else ""
            transcript_lines.append(f"{idx}. {time_prefix}{m.nickname}:{reply_part} {line_content}")

        return "\n".join(transcript_lines), collected_images, actual_count

    def get_most_recent_candidate(
        self,
        group_id: str | int,
        self_id: str,
        current_msg_id: str = "",
    ) -> CachedMessage | None:
        """消解“这个/刚才那个”指代：寻找本群最近一条含富媒体或有实质发言的群友消息"""
        gid = str(group_id).strip()
        if gid not in self._cache:
            return None

        # 从最新往最旧遍历
        for item in reversed(self._cache[gid]):
            if item.user_id and self_id and item.user_id == self_id:
                continue
            if current_msg_id and item.message_id and item.message_id == current_msg_id:
                continue
            if "笑点解析" in item.text or "笑点分析" in item.text:
                continue
            # 优先匹配含有富媒体（图片/视频/分享）的消息
            if item.has_rich_media or item.images or item.text:
                return item
        return None

    def _convert_onebot_raw_message(self, raw: dict[str, Any], self_id: str) -> CachedMessage | None:
        if not isinstance(raw, dict):
            return None

        sender = raw.get("sender", {})
        user_id = str(raw.get("user_id") or (sender.get("user_id") if isinstance(sender, dict) else "") or "")
        if user_id and self_id and user_id == self_id:
            return None

        nickname = ""
        if isinstance(sender, dict):
            nickname = sender.get("card") or sender.get("nickname") or user_id
        if not nickname:
            nickname = user_id or "某群友"

        msg_id = str(raw.get("message_id", "") or "")
        timestamp = raw.get("time", 0) or 0

        content = raw.get("message")
        text_parts: list[str] = []
        images: list[str] = []
        has_rich = False

        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for seg in content:
                if isinstance(seg, dict):
                    stype = seg.get("type")
                    sdata = seg.get("data", {})
                    if stype in ("text", "plain"):
                        text_parts.append(str(sdata.get("text", "")))
                    elif stype == "image":
                        url = sdata.get("url") or sdata.get("file")
                        if url:
                            images.append(str(url))
                            has_rich = True
                    elif stype == "at":
                        text_parts.append(f"@{sdata.get('qq', '')} ")
                    elif stype in ("video", "forward", "json"):
                        has_rich = True

        full_text = "".join(text_parts).strip()
        return CachedMessage(
            message_id=msg_id,
            user_id=user_id,
            nickname=nickname,
            timestamp=timestamp,
            text=full_text,
            images=images,
            has_rich_media=has_rich,
        )

    def _get_call_action(self, event: AstrMessageEvent) -> Callable[..., Any] | None:
        try:
            bot = getattr(event, "bot", None)
            if bot and hasattr(bot, "call_action") and callable(bot.call_action):
                return bot.call_action
            api = getattr(bot, "api", None)
            if api and hasattr(api, "call_action") and callable(api.call_action):
                return api.call_action
        except Exception:
            pass
        return None
