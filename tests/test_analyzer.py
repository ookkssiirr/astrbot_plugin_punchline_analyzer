from __future__ import annotations

import asyncio
import datetime
import os
import shutil
import tempfile
import unittest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astrbot_plugin_punchline_analyzer.config import PluginConfig
from astrbot_plugin_punchline_analyzer.extractor import TargetExtractor
from astrbot_plugin_punchline_analyzer.forward_reader import ForwardMessageReader
from astrbot_plugin_punchline_analyzer.history_manager import CachedMessage, GroupHistoryManager
from astrbot_plugin_punchline_analyzer.main import PunchlineAnalyzerPlugin
from astrbot.core.platform.message_type import MessageType
from astrbot_plugin_punchline_analyzer.parser_bridge import (
    ParserBridge,
    RequestContext,
    extract_url_from_json_segment,
    extract_urls_from_text,
)
from astrbot_plugin_punchline_analyzer.formatter import (
    format_punchline_response,
    validate_punchline_format,
)
from astrbot_plugin_punchline_analyzer.prompt import build_user_prompt


# ---------- MOCK ASTRBOT COMPONENTS ----------
class MockPlain:
    def __init__(self, text: str):
        self.text = text


class MockAt:
    def __init__(self, qq: str | int, name: str = ""):
        self.qq = str(qq)
        self.name = name


class MockImage:
    def __init__(self, url: str):
        self.url = url
        self.file = url
        self.path = url


class MockVideo:
    def __init__(self, url: str = "", file: str = "", path: str = ""):
        self.url = url
        self.file = file
        self.path = path
        self.type = "video"


class MockBot:
    def __init__(self, actions: dict[str, Any] = None):
        self.actions = actions or {}

    async def call_action(self, action: str, **kwargs):
        if action in self.actions:
            handler = self.actions[action]
            if callable(handler):
                res = handler(**kwargs)
                if asyncio.iscoroutine(res):
                    return await res
                return res
            return handler
        return None


class MockReply:
    def __init__(self, message_str: str, chain: list[Any] = None, sender_id: str = "123456", sender_nickname: str = "测试群友", id: str = "msg_reply_001"):
        self.id = id
        self.message_str = message_str
        self.chain = chain or []
        self.sender_id = sender_id
        self.sender_nickname = sender_nickname


class MockForward:
    def __init__(self, forward_id: str):
        self.id = forward_id


class MockJson:
    def __init__(self, data: dict):
        self.data = data


class MockMessageObj:
    def __init__(self, message_id: str = "msg_001", timestamp: int = 1700000000):
        self.message_id = message_id
        self.timestamp = timestamp


class MockAstrMessageEvent:
    def __init__(
        self,
        messages: list[Any],
        sender_id: str = "10001",
        sender_name: str = "张三",
        self_id: str = "99999",
        group_id: str = "88888",
        message_type: Any = MessageType.GROUP_MESSAGE,
        message_id: str = "msg_001",
    ):
        self.messages = messages
        self._sender_id = sender_id
        self._sender_name = sender_name
        self._self_id = self_id
        self._group_id = group_id
        self._message_type = message_type
        self.message_obj = MockMessageObj(message_id=message_id)
        self.stopped = False
        self.results: list[Any] = []
        self.unified_msg_origin = f"aiocqhttp:{group_id}:{sender_id}"
        self.bot = None

    def get_messages(self):
        return self.messages

    def get_sender_id(self):
        return self._sender_id

    def get_sender_name(self):
        return self._sender_name

    def get_self_id(self):
        return self._self_id

    def get_group_id(self):
        return self._group_id

    def get_message_type(self):
        return self._message_type

    def stop_event(self):
        self.stopped = True

    def plain_result(self, text: str):
        return text


class MockProviderMeta:
    def __init__(self, provider_id: str):
        self.id = provider_id


class MockProvider:
    def __init__(self, provider_id: str, model: str, provider_config: dict | None = None):
        self._id = provider_id
        self._model = model
        self.provider_config = provider_config if provider_config is not None else {}

    def meta(self):
        return MockProviderMeta(self._id)

    def get_model(self):
        return self._model


class MockLLMResponse:
    def __init__(self, text: str):
        self.completion_text = text


class MockContext:
    def __init__(self):
        self.providers = [
            MockProvider("deepseek/deepseek-flash", "deepseek-flash"),
            MockProvider("Hagimi3.8/gemini-3.8-flash", "gemini-3.8-flash"),
        ]
        self.stars = {}
        self.llm_calls: list[dict[str, Any]] = []

    def get_registered_star(self, name: str):
        return self.stars.get(name)

    def get_all_providers(self):
        return self.providers

    def get_provider_by_id(self, pid: str):
        return next((p for p in self.providers if p.meta().id == pid), None)

    async def get_using_provider_async(self, umo: str):
        return self.providers[0]  # default deepseek (text-only)

    async def llm_generate(self, chat_provider_id: str, prompt: str, image_urls: list[str] = None, system_prompt: str = None):
        self.llm_calls.append({
            "provider": chat_provider_id,
            "prompt": prompt,
            "image_urls": image_urls or [],
            "system_prompt": system_prompt,
        })
        return MockLLMResponse(f"[Mock Analysis by {chat_provider_id}] 笑点在于反差萌，非常精彩！")


# ---------- UNIT TESTS ----------
class TestPunchlineAnalyzer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context = MockContext()
        # 默认套件关闭冷却，避免用例内密集连续发送被 15s 冷却拦截；专项测试中显式配置冷却
        self.plugin = PunchlineAnalyzerPlugin(self.context, {"user_cooldown_seconds": 0.0, "group_cooldown_seconds": 0.0})

    # 1. 触发规则测试：@当前机器人 + 关键词 触发
    async def test_trigger_valid_mention(self):
        event = MockAstrMessageEvent(
            messages=[
                MockAt(qq="99999", name="Bot"),
                MockPlain(" 帮我对这个做个笑点解析"),
            ],
            self_id="99999",
        )
        responses = []
        async for r in self.plugin.on_message(event):
            responses.append(r)

        self.assertTrue(event.stopped, "命中关键词且正确@应停止事件传播以去重")
        self.assertTrue(len(responses) > 0, "应生成回复")

    # 2. 触发规则测试：无@ 不触发
    async def test_trigger_no_mention(self):
        event = MockAstrMessageEvent(
            messages=[MockPlain("这个笑点解析一下")],
            self_id="99999",
        )
        responses = []
        async for r in self.plugin.on_message(event):
            responses.append(r)

        self.assertFalse(event.stopped, "无@时不应停止事件")
        self.assertEqual(len(responses), 0, "无@时不应触发回复")

    # 3. 触发规则测试：@其他人 不触发
    async def test_trigger_mention_other(self):
        event = MockAstrMessageEvent(
            messages=[
                MockAt(qq="12345", name="其他群友"),
                MockPlain(" 笑点解析"),
            ],
            self_id="99999",
        )
        responses = []
        async for r in self.plugin.on_message(event):
            responses.append(r)

        self.assertFalse(event.stopped)
        self.assertEqual(len(responses), 0)

    # 4. 触发规则测试：仅引用或转发内容中出现“笑点解析”，当前消息无关键词，不触发
    async def test_trigger_only_quoted_has_keyword(self):
        event = MockAstrMessageEvent(
            messages=[
                MockAt(qq="99999", name="Bot"),
                MockReply(message_str="昨天的笑点解析真搞笑", sender_id="111"),
                MockPlain(" 查一下明天天气"),  # 当前用户文字中无“笑点解析”
            ],
            self_id="99999",
        )
        responses = []
        async for r in self.plugin.on_message(event):
            responses.append(r)

        self.assertFalse(event.stopped, "仅引用含关键词时不应误触发")
        self.assertEqual(len(responses), 0)

    # 5. 触发规则测试：普通文字中出现昵称，非消息段@，不触发
    async def test_trigger_text_nickname_not_at_segment(self):
        event = MockAstrMessageEvent(
            messages=[MockPlain("@机器人 来个笑点解析")],  # 纯文字，非 MockAt
            self_id="99999",
        )
        responses = []
        async for r in self.plugin.on_message(event):
            responses.append(r)

        self.assertFalse(event.stopped)
        self.assertEqual(len(responses), 0)

    # 6. 循环防护：机器人自身消息不处理
    async def test_ignore_self_messages(self):
        event = MockAstrMessageEvent(
            messages=[
                MockAt(qq="99999"),
                MockPlain(" 笑点解析"),
            ],
            sender_id="99999",  # 机器人自己
            self_id="99999",
        )
        responses = []
        async for r in self.plugin.on_message(event):
            responses.append(r)

        self.assertFalse(event.stopped)
        self.assertEqual(len(responses), 0)

    # 7. 独立入口：/笑点解析 最近30条 指令测试
    async def test_slash_command_recent_history(self):
        # 预先录入 5 条历史发言
        for i in range(5):
            ev = MockAstrMessageEvent(
                messages=[MockPlain(f"群聊测试发言 {i + 1}")],
                sender_id=f"user_{i}",
                sender_name=f"群友_{i}",
                group_id="88888",
                message_id=f"msg_h_{i}",
            )
            self.plugin.history_manager.record_event(ev)

        cmd_event = MockAstrMessageEvent(
            messages=[MockPlain("/笑点解析 最近30条")],
            sender_id="user_admin",
            group_id="88888",
        )
        responses = []
        async for r in self.plugin.cmd_punchline(cmd_event, arg="最近30条"):
            responses.append(r)

        self.assertTrue(cmd_event.stopped)
        self.assertTrue(len(responses) > 0)
        self.assertIn("反差萌", responses[0])

    # 8. 群隔离测试：不混入其他群的发言
    async def test_group_isolation(self):
        # 群 A 发言
        ev_a = MockAstrMessageEvent(
            messages=[MockPlain("群A的私密话题")],
            sender_id="user_a",
            group_id="group_111",
            message_id="msg_a",
        )
        self.plugin.history_manager.record_event(ev_a)

        # 群 B 查询
        ev_b = MockAstrMessageEvent(
            messages=[MockPlain("/笑点解析 最近30条")],
            sender_id="user_b",
            group_id="group_222",
            message_id="msg_b",
        )
        text, imgs, count = await self.plugin.history_manager.get_recent_history(ev_b, count=30)
        self.assertNotIn("群A的私密话题", text, "群B不应看到群A的历史记录")
        self.assertEqual(count, 0)

    # 9. 引用图片与多模态模型选择降级验证
    async def test_quoted_image_model_selection(self):
        reply_seg = MockReply(
            message_str="",
            chain=[MockImage("http://example.com/test.jpg")],
            sender_nickname="表情包大师",
        )
        event = MockAstrMessageEvent(
            messages=[
                MockAt(qq="99999"),
                reply_seg,
                MockPlain(" 笑点解析"),
            ],
            self_id="99999",
        )
        responses = []
        async for r in self.plugin.on_message(event):
            responses.append(r)

        self.assertTrue(event.stopped)
        self.assertTrue(len(responses) > 0)
        # 会话默认是 deepseek-flash（纯文本），但因为有图片，应自动切换到 gemini-3.8-flash 视觉模型
        self.assertIn("Hagimi3.8/gemini-3.8-flash", responses[0])

    # 10. URL 提取与 JSON 卡片解析测试
    def test_url_and_card_extraction(self):
        # 1. 文本中的 BV 号与链接
        text = "快看这个 BV1GbJ26BEUh 还有 https://b23.tv/xyz123 超好笑"
        urls = extract_urls_from_text(text)
        self.assertEqual(len(urls), 2)
        self.assertIn("https://www.bilibili.com/video/BV1GbJ26BEUh", urls)

        # 2. JSON 小程序卡片
        card_data = {
            "app": "com.tencent.miniapp",
            "meta": {
                "detail_1": {
                    "qqdocurl": "https://www.bilibili.com/video/BV1GbJ26BEUh",
                    "title": "搞笑视频"
                }
            }
        }
        extracted_card_url = extract_url_from_json_segment(card_data)
        self.assertEqual(extracted_card_url, "https://www.bilibili.com/video/BV1GbJ26BEUh")

    # 11. 合并转发读取与深度/条数截断测试
    async def test_forward_reader_limits(self):
        mock_nodes = [
            {"sender": {"nickname": f"群友{i}"}, "time": 1700000000 + i, "message": [{"type": "text", "data": {"text": f"段子{i}"}}]}
            for i in range(40)
        ]

        async def mock_call_action(action, **kwargs):
            if action == "get_forward_msg":
                return {"messages": mock_nodes}
            return None

        reader = ForwardMessageReader(
            call_action=mock_call_action,
            max_depth=2,
            max_messages=15,  # 限制最多 15 条
            max_images=5,
        )
        transcript, images = await reader.read_forward("forward_res_999")
        self.assertIn("共解析 15 条消息", transcript)
        self.assertIn("截断", transcript)

    # 12. 引用视频理解插件联动测试 (astrbot_plugin_video_reference_vision)
    async def test_quoted_video_via_video_reference_vision(self):
        class MockVideoVisionPlugin:
            def __init__(self):
                self.caption_context_cache = type("MockCache", (), {
                    "_items": {
                        ("aiocqhttp:88888:10001", "vid_msg_101"): type("Item", (), {
                            "caption_text": "一只猫试图偷吃鱼干，结果把鱼缸盖子打翻了，猫被扣在里面。"
                        })()
                    }
                })()
                self.video_cache = type("MockVCache", (), {
                    "get": lambda session_id, message_id: None
                })()

        # 注册 mock 视频插件
        self.context.stars["astrbot_plugin_video_reference_vision"] = type("StarMD", (), {
            "star_cls": MockVideoVisionPlugin()
        })()

        reply_seg = MockReply(
            message_str="看这个视频笑死了",
            chain=[MockPlain("看这个视频笑死了")],
            sender_nickname="猫猫视频博主",
        )
        reply_seg.id = "vid_msg_101"

        event = MockAstrMessageEvent(
            messages=[
                MockAt(qq="99999"),
                reply_seg,
                MockPlain(" 笑点解析"),
            ],
            self_id="99999",
        )

        responses = []
        async for r in self.plugin.on_message(event):
            responses.append(r)

        self.assertTrue(event.stopped)
        self.assertTrue(len(responses) > 0)

    # 13. 并发测试：多人交错并发请求深度断言 Prompt、图片及群标识归属
    async def test_concurrent_interleaved_requests_no_cross_talk(self):
        self.context.llm_calls.clear()

        async def make_user_request(user_id: str, group_id: str, content: str, img_url: str):
            ev = MockAstrMessageEvent(
                messages=[
                    MockAt(qq="99999"),
                    MockReply(
                        message_str=f"{user_id} 发送的专属内容: {content}",
                        chain=[MockImage(img_url)],
                        sender_nickname=f"用户_{user_id}",
                    ),
                    MockPlain(" 笑点解析"),
                ],
                sender_id=user_id,
                group_id=group_id,
                self_id="99999",
                message_id=f"msg_{user_id}",
            )
            res = []
            async for r in self.plugin.on_message(ev):
                res.append(r)
            return user_id, group_id, res

        # 启动 2 个用户同时并发请求（在有限排队允许范围内）
        results = await asyncio.gather(
            make_user_request("u1", "group_101", "程序员写出死循环", "http://img/1.png"),
            make_user_request("u2", "group_202", "产品经理要求五彩斑斓的黑", "http://img/2.png"),
        )

        for uid, gid, res_list in results:
            self.assertTrue(len(res_list) > 0)
            self.assertIn("反差萌", res_list[0])

        # 深度断言：检查传给 LLM 的每一次调用的 Prompt、图片与群/用户归属隔离
        self.assertEqual(len(self.context.llm_calls), 2, "必须产生两次完全独立的 LLM 生成调用")

        # 校验请求 1 (u1, group_101) 的调用归属
        call_u1 = next((c for c in self.context.llm_calls if "程序员写出死循环" in c["prompt"]), None)
        self.assertIsNotNone(call_u1, "必须存在包含 u1 专属内容的 Prompt")
        self.assertIn("用户_u1", call_u1["prompt"])
        self.assertNotIn("用户_u2", call_u1["prompt"], "u1 的 Prompt 严禁串入 u2 的标识")
        self.assertNotIn("产品经理要求五彩斑斓的黑", call_u1["prompt"], "u1 的 Prompt 严禁串入 u2 的内容")
        self.assertEqual(call_u1["image_urls"], ["http://img/1.png"], "u1 的视觉素材必须严格归属图片 1")

        # 校验请求 2 (u2, group_202) 的调用归属
        call_u2 = next((c for c in self.context.llm_calls if "产品经理要求五彩斑斓的黑" in c["prompt"]), None)
        self.assertIsNotNone(call_u2, "必须存在包含 u2 专属内容的 Prompt")
        self.assertIn("用户_u2", call_u2["prompt"])
        self.assertNotIn("用户_u1", call_u2["prompt"], "u2 的 Prompt 严禁串入 u1 的标识")
        self.assertNotIn("程序员写出死循环", call_u2["prompt"], "u2 的 Prompt 严禁串入 u1 的内容")
        self.assertEqual(call_u2["image_urls"], ["http://img/2.png"], "u2 的视觉素材必须严格归属图片 2")

    # 14. 有限排队与过载保护测试
    async def test_queue_overflow_protection(self):
        # 设置排队上限为 1
        self.plugin.cfg.max_wait_queue = 1
        self.plugin.cfg.queue_wait_seconds = 0.5

        # 占住锁
        await self.plugin._lock.acquire()
        try:
            # 制造 2 个排队请求，第 2 个应当被有限排队拒绝
            async def fire_request(req_id: str):
                ev = MockAstrMessageEvent(
                    messages=[MockAt(qq="99999"), MockPlain(" 笑点解析")],
                    message_id=f"req_{req_id}",
                )
                res = []
                async for r in self.plugin.on_message(ev):
                    res.append(r)
                return res

            t1 = asyncio.create_task(fire_request("t1"))
            await asyncio.sleep(0.05)
            # t1 正在排队等待锁（waiting_count=1）
            # 现在再发 t2，应当直接触发排队过载拒绝
            res2 = await fire_request("t2")
            self.assertTrue(any("排队人数较多" in str(r) for r in res2))

            # 等待 t1 超时
            res1 = await t1
            self.assertTrue(any("排队等待超过" in str(r) for r in res1))
        finally:
            self.plugin._lock.release()

    # 15. 视频解析失败或缺失目标时的正常反馈与防捏造测试
    async def test_video_parsing_failure_graceful_feedback(self):
        # 空目标反馈
        ev_empty = MockAstrMessageEvent(
            messages=[MockAt(qq="99999"), MockPlain(" 笑点解析")],
            self_id="99999",
        )
        res_empty = []
        async for r in self.plugin.on_message(ev_empty):
            res_empty.append(r)

        self.assertTrue(len(res_empty) > 0)
        self.assertIn("未能识别到要解析的笑点目标", res_empty[0])

    # 16. 异常与取消时严格清理临时文件测试
    async def test_cleanup_on_exception_and_cancellation(self):
        # 创建一个真实的临时文件模拟抽帧结果
        test_temp_file = Path(tempfile.gettempdir()) / "test_frame_cleanup.jpg"
        test_temp_file.write_text("fake frame data")
        self.assertTrue(test_temp_file.exists())

        # 构造异常
        async def mock_faulty_llm(**kwargs):
            raise RuntimeError("LLM 接口突发异常测试")

        self.context.llm_generate = mock_faulty_llm

        reply_seg = MockReply(message_str="测试异常清理", chain=[MockPlain("测试")])
        ev = MockAstrMessageEvent(
            messages=[MockAt(qq="99999"), reply_seg, MockPlain(" 笑点解析")],
            self_id="99999",
        )

        # 注入 temp_files
        original_extract = self.plugin.extractor.extract_target

        async def mock_extract(*args, **kwargs):
            t = await original_extract(*args, **kwargs)
            t.temp_files.append(str(test_temp_file))
            return t

        self.plugin.extractor.extract_target = mock_extract

        res = []
        async for r in self.plugin.on_message(ev):
            res.append(r)

        # 验证临时文件已被清理
        self.assertFalse(test_temp_file.exists(), "发生异常时临时文件必须已被清理")
        self.assertTrue(any("笑点解析遇到错误" in str(r) for r in res))

    # 17. 媒体大小限制测试（超出 max_media_mb 跳过抽帧）
    async def test_media_size_limit_protection(self):
        from astrbot_plugin_punchline_analyzer.parser_bridge import sample_video_frames

        # 创建一个 2MB 的临时虚拟视频文件
        dummy_video = Path(tempfile.gettempdir()) / "dummy_large_video.mp4"
        dummy_video.write_bytes(b"\x00" * (2 * 1024 * 1024))

        try:
            # 限制 max_media_mb 为 1MB，应当直接拦截并返回空列表
            frames = await sample_video_frames(
                video_path=dummy_video,
                output_dir=self.plugin.temp_dir,
                num_frames=4,
                max_media_mb=1,
            )
            self.assertEqual(frames, [], "超出大小上限的视频必须直接跳过抽帧")
        finally:
            if dummy_video.exists():
                dummy_video.unlink()

    # 18. 中途取消与子进程/任务目录严格清理测试
    async def test_mid_extraction_cancellation_and_cleanup(self):
        from astrbot_plugin_punchline_analyzer.parser_bridge import run_cancellable_subprocess

        # 18.1 测试子进程在中途取消时被 kill，不留孤儿进程
        cmd = ["python", "-c", "import time; time.sleep(10)"]
        subproc_task = asyncio.create_task(run_cancellable_subprocess(*cmd, timeout=10.0))
        await asyncio.sleep(0.05)
        # 取消子进程任务
        subproc_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await subproc_task

        # 18.2 测试 _process_flow 执行提取途中外部取消时，专属 task_dir 能够被完全删除
        captured_task_dirs: list[Path] = []
        original_extract = self.plugin.extractor.extract_target

        async def slow_extract(*args, **kwargs):
            td = kwargs.get("task_dir")
            if td:
                captured_task_dirs.append(td)
                # 模拟在任务目录下生成了临时半成品文件
                (td / "partial_download.tmp").write_text("in-progress bytes")
            await asyncio.sleep(5.0)  # 挂起模拟耗时网络/视频处理
            return await original_extract(*args, **kwargs)

        self.plugin.extractor.extract_target = slow_extract

        ev = MockAstrMessageEvent(
            messages=[MockAt(qq="99999"), MockPlain(" 笑点解析")],
            self_id="99999",
        )

        flow_task = asyncio.create_task(
            self.plugin._process_flow(
                event=ev,
                user_text="笑点解析",
            ).__anext__()
        )
        await asyncio.sleep(0.05)
        flow_task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await flow_task

        self.assertTrue(len(captured_task_dirs) > 0, "必须已创建任务目录")
        for td in captured_task_dirs:
            self.assertFalse(td.exists(), f"任务目录 {td} 在中途取消后必须已被彻底删除")

    # 19. 多图在聊天历史中的精确归属与时序对应测试
    async def test_image_attribution_in_chat_history(self):
        group_mgr = GroupHistoryManager(max_buffer_per_group=50)

        # 构造多轮交错群聊发言
        # 消息 1: Alice 发送猫猫图
        ev1 = MockAstrMessageEvent(
            messages=[MockPlain("看这个猫猫打哈欠"), MockImage("http://img/cat.jpg")],
            sender_id="101",
            sender_name="Alice",
            group_id="group_test_attr",
            message_id="msg_attr_1",
        )
        # 消息 2: Bob 纯文字发言，无图
        ev2 = MockAstrMessageEvent(
            messages=[MockPlain("哈哈哈哈这也太困了吧")],
            sender_id="102",
            sender_name="Bob",
            group_id="group_test_attr",
            message_id="msg_attr_2",
        )
        # 消息 3: Charlie 发送柴犬表情包
        ev3 = MockAstrMessageEvent(
            messages=[MockPlain("还有这只柴犬表情包"), MockImage("http://img/dog.jpg")],
            sender_id="103",
            sender_name="Charlie",
            group_id="group_test_attr",
            message_id="msg_attr_3",
        )
        # 消息 4: Dave 引用 Alice 并附带反应图
        reply_seg = MockReply(
            message_str="看这个猫猫打哈欠",
            sender_nickname="Alice",
            sender_id="101",
            id="msg_attr_1",
        )
        ev4 = MockAstrMessageEvent(
            messages=[reply_seg, MockPlain("笑裂开了"), MockImage("http://img/dave_reaction.png")],
            sender_id="104",
            sender_name="Dave",
            group_id="group_test_attr",
            message_id="msg_attr_4",
        )

        group_mgr.record_event(ev1)
        group_mgr.record_event(ev2)
        group_mgr.record_event(ev3)
        group_mgr.record_event(ev4)

        # 触发获取最近历史
        ev_query = MockAstrMessageEvent(
            messages=[MockPlain("/笑点解析 最近30条")],
            sender_id="999",
            group_id="group_test_attr",
            message_id="msg_query",
        )
        history_text, collected_images, count = await group_mgr.get_recent_history(
            event=ev_query,
            count=30,
            max_images=10,
        )

        self.assertEqual(count, 4)
        # 验证图片数组顺序与采集内容完全精准
        self.assertEqual(
            collected_images,
            ["http://img/cat.jpg", "http://img/dog.jpg", "http://img/dave_reaction.png"],
        )

        # 验证文本中图片标记精确归属于具体发言者
        lines = history_text.split("\n")
        alice_line = next((l for l in lines if "Alice" in l), "")
        bob_line = next((l for l in lines if "Bob" in l), "")
        charlie_line = next((l for l in lines if "Charlie" in l), "")
        dave_line = next((l for l in lines if "Dave" in l), "")

        self.assertIn("[图片1]", alice_line, "图片1必须精确归属于 Alice")
        self.assertNotIn("[图片", bob_line, "Bob 纯文字发言严禁被掺入图片标记")
        self.assertIn("[图片2]", charlie_line, "图片2必须精确归属于 Charlie")
        self.assertIn("[图片3]", dave_line, "图片3必须精确归属于 Dave")
        self.assertIn("(引用 Alice", dave_line, "Dave 的引用归属必须保留")

    # 20. 真实 OneBot v11 结构输入与回溯解析测试
    async def test_realistic_onebot_v11_inputs(self):
        from astrbot_plugin_punchline_analyzer.parser_bridge import extract_url_from_json_segment

        # 20.1 OneBot v11 原始消息段结构转换
        raw_onebot_msg = {
            "message_id": 99881,
            "time": 1700001000,
            "sender": {"user_id": 2002, "card": "群管小王", "nickname": "小王"},
            "message": [
                {"type": "text", "data": {"text": "分享一个搞笑小视频"}},
                {"type": "video", "data": {"file": "vid_abc123.mp4", "url": "http://domain.com/v.mp4"}},
            ],
        }
        converted = self.plugin.history_manager._convert_onebot_raw_message(raw_onebot_msg, self_id="99999")
        self.assertIsNotNone(converted)
        self.assertEqual(converted.nickname, "群管小王")
        self.assertEqual(converted.text, "分享一个搞笑小视频")
        self.assertTrue(converted.has_rich_media)

        # 20.2 OneBot Ark 小程序 JSON 卡片跳转链接解析
        ark_card = {
            "app": "com.tencent.miniapp",
            "meta": {
                "detail_1": {
                    "qqdocurl": "https://www.bilibili.com/video/BV1xx411c7mD",
                    "title": "搞笑视频卡片",
                }
            },
        }
        extracted_card_url = extract_url_from_json_segment(ark_card)
        self.assertEqual(extracted_card_url, "https://www.bilibili.com/video/BV1xx411c7mD")

        # 20.3 通过 OneBot get_msg 接口回溯解析未缓存的引用视频
        mock_bot = MockBot({
            "get_msg": lambda message_id: {
                "data": {
                    "message": [
                        {"type": "video", "data": {"url": "http://domain.com/fallback.mp4", "file": "fallback_file"}},
                    ]
                }
            }
        })
        ev_quoted = MockAstrMessageEvent(
            messages=[MockAt(qq="99999"), MockReply(message_str="看这个", id="99881"), MockPlain(" 笑点解析")],
            self_id="99999",
        )
        ev_quoted.bot = mock_bot

        # 调用 resolve_video_source
        with patch("astrbot_plugin_punchline_analyzer.parser_bridge.download_file_with_limit", AsyncMock(return_value=False)):
            res_v = await self.plugin.parser_bridge.resolve_video_source(
                event=ev_quoted,
                reply_message_id="99881",
                task_dir=self.plugin.temp_dir,
                max_media_mb=50,
            )
        self.assertIsNotNone(res_v)
        self.assertTrue(res_v.get("has_video"))
        self.assertEqual(res_v.get("url"), "http://domain.com/fallback.mp4")

    # 21. 流式下载字节上限与超限清理测试
    async def test_download_file_with_limit_enforcement(self):
        from unittest.mock import AsyncMock, patch
        from astrbot_plugin_punchline_analyzer.parser_bridge import download_file_with_limit

        out_file = Path(tempfile.gettempdir()) / f"test_limit_{os.getpid()}.mp4"

        # 21.1 测试 Content-Length 超限时立即中止
        class MockResponseHeaderExceed:
            status = 200
            headers = {"Content-Length": str(100 * 1024 * 1024)}  # 100MB

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockSessionHeader:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def get(self, url, *args, **kwargs):
                return MockResponseHeaderExceed()

        with patch("aiohttp.ClientSession", MockSessionHeader):
            ok = await download_file_with_limit(
                url="http://fake/video.mp4",
                output_path=out_file,
                max_bytes=10 * 1024 * 1024,  # 限制 10MB
            )
            self.assertFalse(ok, "超过 Content-Length 限制必须直接返回 False")
            self.assertFalse(out_file.exists(), "超限时严禁留存文件")

        # 21.2 测试流式 chunk 累加超限时中止并清理已写入数据（无 Content-Length 头）
        class MockChunkStream:
            async def iter_chunked(self, n):
                yield b"A" * 600 * 1024  # 600KB
                yield b"B" * 600 * 1024  # 累计 1200KB > 1MB 限制

        class MockResponseStreamExceed:
            status = 200
            headers = {}  # 无 Content-Length 响应头，纯 chunked 流式
            content = MockChunkStream()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockSessionStream:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def get(self, url, *args, **kwargs):
                return MockResponseStreamExceed()

        with patch("aiohttp.ClientSession", MockSessionStream):
            ok = await download_file_with_limit(
                url="http://fake/stream.mp4",
                output_path=out_file,
                max_bytes=1 * 1024 * 1024,  # 限制 1MB
            )
            self.assertFalse(ok, "流式累加超过上限必须中止并返回 False")
            self.assertFalse(out_file.exists(), "流式超限中止后未完成文件必须已被删除")

    # 22. 流式下载中途取消与残留文件清理测试（无 Content-Length 头）
    async def test_stream_download_cancellation_and_cleanup(self):
        from astrbot_plugin_punchline_analyzer.parser_bridge import download_file_with_limit

        out_file = Path(tempfile.gettempdir()) / f"cancel_stream_{uuid.uuid4().hex[:6]}.mp4"
        write_started_event = asyncio.Event()

        class MockSlowChunkStream:
            async def iter_chunked(self, n):
                yield b"CHUNK_DATA_1" * 1024
                write_started_event.set()
                # 挂起等待被取消
                await asyncio.sleep(10.0)
                yield b"CHUNK_DATA_2" * 1024

        class MockSlowResponse:
            status = 200
            headers = {}  # 无 Content-Length 响应头
            content = MockSlowChunkStream()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockSlowSession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def get(self, url, *args, **kwargs):
                return MockSlowResponse()

        with patch("aiohttp.ClientSession", MockSlowSession):
            dl_task = asyncio.create_task(
                download_file_with_limit(
                    url="http://fake/slow_stream.mp4",
                    output_path=out_file,
                    max_bytes=10 * 1024 * 1024,
                )
            )

            # 等待至少第一块已写入
            await asyncio.wait_for(write_started_event.wait(), timeout=2.0)
            self.assertTrue(out_file.exists(), "第一块写入后文件应当暂时存在")

            # 模拟外部中途取消下载任务
            dl_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await dl_task

            # 断言：任务取消后，未完成的残留文件必须被安全清理
            self.assertFalse(out_file.exists(), "下载中途取消后，未完成的半截文件必须被立即清理")

    # 22. 直接发送视频与视频解析失败时的正常反馈与防捏造测试
    async def test_video_failure_explicit_feedback(self):
        # 用户直接在当前消息中发送 Video 组件
        vid_seg = MockVideo(url="http://domain.com/huge_video.mp4")
        ev = MockAstrMessageEvent(
            messages=[MockAt(qq="99999"), vid_seg, MockPlain(" 笑点解析")],
            self_id="99999",
        )

        # 模拟视频源解析返回：检测到视频，但超出大小上限未能提取画面
        async def mock_resolve_video(*args, **kwargs):
            return {
                "has_video": True,
                "caption": "",
                "video_path": None,
                "frames": [],
                "url": "http://domain.com/huge_video.mp4",
                "error_reason": "视频大小 (85.0MB) 超过上限 50MB",
            }

        self.plugin.parser_bridge.resolve_video_source = mock_resolve_video

        # 提取目标
        target = await self.plugin.extractor.extract_target(
            event=ev,
            user_plain_text="笑点解析",
            task_dir=self.plugin.temp_dir,
        )

        self.assertEqual(target.target_type, "media")
        self.assertIn("限制提示", target.text_content)
        self.assertIn("85.0MB", target.degrade_note)
        self.assertIn("严禁凭空捏造", target.degrade_note)
        self.assertEqual(target.images, [])

    # 23. 统一提取与规范化 Video 各种属性（url/path/file/file_id）分支测试
    def test_unified_video_fields_extraction(self):
        from astrbot_plugin_punchline_analyzer.parser_bridge import extract_video_info, is_video_segment

        # 23.1 AstrBot Video(url=...)
        v1 = MockVideo(url="http://media.com/v1.mp4")
        self.assertTrue(is_video_segment(v1))
        ref1, fid1 = extract_video_info(v1)
        self.assertEqual(ref1, "http://media.com/v1.mp4")
        self.assertIsNone(fid1)

        # 23.2 AstrBot Video(path=...)
        v2 = MockVideo(path="/var/lib/media/v2.mp4")
        ref2, fid2 = extract_video_info(v2)
        self.assertEqual(ref2, "/var/lib/media/v2.mp4")
        self.assertIsNone(fid2)

        # 23.3 AstrBot Video(file="file:///path/to/v3.mp4")
        v3 = MockVideo(file="file:///path/to/v3.mp4")
        ref3, fid3 = extract_video_info(v3)
        self.assertEqual(ref3, "/path/to/v3.mp4")
        self.assertIsNone(fid3)

        # 23.4 AstrBot Video(file="http://media.com/v4.mp4")
        v4 = MockVideo(file="http://media.com/v4.mp4")
        ref4, fid4 = extract_video_info(v4)
        self.assertEqual(ref4, "http://media.com/v4.mp4")
        self.assertIsNone(fid4)

        # 23.5 OneBot file_id in Video.file
        v5 = MockVideo(file="onebot_file_id_555")
        ref5, fid5 = extract_video_info(v5)
        self.assertIsNone(ref5)
        self.assertEqual(fid5, "onebot_file_id_555")

        # 23.6 Video.file_id
        v6 = MockVideo()
        v6.file_id = "fid_666"
        ref6, fid6 = extract_video_info(v6)
        self.assertEqual(fid6, "fid_666")

        # 23.7 OneBot v11 raw dict segment
        v7 = {"type": "video", "data": {"file": "hash_777", "url": "https://video.org/7.mp4"}}
        self.assertTrue(is_video_segment(v7))
        ref7, fid7 = extract_video_info(v7)
        self.assertEqual(ref7, "https://video.org/7.mp4")
        self.assertEqual(fid7, "hash_777")

    # 24. 共享解析器下载路径 50MB 限额落实且不改全局配置测试（基于独立上下文隔离）
    async def test_parser_download_50mb_limit_enforcement(self):
        class MockDownloader:
            def __init__(self, cfg=None):
                self.cfg = cfg
                self.max_size = getattr(cfg, "source_max_size", 90)

            async def close(self):
                pass

        class MockCfg:
            def __init__(self):
                self.source_max_size = 90
                self.max_duration = 300

        class SizeLimitException(Exception):
            pass

        shared_downloader = MockDownloader()
        shared_cfg = MockCfg()

        class MockVideoContentSizeLimit:
            def __init__(self):
                self.duration = 45.0

            async def get_path(self):
                # 模拟独立下载器在下载前或下载中触发了 50MB SizeLimitException
                raise SizeLimitException("视频大小超过 50MB，下载前/下载中阻断")

        class MockParseResult:
            def __init__(self):
                self.title = "超限视频"
                self.text = "75MB视频"
                self.author = None
                self.extra = None
                self.cover_path = asyncio.sleep(0, result=None)
                self.video_contents = [MockVideoContentSizeLimit()]
                self.img_contents = []
                self.url = "http://bilibili.com/video/BV1big75mb"

        class MockPlatformParser:
            def __init__(self, config=None, downloader=None):
                self.cfg = config
                self.downloader = downloader

            def create_video_content(self, url_or_task, cover_url=None, duration=0.0, headers=None):
                return MockVideoContentSizeLimit()

            async def parse(self, kw, match):
                # 关键断言：本请求专属实例的 downloader 必须是 50MB，cfg 必须是 180s
                assert self.downloader.max_size == 50, "独立下载器 max_size 必须为 50MB"
                assert self.cfg.max_duration == 180, "独立配置 max_duration 必须为 180s"
                # 同时断言全局共享单例绝对未被修改
                assert shared_downloader.max_size == 90, "全局共享下载器 max_size 绝不能被修改"
                assert shared_cfg.max_duration == 300, "全局共享配置 max_duration 绝不能被修改"
                return MockParseResult()

        shared_platform_parser = MockPlatformParser(shared_cfg, shared_downloader)

        class MockPattern:
            def search(self, text):
                return True

        class MockParserPlugin:
            def __init__(self):
                self.downloader = shared_downloader
                self.cfg = shared_cfg
                self.parser_map = {"bilibili": shared_platform_parser}
                self.key_pattern_list = [("bilibili", MockPattern())]

            async def parse_url(self, url: str):
                return await shared_platform_parser.parse("bilibili", None)

        parser_inst = MockParserPlugin()

        class MockStarMetadata:
            star_cls = parser_inst

        self.context.stars["astrbot_plugin_parser"] = MockStarMetadata()

        # 断言调用前全局配置状态
        self.assertEqual(shared_downloader.max_size, 90)
        self.assertEqual(shared_cfg.max_duration, 300)

        parsed = await self.plugin.parser_bridge.parse_url_with_existing_parser(
            url="http://bilibili.com/video/BV1big75mb",
            max_media_mb=50,
            max_video_duration=180,
            task_dir=self.plugin.temp_dir,
        )

        self.assertIsNotNone(parsed)
        # 必须正确捕获 SizeLimitException 并反馈给 video_error
        self.assertIn("超过本插件 50MB 限额", parsed.get("video_error", ""))
        self.assertEqual(parsed.get("frame_paths"), [])

        # 关键断言：调用结束后，共享解析器的全局配置从未被修改
        self.assertEqual(shared_downloader.max_size, 90, "共享解析器 downloader.max_size 始终为 90MB")
        self.assertEqual(shared_cfg.max_duration, 300, "共享解析器 cfg.max_duration 始终为 300s")

    # 25. 回归测试 ①：笑点解析与普通解析交错执行，普通请求全过程保持原限额及方法
    async def test_regression_interleaved_punchline_and_normal_parser_requests(self):
        class MockDownloader:
            def __init__(self, cfg=None):
                self.cfg = cfg
                self.max_size = getattr(cfg, "source_max_size", 90)

            async def close(self):
                pass

        class MockCfg:
            def __init__(self):
                self.source_max_size = 90
                self.max_duration = 300

        shared_downloader = MockDownloader()
        shared_cfg = MockCfg()

        punchline_running_event = asyncio.Event()

        class Mock80MBVideoContent:
            def __init__(self):
                self.duration = 60.0

            async def get_path(self):
                # 80MB 在 90MB 内，普通解析应当能顺利获取
                return "/tmp/normal_80mb.mp4"

        class MockPunchlineVideoContent:
            def __init__(self):
                self.duration = 60.0

            async def get_path(self):
                # 等待笑点解析运行
                punchline_running_event.set()
                await asyncio.sleep(0.05)
                # 80MB 超过 50MB，笑点解析必须在独立上下文中阻断
                raise Exception("SizeLimitException: 视频大小超过 50MB")

        class MockPlatformParser:
            def __init__(self, config=None, downloader=None):
                self.cfg = config
                self.downloader = downloader

            def create_video_content(self, *args, **kwargs):
                return "ORIGINAL_CREATE_METHOD"

            async def parse(self, kw, match):
                if getattr(self.downloader, "max_size", 90) == 50:
                    # 笑点解析路径（独立实例）
                    res = MagicMock()
                    res.title = "80MB视频"
                    res.text = "笑点解析"
                    res.author = None
                    res.extra = None
                    res.cover_path = None
                    res.video_contents = [MockPunchlineVideoContent()]
                    res.img_contents = []
                    res.url = "http://bilibili.com/video/BV1test"
                    return res
                else:
                    # 普通解析路径（共享实例）
                    res = MagicMock()
                    res.title = "80MB视频"
                    res.text = "普通解析"
                    res.author = None
                    res.extra = None
                    res.cover_path = None
                    res.video_contents = [Mock80MBVideoContent()]
                    res.img_contents = []
                    res.url = "http://bilibili.com/video/BV1test"
                    return res

        shared_platform = MockPlatformParser(shared_cfg, shared_downloader)
        orig_create_fn = shared_platform.create_video_content

        class MockPattern:
            def search(self, text):
                return True

        class MockParserPlugin:
            def __init__(self):
                self.downloader = shared_downloader
                self.cfg = shared_cfg
                self.parser_map = {"bilibili": shared_platform}
                self.key_pattern_list = [("bilibili", MockPattern())]

            async def parse_url(self, url: str):
                # 普通解析直接调用共享实例
                return await shared_platform.parse("bilibili", None)

        parser_plugin = MockParserPlugin()

        class MockStarMetadata:
            star_cls = parser_plugin

        self.context.stars["astrbot_plugin_parser"] = MockStarMetadata()

        # 启动笑点解析任务
        punchline_task = asyncio.create_task(
            self.plugin.parser_bridge.parse_url_with_existing_parser(
                url="http://bilibili.com/video/BV1test",
                max_media_mb=50,
                max_video_duration=180,
            )
        )

        # 等待笑点解析进入运行状态
        await punchline_running_event.wait()

        # 在笑点解析执行期间，并发执行普通万能解析请求
        normal_res = await parser_plugin.parse_url("http://bilibili.com/video/BV1test")
        normal_vpath = await normal_res.video_contents[0].get_path()

        # 断言普通请求全过程保持原配置、原限额与原方法
        self.assertEqual(shared_downloader.max_size, 90, "并发期间共享下载器 max_size 始终为 90MB")
        self.assertEqual(shared_cfg.max_duration, 300, "并发期间共享配置 max_duration 始终为 300s")
        self.assertEqual(shared_platform.create_video_content, orig_create_fn, "并发期间共享平台方法从未被猴子补丁")
        self.assertNotIn("create_video_content", shared_platform.__dict__, "共享平台实例未被注入任何实例级覆盖方法")
        self.assertEqual(normal_vpath, "/tmp/normal_80mb.mp4", "普通万能解析请求必须成功取得 80MB 视频")

        # 等待笑点解析完成
        punchline_res = await punchline_task
        self.assertIn("超过本插件 50MB 限额", punchline_res.get("video_error", ""), "笑点解析必须在 50MB 限额处受控阻断")

        # 再次确认共享实例毫无改动
        self.assertEqual(shared_downloader.max_size, 90)
        self.assertEqual(shared_cfg.max_duration, 300)
        self.assertEqual(shared_platform.create_video_content, orig_create_fn)
        self.assertNotIn("create_video_content", shared_platform.__dict__)

    # 26. 回归测试 ②：主插件与平台共享同一 cfg 对象，正常/异常/取消退出均检查共享状态
    async def test_regression_shared_cfg_between_plugin_and_platforms(self):
        class MockDownloader:
            def __init__(self, cfg=None):
                self.cfg = cfg
                self.max_size = getattr(cfg, "source_max_size", 90)

            async def close(self):
                pass

        class MockCfg:
            def __init__(self):
                self.source_max_size = 90
                self.max_duration = 300

        # 主插件与各平台共享同一个 cfg 对象 (parser_plugin.cfg is platform_parser.cfg)
        single_shared_cfg = MockCfg()
        shared_downloader = MockDownloader(single_shared_cfg)

        class MockPlatformParser:
            def __init__(self, config=None, downloader=None):
                self.cfg = config
                self.downloader = downloader

            def create_video_content(self, *args, **kwargs):
                return "ORIGINAL_METHOD"

            async def parse(self, kw, match):
                # 模拟根据参数抛出不同事件
                if "error" in kw:
                    raise RuntimeError("模拟解析阶段突发异常")
                elif "cancel" in kw:
                    await asyncio.sleep(10.0)  # 挂起等待外部取消
                res = MagicMock()
                res.title = "测试"
                res.text = "内容"
                res.author = None
                res.extra = None
                res.cover_path = None
                res.video_contents = []
                res.img_contents = []
                res.url = "http://domain.com/video"
                return res

        shared_platform = MockPlatformParser(single_shared_cfg, shared_downloader)
        self.assertIs(single_shared_cfg, shared_platform.cfg)

        class MockPattern:
            def search(self, text):
                return True

        class MockParserPlugin:
            def __init__(self):
                self.downloader = shared_downloader
                self.cfg = single_shared_cfg
                self.parser_map = {
                    "normal_kw": shared_platform,
                    "error_kw": shared_platform,
                    "cancel_kw": shared_platform,
                }
                self.key_pattern_list = [
                    ("normal_kw", MockPattern()),
                    ("error_kw", MockPattern()),
                    ("cancel_kw", MockPattern()),
                ]

        parser_plugin = MockParserPlugin()
        self.assertIs(parser_plugin.cfg, shared_platform.cfg)

        class MockStarMetadata:
            star_cls = parser_plugin

        self.context.stars["astrbot_plugin_parser"] = MockStarMetadata()

        # 26.1 正常退出检查
        res_normal = await self.plugin.parser_bridge.parse_url_with_existing_parser(
            url="http://domain.com/normal_kw/1",
            max_media_mb=50,
            max_video_duration=180,
        )
        self.assertIsNotNone(res_normal)
        self.assertIs(parser_plugin.cfg, shared_platform.cfg)
        self.assertEqual(single_shared_cfg.source_max_size, 90, "正常退出后 source_max_size 仍为 90")
        self.assertEqual(single_shared_cfg.max_duration, 300, "正常退出后 max_duration 仍为 300")

        # 26.2 异常退出检查
        res_err = await self.plugin.parser_bridge.parse_url_with_existing_parser(
            url="http://domain.com/error_kw/1",
            max_media_mb=50,
            max_video_duration=180,
        )
        self.assertIs(parser_plugin.cfg, shared_platform.cfg)
        self.assertEqual(single_shared_cfg.source_max_size, 90, "异常退出后 source_max_size 仍为 90")
        self.assertEqual(single_shared_cfg.max_duration, 300, "异常退出后 max_duration 仍为 300")

        # 26.3 取消退出检查
        cancel_task = asyncio.create_task(
            self.plugin.parser_bridge.parse_url_with_existing_parser(
                url="http://domain.com/cancel_kw/1",
                max_media_mb=50,
                max_video_duration=180,
            )
        )
        await asyncio.sleep(0.01)
        cancel_task.cancel()
        try:
            await cancel_task
        except asyncio.CancelledError:
            pass

        self.assertIs(parser_plugin.cfg, shared_platform.cfg)
        self.assertEqual(single_shared_cfg.source_max_size, 90, "取消退出后 source_max_size 仍为 90")
        self.assertEqual(single_shared_cfg.max_duration, 300, "取消退出后 max_duration 仍为 300")

    # 27. 回归测试 ③：多个 parser_map 键指向同一平台实例，正常/异常/取消退出均检查共享状态
    async def test_regression_multiple_parser_map_keys_same_instance(self):
        class MockDownloader:
            def __init__(self, cfg=None):
                self.cfg = cfg
                self.max_size = getattr(cfg, "source_max_size", 90)

            async def close(self):
                pass

        class MockCfg:
            def __init__(self):
                self.source_max_size = 90
                self.max_duration = 300

        shared_downloader = MockDownloader()
        shared_cfg = MockCfg()

        class MockBilibiliPlatform:
            def __init__(self, config=None, downloader=None):
                self.cfg = config
                self.downloader = downloader

            def create_video_content(self, *args, **kwargs):
                return "UNTOUCHED_CREATE_FN"

            async def parse(self, kw, match):
                if kw == "bili_err":
                    raise ValueError("平台解析异常测试")
                elif kw == "bili_cancel":
                    await asyncio.sleep(10.0)
                res = MagicMock()
                res.title = f"视频_{kw}"
                res.text = "描述"
                res.author = None
                res.extra = None
                res.cover_path = None
                res.video_contents = []
                res.img_contents = []
                res.url = f"http://domain/{kw}"
                return res

        single_platform_inst = MockBilibiliPlatform(shared_cfg, shared_downloader)
        orig_fn = single_platform_inst.create_video_content

        class MockPattern:
            def search(self, text):
                return True

        # 多个 parser_map 键均指向同一个 single_platform_inst 实例
        parser_map = {
            "bilibili.com": single_platform_inst,
            "b23.tv": single_platform_inst,
            "bili2233.cn": single_platform_inst,
            "bili_err": single_platform_inst,
            "bili_cancel": single_platform_inst,
        }

        key_patterns = [
            ("bilibili.com", MockPattern()),
            ("b23.tv", MockPattern()),
            ("bili2233.cn", MockPattern()),
            ("bili_err", MockPattern()),
            ("bili_cancel", MockPattern()),
        ]

        class MockParserPlugin:
            def __init__(self):
                self.downloader = shared_downloader
                self.cfg = shared_cfg
                self.parser_map = parser_map
                self.key_pattern_list = key_patterns

        parser_plugin = MockParserPlugin()

        class MockStarMetadata:
            star_cls = parser_plugin

        self.context.stars["astrbot_plugin_parser"] = MockStarMetadata()

        # 27.1 正常通过 b23.tv 请求
        res1 = await self.plugin.parser_bridge.parse_url_with_existing_parser(
            url="https://b23.tv/BV123456",
            max_media_mb=50,
            max_video_duration=180,
        )
        self.assertIsNotNone(res1)
        # 校验所有映射键上的实例方法和属性未被破坏
        self.assertEqual(single_platform_inst.create_video_content, orig_fn)
        self.assertEqual(parser_map["bilibili.com"].create_video_content, orig_fn)
        self.assertEqual(parser_map["b23.tv"].create_video_content, orig_fn)
        self.assertNotIn("create_video_content", single_platform_inst.__dict__)
        self.assertEqual(shared_downloader.max_size, 90)

        # 27.2 异常通过 bili_err 请求
        res2 = await self.plugin.parser_bridge.parse_url_with_existing_parser(
            url="https://bili_err/BV123456",
            max_media_mb=50,
            max_video_duration=180,
        )
        self.assertEqual(single_platform_inst.create_video_content, orig_fn)
        self.assertNotIn("create_video_content", single_platform_inst.__dict__)
        self.assertEqual(shared_downloader.max_size, 90)

        # 27.3 取消通过 bili_cancel 请求
        task3 = asyncio.create_task(
            self.plugin.parser_bridge.parse_url_with_existing_parser(
                url="https://bili_cancel/BV123456",
                max_media_mb=50,
                max_video_duration=180,
            )
        )
        await asyncio.sleep(0.01)
        task3.cancel()
        try:
            await task3
        except asyncio.CancelledError:
            pass

        self.assertEqual(single_platform_inst.create_video_content, orig_fn)
        self.assertEqual(parser_map["bili2233.cn"].create_video_content, orig_fn)
        self.assertNotIn("create_video_content", single_platform_inst.__dict__)
        self.assertEqual(shared_downloader.max_size, 90)

    # 28. 视频失败原因传入 Prompt 与封面明确声明测试
    async def test_video_failure_reason_and_cover_warning_in_prompt(self):
        # 1. 测试视频下载失败原因传入 Prompt
        video_seg = MockVideo(url="http://domain.com/failed.mp4")
        ev = MockAstrMessageEvent(
            messages=[MockAt(qq="99999"), video_seg, MockPlain(" 笑点解析")],
            self_id="99999",
        )

        async def mock_resolve_fail(*args, **kwargs):
            return {
                "has_video": True,
                "caption": "",
                "video_path": None,
                "frames": [],
                "url": "http://domain.com/failed.mp4",
                "error_reason": "视频下载超时（超过 15 秒）",
                "cover_path": None,
            }

        self.plugin.parser_bridge.resolve_video_source = mock_resolve_fail
        target = await self.plugin.extractor.extract_target(
            event=ev,
            user_plain_text="笑点解析",
            task_dir=self.plugin.temp_dir,
        )

        prompt = build_user_prompt(
            target_description=target.description,
            text_content=target.text_content,
            supplementary_text=target.supplementary_text,
            degrade_note=target.degrade_note,
        )

        self.assertIn("视频下载超时（超过 15 秒）", prompt)
        self.assertIn("封面不代表已读取视频画面", prompt)

        # 2. 测试仅有封面时明确警告封面不代表视频画面
        cover_temp = Path(tempfile.gettempdir()) / f"fake_cover_{uuid.uuid4().hex[:6]}.jpg"
        cover_temp.write_text("cover")
        try:
            async def mock_resolve_cover_only(*args, **kwargs):
                return {
                    "has_video": True,
                    "caption": "",
                    "video_path": None,
                    "frames": [],
                    "url": "http://domain.com/cover_only.mp4",
                    "error_reason": "视频超出50MB大小限制",
                    "cover_path": str(cover_temp),
                }

            self.plugin.parser_bridge.resolve_video_source = mock_resolve_cover_only
            target_cover = await self.plugin.extractor.extract_target(
                event=ev,
                user_plain_text="笑点解析",
                task_dir=self.plugin.temp_dir,
            )

            prompt_cover = build_user_prompt(
                target_description=target_cover.description,
                text_content=target_cover.text_content,
                supplementary_text=target_cover.supplementary_text,
                degrade_note=target_cover.degrade_note,
            )

            self.assertIn("封面图片绝不代表已读取视频画面", prompt_cover)
            self.assertIn("严禁根据封面脑补捏造视频具体情节", prompt_cover)
        finally:
            if cover_temp.exists():
                cover_temp.unlink()

    # 29. 独立备份脚本与回滚两种场景（全新安装与旧版升级覆盖）自动化测试
    def test_backup_manifest_and_state_restoration(self):
        from astrbot_plugin_punchline_analyzer.tools.backup_and_rollback import run_scenario_verifications

        # 执行独立脚本的两种场景完整自检（全新安装无残留清除 + 旧版升级覆盖哈希恢复）
        ok = run_scenario_verifications()
        self.assertTrue(ok, "独立备份与回滚工具在两种场景下的测试必须全部通过")

    # 30. 全入口统一恢复经典纯文本输出格式回归测试 (V1)
    async def test_format_verification_across_all_entry_points(self):
        """全面覆盖全部输入入口，严格断言首行、尾行、编号格式与无 Markdown"""
        # 入口 1: @当前机器人 + “笑点解析” 触发普通文本引用
        ev_reply_text = MockAstrMessageEvent(
            messages=[
                MockAt(qq="99999"),
                MockReply(message_str="昨晚写代码把数据库删了，现在在跑路", sender_nickname="苦逼程序员"),
                MockPlain(" 笑点解析"),
            ],
            self_id="99999",
        )
        res1 = []
        async for r in self.plugin.on_message(ev_reply_text):
            res1.append(r)
        self.assertTrue(len(res1) > 0)
        ok1, reason1 = validate_punchline_format(res1[0])
        self.assertTrue(ok1, f"入口 1 (@机器人+引用文字) 格式校验失败: {reason1}\n输出内容:\n{res1[0]}")
        lines1 = res1[0].split("\n")
        self.assertEqual(lines1[0], "笑点解析：")
        self.assertEqual(lines1[-1], "令人忍俊不禁。[张三]问的")
        for mid in lines1[1:-1]:
            self.assertTrue(mid.startswith("笑点") and mid.endswith("很好笑。"))
            self.assertFalse(any(c in mid for c in ["*", "#", "_", "`"]))

        # 入口 2: 独立指令 /笑点解析 最近30条
        for i in range(3):
            ev_h = MockAstrMessageEvent(
                messages=[MockPlain(f"群友发言测试第 {i + 1} 条：今天天气真好")],
                sender_id=f"u_{i}",
                sender_name=f"群友_{i}",
                group_id="group_fmt_all",
                message_id=f"msg_h_{i}",
            )
            self.plugin.history_manager.record_event(ev_h)

        cmd_ev = MockAstrMessageEvent(
            messages=[MockPlain("/笑点解析 最近30条")],
            sender_id="u_admin",
            group_id="group_fmt_all",
        )
        res2 = []
        async for r in self.plugin.cmd_punchline(cmd_ev, arg="最近30条"):
            res2.append(r)
        self.assertTrue(len(res2) > 0)
        ok2, reason2 = validate_punchline_format(res2[0])
        self.assertTrue(ok2, f"入口 2 (/笑点解析 最近30条) 格式校验失败: {reason2}")

        # 入口 3: 普通当前消息直接附带图片/表情包
        ev_direct_img = MockAstrMessageEvent(
            messages=[
                MockAt(qq="99999"),
                MockImage("http://img/funny_cat.png"),
                MockPlain(" 笑点解析"),
            ],
            self_id="99999",
        )
        res3 = []
        async for r in self.plugin.on_message(ev_direct_img):
            res3.append(r)
        self.assertTrue(len(res3) > 0)
        ok3, reason3 = validate_punchline_format(res3[0])
        self.assertTrue(ok3, f"入口 3 (直接图片/表情包) 格式校验失败: {reason3}")

        # 入口 4: 引用图片/表情包
        ev_reply_img = MockAstrMessageEvent(
            messages=[
                MockAt(qq="99999"),
                MockReply(message_str="", chain=[MockImage("http://img/meme.jpg")], sender_nickname="梗王"),
                MockPlain(" 笑点解析"),
            ],
            self_id="99999",
        )
        res4 = []
        async for r in self.plugin.on_message(ev_reply_img):
            res4.append(r)
        self.assertTrue(len(res4) > 0)
        ok4, reason4 = validate_punchline_format(res4[0])
        self.assertTrue(ok4, f"入口 4 (引用图片/表情包) 格式校验失败: {reason4}")

        # 入口 5: 合并转发聊天记录
        mock_forward_nodes = [
            {"sender": {"nickname": "甲"}, "time": 1700000001, "message": [{"type": "text", "data": {"text": "今天吃啥"}}]},
            {"sender": {"nickname": "乙"}, "time": 1700000002, "message": [{"type": "text", "data": {"text": "西北风"}}]},
        ]
        mock_bot = MockBot({"get_forward_msg": lambda **kw: {"messages": mock_forward_nodes}})
        ev_forward = MockAstrMessageEvent(
            messages=[
                MockAt(qq="99999"),
                MockForward("fwd_9999"),
                MockPlain(" 笑点解析"),
            ],
            self_id="99999",
        )
        ev_forward.bot = mock_bot
        res5 = []
        async for r in self.plugin.on_message(ev_forward):
            res5.append(r)
        self.assertTrue(len(res5) > 0)
        ok5, reason5 = validate_punchline_format(res5[0])
        self.assertTrue(ok5, f"入口 5 (合并转发记录) 格式校验失败: {reason5}")

        # 入口 6: 直接视频/引用视频抽帧分析正常路径
        dummy_frame = Path(tempfile.gettempdir()) / f"fmt_frame_{uuid.uuid4().hex[:6]}.jpg"
        dummy_frame.write_text("frame_data")
        try:
            async def mock_resolve_normal_video(*args, **kwargs):
                return {
                    "has_video": True,
                    "caption": "",
                    "video_path": "/fake/video.mp4",
                    "frames": [str(dummy_frame)],
                    "url": "http://domain.com/normal.mp4",
                    "error_reason": "",
                }

            self.plugin.parser_bridge.resolve_video_source = mock_resolve_normal_video
            ev_video = MockAstrMessageEvent(
                messages=[
                    MockAt(qq="99999"),
                    MockVideo(url="http://domain.com/normal.mp4"),
                    MockPlain(" 笑点解析"),
                ],
                self_id="99999",
            )
            res6 = []
            async for r in self.plugin.on_message(ev_video):
                res6.append(r)
            self.assertTrue(len(res6) > 0)
            ok6, reason6 = validate_punchline_format(res6[0])
            self.assertTrue(ok6, f"入口 6 (直接视频抽帧) 格式校验失败: {reason6}")
            self.assertNotIn("辅助截图", res6[0], "视频分析回复严禁提及辅助截图")
            self.assertNotIn("抽帧", res6[0])
        finally:
            if dummy_frame.exists():
                dummy_frame.unlink()

        # 入口 7: B站链接与万能解析卡片正常抽帧路径
        dummy_bili_frame = Path(tempfile.gettempdir()) / f"fmt_bili_{uuid.uuid4().hex[:6]}.jpg"
        dummy_bili_frame.write_text("bili_frame_data")
        try:
            async def mock_parse_bili(*args, **kwargs):
                return {
                    "title": "搞笑日常",
                    "text": "沙雕小视频",
                    "author": "UP主小王",
                    "extra_info": "",
                    "images": [str(dummy_bili_frame)],
                    "frame_paths": [str(dummy_bili_frame)],
                    "video_error": "",
                }

            self.plugin.parser_bridge.parse_url_with_existing_parser = mock_parse_bili
            ev_bili = MockAstrMessageEvent(
                messages=[
                    MockAt(qq="99999"),
                    MockPlain(" https://www.bilibili.com/video/BV1test123 笑点解析"),
                ],
                self_id="99999",
            )
            res7 = []
            async for r in self.plugin.on_message(ev_bili):
                res7.append(r)
            self.assertTrue(len(res7) > 0)
            ok7, reason7 = validate_punchline_format(res7[0])
            self.assertTrue(ok7, f"入口 7 (B站链接解析) 格式校验失败: {reason7}")
        finally:
            if dummy_bili_frame.exists():
                dummy_bili_frame.unlink()

    # 31. 降级与防捏造测试 (V2)
    async def test_degrade_and_anti_hallucination_scenarios(self):
        """测试视频未读取画面、仅有封面及平淡无笑点场景，严格断言不产生伪造笑点且不带“很好笑。”"""
        # 场景 1: 视频画面读取失败（超出大小上限/网络异常）
        async def mock_resolve_video_fail(*args, **kwargs):
            return {
                "has_video": True,
                "caption": "",
                "video_path": None,
                "frames": [],
                "url": "http://domain.com/large_video.mp4",
                "error_reason": "视频大小 (80MB) 超过上限 50MB",
            }

        self.plugin.parser_bridge.resolve_video_source = mock_resolve_video_fail

        # 模拟真实模型偶发捏造或返回提示
        self.context.llm_generate = AsyncMock(
            return_value=MockLLMResponse("笑点1：视频中跌倒动作很好笑。")
        )

        ev_fail = MockAstrMessageEvent(
            messages=[MockAt(qq="99999"), MockVideo(url="http://domain.com/large_video.mp4"), MockPlain(" 笑点解析")],
            self_id="99999",
        )
        res_fail = []
        async for r in self.plugin.on_message(ev_fail):
            res_fail.append(r)

        self.assertTrue(len(res_fail) > 0)
        out_fail = res_fail[0]
        ok, reason = validate_punchline_format(out_fail)
        self.assertTrue(ok, f"视频读取失败降级格式不合规: {reason}\n输出:\n{out_fail}")

        # 严格断言防捏造与格式规范
        self.assertTrue(out_fail.startswith("笑点解析：\n"))
        self.assertTrue(out_fail.endswith("\n令人忍俊不禁。[张三]问的") or out_fail.endswith("\n令人忍俊不禁"))
        self.assertIn("未能读取视频画面", out_fail)
        self.assertNotIn("笑点1：", out_fail, "降级场景严禁产生伪造编号笑点")
        self.assertNotIn("很好笑。", out_fail, "失败说明严禁追加“很好笑。”")
        self.assertNotIn("很好笑", out_fail, "失败说明严禁包含“很好笑”")

        # 场景 1b: 用户提问附带长文本（>5字），纯视频读取失败时依然强力防捏造
        ev_fail_long = MockAstrMessageEvent(
            messages=[
                MockAt(qq="99999"),
                MockVideo(url="http://domain.com/large_video.mp4"),
                MockPlain(" 笑点解析 帮我看看这个视频好不好笑"),
            ],
            self_id="99999",
        )
        res_fail_long = []
        async for r in self.plugin.on_message(ev_fail_long):
            res_fail_long.append(r)
        self.assertTrue(len(res_fail_long) > 0)
        out_fail_long = res_fail_long[0]
        ok_fl, reason_fl = validate_punchline_format(out_fail_long)
        self.assertTrue(ok_fl, f"长提问下视频失败防捏造格式不合规: {reason_fl}\n输出:\n{out_fail_long}")
        self.assertIn("未能读取视频画面", out_fail_long)
        self.assertNotIn("笑点1：", out_fail_long, "用户长提示词绝不能击穿视频失败防捏造保护")

        # 场景 2: 仅获取到封面，无视频内部画面
        cover_temp = Path(tempfile.gettempdir()) / f"degrade_cov_{uuid.uuid4().hex[:6]}.jpg"
        cover_temp.write_text("cover")
        try:
            async def mock_resolve_cover_only(*args, **kwargs):
                return {
                    "has_video": True,
                    "caption": "",
                    "video_path": None,
                    "frames": [],
                    "url": "http://domain.com/cover_only.mp4",
                    "error_reason": "视频格式不支持下载",
                    "cover_path": str(cover_temp),
                }

            self.plugin.parser_bridge.resolve_video_source = mock_resolve_cover_only
            ev_cover = MockAstrMessageEvent(
                messages=[MockAt(qq="99999"), MockVideo(url="http://domain.com/cover_only.mp4"), MockPlain(" 笑点解析")],
                self_id="99999",
            )
            res_cover = []
            async for r in self.plugin.on_message(ev_cover):
                res_cover.append(r)

            self.assertTrue(len(res_cover) > 0)
            out_cover = res_cover[0]
            ok_c, reason_c = validate_punchline_format(out_cover)
            self.assertTrue(ok_c, f"仅有封面降级格式不合规: {reason_c}\n输出:\n{out_cover}")
            self.assertNotIn("笑点1：", out_cover, "仅有封面时严禁伪造视频动态笑点")
            self.assertNotIn("很好笑。", out_cover)
        finally:
            if cover_temp.exists():
                cover_temp.unlink()

        # 场景 1c: 模型将视频读取失败拆分为多行编号笑点，强制纠正为纯文本失败说明
        raw_multiline_fail = (
            "笑点解析：\n"
            "笑点1：未能读取视频画面。\n"
            "笑点2：现有信息不足以判断笑点，请补充视频内容。\n"
            "令人忍俊不禁"
        )
        formatted_mf = format_punchline_response(raw_multiline_fail)
        ok_mf, reason_mf = validate_punchline_format(formatted_mf)
        self.assertTrue(ok_mf, f"多行失败说明纠正失败: {reason_mf}\n输出:\n{formatted_mf}")
        self.assertNotIn("笑点1：", formatted_mf, "失败说明严禁包含笑点1：编号")
        self.assertNotIn("很好笑。", formatted_mf, "多行失败说明严禁追加“很好笑。”")
        self.assertIn("未能读取视频画面", formatted_mf)

        # 场景 3: 材料日常平淡，无明显笑点（实事求是）
        plain_no_joke = "素材内容较为日常平淡，未发现明显的幽默反差或笑点。"
        formatted_plain = format_punchline_response(plain_no_joke)
        ok_p, reason_p = validate_punchline_format(formatted_plain)
        self.assertTrue(ok_p, f"平淡无笑点说明格式不合规: {reason_p}\n输出:\n{formatted_plain}")
        self.assertIn("未发现明显的幽默反差或笑点", formatted_plain)
        self.assertNotIn("很好笑。", formatted_plain, "平淡无笑点说明严禁追加“很好笑。”")

        # 场景 3b: 日常平淡普通场景被模型误加编号“笑点1：”，非视频降级路径下强力纠正
        plain_numbered = "笑点解析：\n笑点1：素材内容较为日常平淡，现有信息未发现明显的幽默反差或笑点。\n令人忍俊不禁"
        formatted_pn = format_punchline_response(plain_numbered, is_degraded_video=False)
        ok_pn, reason_pn = validate_punchline_format(formatted_pn)
        self.assertTrue(ok_pn, f"日常平淡编号纠正格式不合规: {reason_pn}\n输出:\n{formatted_pn}")
        self.assertNotIn("笑点1：", formatted_pn, "日常平淡失败说明严禁保留假编号")
        self.assertNotIn("很好笑。", formatted_pn, "日常平淡失败说明严禁追加“很好笑。”")

        # 场景 4: 降级分析——媒体卡片/B站视频超出 50MB 限额但具备标题与文本（部分材料读取降级分析）
        async def mock_parse_bili_degraded(*args, **kwargs):
            return {
                "title": "教你一秒学会高数",
                "text": "UP主简介：梦里什么都有",
                "author": "学霸君",
                "extra_info": "",
                "images": [],
                "frame_paths": [],
                "video_error": "视频超出本插件 50MB 限额",
            }

        self.plugin.parser_bridge.parse_url_with_existing_parser = mock_parse_bili_degraded
        self.context.llm_generate = AsyncMock(
            return_value=MockLLMResponse("笑点1：标题写着一秒学会高数，简介却是梦里什么都有，反差强烈，很好笑。")
        )
        ev_bili_degraded = MockAstrMessageEvent(
            messages=[MockAt(qq="99999"), MockPlain(" https://www.bilibili.com/video/BV1part 笑点解析")],
            self_id="99999",
        )
        res_bili_degraded = []
        async for r in self.plugin.on_message(ev_bili_degraded):
            res_bili_degraded.append(r)

        self.assertTrue(len(res_bili_degraded) > 0)
        out_bili_degraded = res_bili_degraded[0]
        ok_bd, reason_bd = validate_punchline_format(out_bili_degraded)
        self.assertTrue(ok_bd, f"部分材料降级分析格式不合规: {reason_bd}\n输出:\n{out_bili_degraded}")
        self.assertIn("笑点1：标题写着一秒学会高数", out_bili_degraded)
        self.assertNotIn("未能读取视频画面，现有信息不足以判断笑点", out_bili_degraded)

    # 32. 格式校验与有限修正机制专项测试 (V1)
    def test_formatter_and_corrector_edge_cases(self):
        """测试 Markdown 清理、编号归一化、截图词汇过滤与异常修正"""
        # 32.1 混入 Markdown 标题、加粗与列表符号
        raw_markdown = (
            "### 笑点解析：\n"
            "- **笑点一：** 程序员删库跑路，态度极其淡定。\n"
            "- **笑点2：** 运维经理当场破防晕倒\n"
            "令人忍俊不禁"
        )
        formatted1 = format_punchline_response(raw_markdown)
        ok1, reason1 = validate_punchline_format(formatted1)
        self.assertTrue(ok1, f"Markdown 清理修复失败: {reason1}\n{formatted1}")
        self.assertEqual(
            formatted1,
            "笑点解析：\n笑点1：程序员删库跑路，态度极其淡定，很好笑。\n笑点2：运维经理当场破防晕倒，很好笑。\n令人忍俊不禁",
        )

        # 32.2 包含辅助截图等违规技术词汇
        raw_screenshot = (
            "笑点解析：\n"
            "笑点1：从辅助截图中可以看出，猫猫在偷吃鱼干，动作非常搞笑。\n"
            "令人忍俊不禁"
        )
        formatted2 = format_punchline_response(raw_screenshot)
        ok2, reason2 = validate_punchline_format(formatted2)
        self.assertTrue(ok2, f"辅助截图过滤失败: {reason2}\n{formatted2}")
        self.assertNotIn("辅助截图", formatted2)
        self.assertIn("从画面中可以看出", formatted2)

        # 32.3 失败说明被偶发追加“很好笑。”修正清除
        raw_failed_with_haoxiao = (
            "笑点解析：\n"
            "未能读取视频画面，现有信息不足以判断笑点，请补充视频内容或相关台词。很好笑。\n"
            "令人忍俊不禁"
        )
        formatted3 = format_punchline_response(raw_failed_with_haoxiao)
        ok3, reason3 = validate_punchline_format(formatted3)
        self.assertTrue(ok3, f"失败说明清除很好笑失败: {reason3}\n{formatted3}")
        self.assertNotIn("很好笑", formatted3)
        self.assertEqual(
            formatted3,
            "笑点解析：\n未能读取视频画面，现有信息不足以判断笑点，请补充视频内容或相关台词。\n令人忍俊不禁",
        )

        # 32.4 携带前后闲聊废话自动剥离
        raw_fluff = (
            "好的，为您进行本次笑点拆解分析如下：\n"
            "笑点1：反差感十足，让人猝不及防\n"
            "令人忍俊不禁\n"
            "以上就是本次笑点解析，希望能给您带来欢乐！"
        )
        formatted4 = format_punchline_response(raw_fluff)
        ok4, reason4 = validate_punchline_format(formatted4)
        self.assertTrue(ok4, f"闲聊废话剥离失败: {reason4}\n{formatted4}")
        self.assertNotIn("好的", formatted4)
        self.assertNotIn("以上就是", formatted4)
        self.assertEqual(
            formatted4,
            "笑点解析：\n笑点1：反差感十足，让人猝不及防，很好笑。\n令人忍俊不禁",
        )

        # 32.5 空白与极端异常输入
        formatted5 = format_punchline_response("")
        ok5, reason5 = validate_punchline_format(formatted5)
        self.assertTrue(ok5, f"空输入兜底失败: {reason5}\n{formatted5}")
        self.assertEqual(
            formatted5,
            "笑点解析：\n现有信息不足以判断笑点。\n令人忍俊不禁",
        )

        # 32.6 末尾带有感叹号或波浪号的“很好笑！”防重复追加
        raw_punct_haoxiao = "笑点解析：\n笑点1：猫猫翻车了，很好笑！\n笑点2：狗狗看戏，很好笑~\n令人忍俊不禁"
        formatted6 = format_punchline_response(raw_punct_haoxiao)
        ok6, reason6 = validate_punchline_format(formatted6)
        self.assertTrue(ok6, f"标点很好笑去重失败: {reason6}\n{formatted6}")
        self.assertNotIn("很好笑，很好笑", formatted6)
        self.assertEqual(
            formatted6,
            "笑点解析：\n笑点1：猫猫翻车了，很好笑。\n笑点2：狗狗看戏，很好笑。\n令人忍俊不禁",
        )

        # 32.7 编号后顿号、英文点、减号等各种异形分隔符清洗
        raw_odd_delims = "笑点解析：\n笑点一、这是第一个笑点\n笑点二. 这是第二个笑点\n笑点3 - 这是第三个笑点\n令人忍俊不禁"
        formatted7 = format_punchline_response(raw_odd_delims)
        ok7, reason7 = validate_punchline_format(formatted7)
        self.assertTrue(ok7, f"异形分隔符清洗失败: {reason7}\n{formatted7}")
        self.assertNotIn("笑点1：、", formatted7)
        self.assertNotIn("笑点2：.", formatted7)
        self.assertNotIn("笑点3：-", formatted7)

        # 32.8 括号编号、方头括号与圆圈数字识别
        raw_brackets = "笑点解析：\n【笑点1】这是方头括号笑点\n(2) 这是圆括号笑点\n③ 这是圆圈数字笑点\n令人忍俊不禁"
        formatted8 = format_punchline_response(raw_brackets)
        ok8, reason8 = validate_punchline_format(formatted8)
        self.assertTrue(ok8, f"括号及圆圈编号解析失败: {reason8}\n{formatted8}")
        self.assertIn("笑点1：这是方头括号笑点，很好笑。", formatted8)
        self.assertIn("笑点2：这是圆括号笑点，很好笑。", formatted8)
        self.assertIn("笑点3：这是圆圈数字笑点，很好笑。", formatted8)

        # 32.9 代码块包裹与行内反引号 Markdown 清理
        raw_code = "```text\n笑点解析：\n笑点1：群友输入了 `rm -rf /`，胆量惊人。\n令人忍俊不禁\n```"
        formatted9 = format_punchline_response(raw_code)
        ok9, reason9 = validate_punchline_format(formatted9)
        self.assertTrue(ok9, f"代码块与反引号清理失败: {reason9}\n{formatted9}")
        self.assertNotIn("`", formatted9)
        self.assertIn("笑点1：群友输入了 rm -rf /，胆量惊人，很好笑。", formatted9)

        # 32.10 Markdown 引用块符号 > 清理
        raw_quote = "> 笑点解析：\n> 笑点1：猫咪跌倒了\n> 笑点2：主人大笑\n> 令人忍俊不禁"
        formatted10 = format_punchline_response(raw_quote)
        ok10, reason10 = validate_punchline_format(formatted10)
        self.assertTrue(ok10, f"引用块符号清理失败: {reason10}\n{formatted10}")
        self.assertNotIn(">", formatted10)

        # 32.11 尾行带标点“令人忍俊不禁。”过滤防污染
        raw_footer_punct = "笑点解析：\n笑点1：猫猫翻车了\n令人忍俊不禁。"
        formatted11 = format_punchline_response(raw_footer_punct)
        ok11, reason11 = validate_punchline_format(formatted11)
        self.assertTrue(ok11, f"尾行带标点过滤失败: {reason11}\n{formatted11}")
        self.assertEqual(
            formatted11,
            "笑点解析：\n笑点1：猫猫翻车了，很好笑。\n令人忍俊不禁",
        )

        # 32.12 无编号的多段落纯文本自动分段提取
        raw_multi_paras = (
            "群友甲说今天不加班，结果一小时后还在群里发报错截图。\n\n"
            "群友乙回复了一张加班人专属表情包，直接把气氛拉满。"
        )
        formatted12 = format_punchline_response(raw_multi_paras)
        ok12, reason12 = validate_punchline_format(formatted12)
        self.assertTrue(ok12, f"无编号多段落提取失败: {reason12}\n{formatted12}")
        self.assertIn("笑点1：群友甲", formatted12)
        self.assertIn("笑点2：群友乙", formatted12)

        # 32.13 前置背景词（如信息不足）但紧随编号笑点时，多行正则严格保障正常提取
        raw_pre_info = (
            "笑点解析：\n"
            "虽然部分聊天记录信息不足，但依然有明确笑点：\n"
            "笑点1：群友发言前后反差极大，态度转变极快。\n"
            "令人忍俊不禁"
        )
        formatted13 = format_punchline_response(raw_pre_info)
        ok13, reason13 = validate_punchline_format(formatted13)
        self.assertTrue(ok13, f"多行编号笑点防误判失败: {reason13}\n{formatted13}")
        self.assertIn("笑点1：群友发言前后反差极大", formatted13)

        # 32.14 单个笑点阐述中包含“平淡无奇”等反衬修饰词时，不被误杀为失败说明
        raw_contrast_pingdan = (
            "笑点解析：\n"
            "笑点1：主人公故意装作平淡无奇，结果下一秒被打脸，反差感拉满，很好笑。\n"
            "令人忍俊不禁"
        )
        formatted14 = format_punchline_response(raw_contrast_pingdan)
        ok14, reason14 = validate_punchline_format(formatted14)
        self.assertTrue(ok14, f"反衬平淡无奇被误判为失败说明: {reason14}\n{formatted14}")
        self.assertIn("笑点1：主人公故意装作平淡无奇", formatted14)

        # 32.15 嵌套前缀与括号/圆圈/方头括号编号无缝兼容
        raw_complex_prefixes = (
            "笑点解析：\n"
            "笑点(1)：猫猫翻车了\n"
            "【笑点】狗狗看戏\n"
            "第一个笑点是仓鼠越狱\n"
            "令人忍俊不禁"
        )
        formatted15 = format_punchline_response(raw_complex_prefixes)
        ok15, reason15 = validate_punchline_format(formatted15)
        self.assertTrue(ok15, f"复杂括号前缀规范化失败: {reason15}\n{formatted15}")
        self.assertIn("笑点1：猫猫翻车了，很好笑。", formatted15)
        self.assertIn("笑点2：狗狗看戏，很好笑。", formatted15)
        self.assertIn("笑点3：仓鼠越狱，很好笑。", formatted15)
        self.assertNotIn("笑点(1)", formatted15)
        self.assertNotIn("【笑点】", formatted15)

        # 32.16 无编号仅有“笑点：”前缀去重
        raw_dup_prefix = (
            "笑点解析：\n"
            "笑点：程序员删库跑路\n"
            "笑点：运维经理当场破防\n"
            "令人忍俊不禁"
        )
        formatted16 = format_punchline_response(raw_dup_prefix)
        ok16, reason16 = validate_punchline_format(formatted16)
        self.assertTrue(ok16, f"无编号笑点前缀去重失败: {reason16}\n{formatted16}")
        self.assertNotIn("笑点1：笑点：", formatted16)
        self.assertIn("笑点1：程序员删库跑路，很好笑。", formatted16)

        # 32.17 尾行“令人忍俊不禁”后附带闲聊追问被彻底截断丢弃
        raw_tail_chatter = (
            "笑点解析：\n"
            "笑点1：猫猫翻车了\n"
            "令人忍俊不禁\n"
            "大家觉得好笑吗？欢迎在群里讨论！\n"
            "如有疑问请@我"
        )
        formatted17 = format_punchline_response(raw_tail_chatter)
        ok17, reason17 = validate_punchline_format(formatted17)
        self.assertTrue(ok17, f"尾行后闲聊追问未截断: {reason17}\n{formatted17}")
        self.assertNotIn("大家觉得好笑吗", formatted17)
        self.assertNotIn("如有疑问", formatted17)
        self.assertTrue(formatted17.endswith("\n令人忍俊不禁"))

        # 32.18 失败说明尾部自带“令人忍俊不禁”防重复追加
        raw_fail_inline_footer = "未能读取视频画面，现有信息不足以判断笑点。令人忍俊不禁"
        formatted18 = format_punchline_response(raw_fail_inline_footer)
        ok18, reason18 = validate_punchline_format(formatted18)
        self.assertTrue(ok18, f"失败说明尾行去重失败: {reason18}\n{formatted18}")
        self.assertEqual(formatted18.count("令人忍俊不禁"), 1)

        # 32.19 失败说明被模型误带“笑点1：”且末尾被带“很好笑。”强力纠正
        raw_fail_with_haoxiao_num = (
            "笑点解析：\n"
            "笑点1：未能读取视频画面，现有信息不足以判断笑点，请补充视频内容或相关台词。很好笑。\n"
            "令人忍俊不禁"
        )
        formatted19 = format_punchline_response(raw_fail_with_haoxiao_num)
        ok19, reason19 = validate_punchline_format(formatted19)
        self.assertTrue(ok19, f"带编号与很好笑的失败说明格式纠正不合规: {reason19}\n{formatted19}")
        self.assertNotIn("笑点1：", formatted19, "失败说明严禁出现编号笑点")
        self.assertNotIn("很好笑", formatted19, "失败说明严禁包含很好笑")

        # 32.20 正文以数字或时间词开头（如 300个程序员、一天、一段）严格保护不被误吞
        raw_number_start = (
            "笑点解析：\n"
            "笑点1：300个程序员同时提交代码导致Git崩溃\n"
            "笑点2：一天吃了三碗面\n"
            "令人忍俊不禁"
        )
        formatted20 = format_punchline_response(raw_number_start)
        ok20, reason20 = validate_punchline_format(formatted20)
        self.assertTrue(ok20, f"数量词开头笑点规范化失败: {reason20}\n{formatted20}")
        self.assertIn("笑点1：300个程序员同时提交代码导致Git崩溃，很好笑。", formatted20)
        self.assertIn("笑点2：一天吃了三碗面，很好笑。", formatted20)

        # 32.21 正文以“注视”、“注定”等词汇开头无编号段落保护不被误过滤
        raw_zhu_start = (
            "注视着对方的表情逐渐凝固，产生了强烈的戏剧反差。\n\n"
            "注定是一个令人爆笑的误会展开。"
        )
        formatted21 = format_punchline_response(raw_zhu_start)
        ok21, reason21 = validate_punchline_format(formatted21)
        self.assertTrue(ok21, f"注字开头正文保留失败: {reason21}\n{formatted21}")
        self.assertIn("笑点1：注视着对方的表情逐渐凝固", formatted21)
        self.assertIn("笑点2：注定是一个令人爆笑的误会展开", formatted21)

        # 32.22 尾部带有“太好笑了”、“非常好笑”等修饰词规范收尾为单次“很好笑。”，防叠词
        raw_various_haoxiao = (
            "笑点解析：\n"
            "笑点1：猫猫翻车了，太好笑了。\n"
            "笑点2：狗狗看戏，非常好笑！\n"
            "笑点3：仓鼠越狱，确实很好笑\n"
            "令人忍俊不禁"
        )
        formatted22 = format_punchline_response(raw_various_haoxiao)
        ok22, reason22 = validate_punchline_format(formatted22)
        self.assertTrue(ok22, f"多样好笑收尾规范化失败: {reason22}\n{formatted22}")
        self.assertIn("笑点1：猫猫翻车了，很好笑。", formatted22)
        self.assertIn("笑点2：狗狗看戏，很好笑。", formatted22)
        self.assertIn("笑点3：仓鼠越狱，很好笑。", formatted22)
        self.assertNotIn("太好笑了，很好笑", formatted22)
        self.assertNotIn("非常好笑，很好笑", formatted22)
        self.assertNotIn("确实很好笑，很好笑", formatted22)

        # 32.23 全角句号 1。与全角点 1．编号笑点正常识别解析
        raw_fullwidth_punct = (
            "笑点解析：\n"
            "1．程序员删库跑路\n"
            "2。运维经理当场破防\n"
            "令人忍俊不禁"
        )
        formatted23 = format_punchline_response(raw_fullwidth_punct)
        ok23, reason23 = validate_punchline_format(formatted23)
        self.assertTrue(ok23, f"全角标点编号解析失败: {reason23}\n{formatted23}")
        self.assertIn("笑点1：程序员删库跑路，很好笑。", formatted23)
        self.assertIn("笑点2：运维经理当场破防，很好笑。", formatted23)

        # 32.24 尾行“令人忍俊不禁哈哈”截断丢弃，绝不污染追加至最后一个笑点中
        raw_tail_variant = (
            "笑点解析：\n"
            "笑点1：猫猫翻车了\n"
            "令人忍俊不禁哈哈\n"
            "大家觉得好笑吗"
        )
        formatted24 = format_punchline_response(raw_tail_variant)
        ok24, reason24 = validate_punchline_format(formatted24)
        self.assertTrue(ok24, f"尾行变体截断失败: {reason24}\n{formatted24}")
        self.assertEqual(
            formatted24,
            "笑点解析：\n笑点1：猫猫翻车了，很好笑。\n令人忍俊不禁",
        )

        # 32.25 失败说明以“一段聊天记录不足以判断笑点”开头，文字不被误删首字
        raw_fail_yiduan = "一段聊天记录不足以判断笑点。"
        formatted25 = format_punchline_response(raw_fail_yiduan)
        ok25, reason25 = validate_punchline_format(formatted25)
        self.assertTrue(ok25, f"一段聊天记录失败说明规范化失败: {reason25}\n{formatted25}")
        self.assertIn("一段聊天记录不足以判断笑点", formatted25)
        self.assertNotIn("段聊天记录", formatted25.replace("一段聊天记录", ""))

        # 32.26 validate_punchline_format 严格拦截将失败说明编造成编号笑点的伪造内容
        pseudo_punchline = (
            "笑点解析：\n"
            "笑点1：未能读取视频画面，现有信息不足以判断笑点，很好笑。\n"
            "令人忍俊不禁"
        )
        ok26, reason26 = validate_punchline_format(pseudo_punchline)
        self.assertFalse(ok26, "校验器必须拦截伪造编号笑点")
        self.assertIn("失败/降级说明，严禁编造成编号笑点", reason26)

    # 33. V1 降级路径隔离断言测试：构造失败与解析异常下严格禁止调用共享 parse_url 与共享下载器
    async def test_v1_fallback_isolation_on_construction_failure_and_parse_exception(self):
        """
        验证两条降级路径：
        1. 独立解析器/下载器构造失败
        2. 独立解析器 parse 抛出异常
        断言：
        - 共享解析器 parse、parse_url 均未被调用
        - 共享下载器 streamd 绝未被调用
        - 返回受限提示，且注入封面非画面警告
        """
        single_shared_cfg = MagicMock()
        single_shared_cfg.source_max_size = 90
        single_shared_cfg.max_duration = 300
        single_shared_cfg.cache_dir = Path("/tmp/shared_cache")

        shared_downloader = MagicMock()
        shared_downloader.streamd = MagicMock()
        shared_downloader.max_size = 90

        class MockFailingDownloader:
            def __init__(self, cfg):
                raise RuntimeError("Downloader initialization simulated failure")

        class MockNormalPlatformParser:
            def __init__(self, cfg, downloader):
                self.cfg = cfg
                self.downloader = downloader

            async def parse(self, kw, match):
                raise RuntimeError("Independent parser internal exception")

        shared_platform = MagicMock()
        shared_platform.parse = AsyncMock()

        class MockPattern:
            def search(self, text):
                return True

        class MockParserPlugin:
            def __init__(self):
                self.downloader = shared_downloader
                self.cfg = single_shared_cfg
                self.parser_map = {
                    "fail_init": shared_platform,
                    "fail_parse": shared_platform,
                }
                self.key_pattern_list = [
                    ("fail_init", MockPattern()),
                    ("fail_parse", MockPattern()),
                ]
                self.parse_url = AsyncMock()

        plugin_inst = MockParserPlugin()

        class MockStarMetadata:
            star_cls = plugin_inst

        self.context.stars["astrbot_plugin_parser"] = MockStarMetadata()

        # 场景 1: 独立解析器/下载器构造失败
        shared_downloader.__class__ = MockFailingDownloader
        shared_platform.__class__ = MockNormalPlatformParser

        task_dir1 = self.plugin.temp_dir / "task_v1_fail_init"
        task_dir1.mkdir(parents=True, exist_ok=True)
        req_ctx1 = RequestContext(task_dir=task_dir1)

        res_fail_init = await self.plugin.parser_bridge.parse_url_with_existing_parser(
            url="http://domain.com/fail_init/test",
            max_media_mb=50,
            max_video_duration=180,
            task_dir=task_dir1,
            req_ctx=req_ctx1,
        )

        self.assertIsNotNone(res_fail_init)
        self.assertIn("已降级受限处理（封面不代表已读取视频画面）", res_fail_init.get("video_error", ""))
        self.assertFalse(shared_platform.parse.called, "严禁调用共享解析器 parse 方法")
        self.assertFalse(plugin_inst.parse_url.called, "降级路径严格禁止调用共享 parse_url")
        self.assertFalse(shared_downloader.streamd.called, "严禁在共享下载器上发起未受限下载")

        target_init = self.plugin.extractor._format_media_target(
            res_fail_init,
            supplementary_text="笑点解析",
            media_url="http://domain.com/fail_init/test",
        )
        prompt_init = build_user_prompt(
            target_description=target_init.description,
            text_content=target_init.text_content,
            supplementary_text=target_init.supplementary_text,
            degrade_note=target_init.degrade_note,
        )
        self.assertIn("封面不代表已读取视频画面", prompt_init)
        self.assertIn("已降级受限处理", prompt_init)
        await req_ctx1.close_normal()

        # 场景 2: 独立解析器 parse 抛出异常
        class MockWorkingDownloader:
            def __init__(self, cfg):
                self.cfg = cfg
                self.cache_dir = getattr(cfg, "cache_dir", None)
                self.streamd = MagicMock()

        shared_downloader.__class__ = MockWorkingDownloader
        task_dir2 = self.plugin.temp_dir / "task_v1_fail_parse"
        task_dir2.mkdir(parents=True, exist_ok=True)
        req_ctx2 = RequestContext(task_dir=task_dir2)

        res_fail_parse = await self.plugin.parser_bridge.parse_url_with_existing_parser(
            url="http://domain.com/fail_parse/test",
            max_media_mb=50,
            max_video_duration=180,
            task_dir=task_dir2,
            req_ctx=req_ctx2,
        )

        self.assertIsNotNone(res_fail_parse)
        self.assertIn("已降级受限处理（封面不代表已读取视频画面）", res_fail_parse.get("video_error", ""))
        self.assertFalse(shared_platform.parse.called, "解析异常时严禁调用共享解析器 parse 方法")
        self.assertFalse(plugin_inst.parse_url.called, "解析异常时降级路径严格禁止调用共享 parse_url")
        self.assertFalse(shared_downloader.streamd.called, "解析异常时严禁调用共享下载器 streamd")

        target_parse = self.plugin.extractor._format_media_target(
            res_fail_parse,
            supplementary_text="笑点解析",
            media_url="http://domain.com/fail_parse/test",
        )
        prompt_parse = build_user_prompt(
            target_description=target_parse.description,
            text_content=target_parse.text_content,
            supplementary_text=target_parse.supplementary_text,
            degrade_note=target_parse.degrade_note,
        )
        self.assertIn("封面不代表已读取视频画面", prompt_parse)
        self.assertIn("已降级受限处理", prompt_parse)
        await req_ctx2.close_normal()

    # 34. V1 降级路径下仅允许调用确认不会下载媒体的元数据接口
    async def test_v1_fallback_with_safe_readonly_metadata_interface(self):
        """
        当未构建独立上下文或独立上下文不可用时，若共享解析器提供了确认只读的元数据接口：
        允许调用只读元数据接口获取信息，但严格禁止调用共享 parse_url 与 streamd
        """
        class MockSafeMetadataParserPlugin:
            def __init__(self):
                self.key_pattern_list = []
                self.parser_map = {}
                self.downloader = MagicMock()
                self.downloader.streamd = MagicMock()
                self.parse_url = AsyncMock()

            async def get_metadata(self, url: str):
                res = MagicMock()
                res.title = "安全只读元数据标题"
                res.text = "只读视频简介"
                res.author = MagicMock(name="author")
                res.author.name = "元数据作者"
                res.extra = {"info": "AI总结"}
                res.cover_path = None
                res.img_contents = []
                return res

        MockSafeMetadataParserPlugin.get_metadata.is_metadata_only = True
        safe_plugin = MockSafeMetadataParserPlugin()

        class MockStarMetadata:
            star_cls = safe_plugin

        self.context.stars["astrbot_plugin_parser"] = MockStarMetadata()

        task_dir = self.plugin.temp_dir / "task_v1_meta"
        task_dir.mkdir(parents=True, exist_ok=True)
        req_ctx = RequestContext(task_dir=task_dir)

        res_meta = await self.plugin.parser_bridge.parse_url_with_existing_parser(
            url="http://domain.com/some_video",
            max_media_mb=50,
            max_video_duration=180,
            task_dir=task_dir,
            req_ctx=req_ctx,
        )

        self.assertIsNotNone(res_meta)
        self.assertEqual(res_meta.get("title"), "安全只读元数据标题")
        self.assertEqual(res_meta.get("text"), "只读视频简介")
        self.assertFalse(safe_plugin.parse_url.called, "共享 parse_url 严禁被调用")
        self.assertFalse(safe_plugin.downloader.streamd.called, "共享下载器 streamd 严禁被调用")
        await req_ctx.close_normal()

    # 35. V2 同一 URL 并发隔离与专属 cache_dir 测试
    async def test_v2_concurrent_same_url_directory_and_cache_isolation(self):
        """
        针对同一 URL 的并发请求，断言：
        1. 各请求分配完全独立的 task_dir
        2. 各请求的 isolated_downloader.cache_dir 严格指向自身的 task_dir
        3. 共享单例的 cache_dir 与配置不受并发请求影响与污染
        4. 并发执行无文件冲突或死锁
        """
        shared_cfg = MagicMock()
        shared_cfg.source_max_size = 90
        shared_cfg.max_duration = 300
        shared_cfg.cache_dir = Path("/tmp/global_cache")

        created_downloaders = []

        class MockDownloader:
            def __init__(self, cfg):
                self.cfg = cfg
                self.cache_dir = getattr(cfg, "cache_dir", None)
                created_downloaders.append(self)

        class MockPlatformParser:
            def __init__(self, cfg, downloader):
                self.cfg = cfg
                self.downloader = downloader

            async def parse(self, kw, match):
                await asyncio.sleep(0.05)
                res = MagicMock()
                res.title = f"标题-{id(self)}"
                res.text = "内容"
                res.author = None
                res.extra = None
                res.cover_path = None
                res.img_contents = []
                res.video_contents = []
                return res

        class MockPattern:
            def search(self, text):
                return True

        class MockParserPlugin:
            def __init__(self):
                self.downloader = MockDownloader(shared_cfg)
                self.cfg = shared_cfg
                self.parser_map = {
                    "concurrent_kw": MockPlatformParser(shared_cfg, self.downloader),
                }
                self.key_pattern_list = [
                    ("concurrent_kw", MockPattern()),
                ]

        parser_plugin = MockParserPlugin()
        class MockStarMetadata:
            star_cls = parser_plugin

        self.context.stars["astrbot_plugin_parser"] = MockStarMetadata()

        task_dir1 = self.plugin.temp_dir / f"task_conc_{uuid.uuid4().hex}"
        task_dir2 = self.plugin.temp_dir / f"task_conc_{uuid.uuid4().hex}"
        task_dir1.mkdir(parents=True, exist_ok=True)
        task_dir2.mkdir(parents=True, exist_ok=True)

        req_ctx1 = RequestContext(task_dir=task_dir1)
        req_ctx2 = RequestContext(task_dir=task_dir2)

        same_url = "http://domain.com/concurrent_kw/same_video_123"

        t1 = asyncio.create_task(
            self.plugin.parser_bridge.parse_url_with_existing_parser(
                url=same_url,
                max_media_mb=50,
                max_video_duration=180,
                task_dir=task_dir1,
                req_ctx=req_ctx1,
            )
        )
        t2 = asyncio.create_task(
            self.plugin.parser_bridge.parse_url_with_existing_parser(
                url=same_url,
                max_media_mb=50,
                max_video_duration=180,
                task_dir=task_dir2,
                req_ctx=req_ctx2,
            )
        )

        res1, res2 = await asyncio.gather(t1, t2)
        self.assertIsNotNone(res1)
        self.assertIsNotNone(res2)

        self.assertNotEqual(str(task_dir1), str(task_dir2))
        self.assertGreaterEqual(len(created_downloaders), 3)

        iso_dl1 = created_downloaders[1]
        iso_dl2 = created_downloaders[2]
        self.assertEqual(iso_dl1.cache_dir, task_dir1)
        self.assertEqual(iso_dl2.cache_dir, task_dir2)
        self.assertNotEqual(iso_dl1.cache_dir, iso_dl2.cache_dir)
        self.assertEqual(shared_cfg.cache_dir, Path("/tmp/global_cache"), "共享配置未被篡改")

        await req_ctx1.close_normal()
        await req_ctx2.close_normal()

    # 36. V2 受控流式字节上限阻断与部分写入残留文件即时清理测试
    async def test_v2_controlled_streamed_bytes_limit_and_partial_file_cleanup(self):
        """
        验证流式下载过程中的受控字节上限阻断：
        1. 模拟分块流式数据传输（4 块，每块 32KB = 128KB）
        2. 设置流式下载上限为 50KB（51200 字节）
        3. 在累积写入第 2 块（64KB > 50KB）时触发即时阻断
        4. 断言 download_file_with_limit 返回 False
        5. 断言未完成的部分写入文件（partial file）被立即删除，磁盘无残留
        """
        from astrbot_plugin_punchline_analyzer.parser_bridge import download_file_with_limit

        task_dir = self.plugin.temp_dir / f"stream_limit_test_{uuid.uuid4().hex}"
        task_dir.mkdir(parents=True, exist_ok=True)
        out_file = task_dir / "test_partial_video.mp4"

        chunk_size = 32 * 1024
        chunks = [b"A" * chunk_size, b"B" * chunk_size, b"C" * chunk_size, b"D" * chunk_size]

        class MockStreamContent:
            def __init__(self, data_chunks):
                self.data_chunks = data_chunks

            async def iter_chunked(self, n):
                for c in self.data_chunks:
                    await asyncio.sleep(0.01)
                    yield c

        class MockStreamResponse:
            def __init__(self, data_chunks):
                self.status = 200
                self.headers = {}
                self.content = MockStreamContent(data_chunks)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        class MockSession:
            def __init__(self, data_chunks):
                self.data_chunks = data_chunks
                self.closed = False

            def get(self, url, *args, **kwargs):
                return MockStreamResponse(self.data_chunks)

            async def close(self):
                self.closed = True

        mock_session = MockSession(chunks)
        max_bytes_limit = 50 * 1024

        req_ctx = RequestContext(task_dir=task_dir)

        success = await download_file_with_limit(
            url="http://mock.stream/video.mp4",
            output_path=out_file,
            max_bytes=max_bytes_limit,
            session=mock_session,
            req_ctx=req_ctx,
        )

        self.assertFalse(success, "超过受控字节上限必须返回 False 阻断")
        self.assertFalse(out_file.exists(), "触发字节上限后部分写入的 partial file 必须立即被清理删除")

        async def failing_iter(n):
            yield b"X" * 1024
            raise RuntimeError("Network stream disrupted")

        failing_content = MockStreamContent([])
        failing_content.iter_chunked = failing_iter
        failing_resp = MockStreamResponse([])
        failing_resp.content = failing_content
        mock_session.get = lambda url: failing_resp

        out_file_fail = task_dir / "test_partial_fail.mp4"
        success_fail = await download_file_with_limit(
            url="http://mock.stream/failing.mp4",
            output_path=out_file_fail,
            max_bytes=max_bytes_limit,
            session=mock_session,
            req_ctx=req_ctx,
        )
        self.assertFalse(success_fail)
        self.assertFalse(out_file_fail.exists(), "异常中断后 partial file 必须被即时删除")

        await req_ctx.close_normal()
        self.assertFalse(task_dir.exists())

    # 37. V2 中途取消与异常超时清理时序测试
    async def test_v2_mid_cancellation_and_timeout_cleanup_sequence(self):
        """
        验证异常、超时或中途取消流程的精确清理时序：
        1. 显式取消所有后台异步任务 (download/stream/sample)
        2. await asyncio.gather 等待所有任务彻底退出（done() is True）
        3. 安全关闭 ClientSession 与网络连接
        4. 彻底清理专属任务目录与临时文件，磁盘零残留
        """
        task_dir = self.plugin.temp_dir / f"cancel_seq_{uuid.uuid4().hex}"
        task_dir.mkdir(parents=True, exist_ok=True)
        dummy_file = task_dir / "temp_download.part"
        dummy_file.write_bytes(b"temp_data_content")

        extra_temp = Path(tempfile.gettempdir()) / f"extra_temp_{uuid.uuid4().hex}.jpg"
        extra_temp.write_bytes(b"extra_frame")

        req_ctx = RequestContext(task_dir=task_dir)
        req_ctx.add_temp_file(extra_temp)

        task_cancelled_event = asyncio.Event()

        async def background_streaming_job():
            try:
                while True:
                    await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                task_cancelled_event.set()
                raise

        bg_task = asyncio.create_task(background_streaming_job())
        req_ctx.track_task(bg_task)
        await asyncio.sleep(0.01)

        class MockSession:
            def __init__(self):
                self.closed = False

            async def close(self):
                self.closed = True

        mock_session = MockSession()
        req_ctx.track_session(mock_session)

        self.assertFalse(bg_task.done())
        self.assertFalse(mock_session.closed)
        self.assertTrue(task_dir.exists())
        self.assertTrue(dummy_file.exists())
        self.assertTrue(extra_temp.exists())

        await req_ctx.cancel_and_cleanup()

        self.assertTrue(bg_task.done(), "后台任务必须已彻底退出 (done)")
        self.assertTrue(task_cancelled_event.is_set(), "后台任务的 CancelledError 必须被妥善捕获与触发")
        self.assertTrue(mock_session.closed, "ClientSession 必须已被安全关闭")
        self.assertFalse(task_dir.exists(), "专属任务目录必须已被完全删除")
        self.assertFalse(dummy_file.exists(), "任务目录内文件必须已不存在")
        self.assertFalse(extra_temp.exists(), "外部临时文件必须已被彻底删除")

    # 38. V1 降级路径下未显式标注 is_metadata_only 的伪元数据接口严禁被调用
    async def test_v1_fallback_unmarked_metadata_not_called(self):
        class MockUnsafePlugin:
            def __init__(self):
                self.key_pattern_list = []
                self.parser_map = {}
                self.downloader = MagicMock()
                self.downloader.streamd = MagicMock()
                self.get_info = AsyncMock()

        unsafe_plugin = MockUnsafePlugin()
        class MockStarMetadata:
            star_cls = unsafe_plugin

        self.context.stars["astrbot_plugin_parser"] = MockStarMetadata()

        task_dir = self.plugin.temp_dir / f"task_unsafe_{uuid.uuid4().hex}"
        task_dir.mkdir(parents=True, exist_ok=True)
        req_ctx = RequestContext(task_dir=task_dir)

        res = await self.plugin.parser_bridge.parse_url_with_existing_parser(
            url="http://domain.com/unmarked_info",
            max_media_mb=50,
            max_video_duration=180,
            task_dir=task_dir,
            req_ctx=req_ctx,
        )

        self.assertIsNotNone(res)
        self.assertFalse(unsafe_plugin.get_info.called, "未标注 is_metadata_only=True 的接口绝不能被调用")
        self.assertIn("已降级受限处理", res.get("video_error", ""))
        await req_ctx.close_normal()

    # 39. V1 降级路径下图集与封面严禁调用共享下载器下载
    async def test_v1_fallback_images_and_cover_never_download_on_shared(self):
        shared_dl = MagicMock()
        shared_dl.download_img = AsyncMock()
        shared_dl.download_file = AsyncMock()
        shared_dl.streamd = MagicMock()

        class MockMetadataWithMedia:
            def __init__(self):
                self.key_pattern_list = []
                self.parser_map = {}
                self.downloader = shared_dl

            async def get_metadata(self, url: str):
                class MockImg:
                    async def get_path(self):
                        return await shared_dl.download_img("img.jpg")

                res = MagicMock()
                res.title = "只读元数据"
                res.text = "只读简介"
                res.author = None
                res.extra = None
                # 封面为下载协程函数
                async def mock_cover_download():
                    return await shared_dl.download_file("cover.jpg")
                res.cover_path = mock_cover_download
                res.img_contents = [MockImg()]
                res.video_contents = []
                return res

        MockMetadataWithMedia.get_metadata.is_metadata_only = True
        plugin_inst = MockMetadataWithMedia()

        class MockStarMetadata:
            star_cls = plugin_inst

        self.context.stars["astrbot_plugin_parser"] = MockStarMetadata()

        task_dir = self.plugin.temp_dir / f"task_meta_media_{uuid.uuid4().hex}"
        task_dir.mkdir(parents=True, exist_ok=True)
        req_ctx = RequestContext(task_dir=task_dir)

        res = await self.plugin.parser_bridge.parse_url_with_existing_parser(
            url="http://domain.com/meta_with_media",
            max_media_mb=50,
            max_video_duration=180,
            task_dir=task_dir,
            req_ctx=req_ctx,
        )

        self.assertIsNotNone(res)
        self.assertFalse(shared_dl.download_img.called, "降级路径下严禁调用共享下载器下载图集")
        self.assertFalse(shared_dl.download_file.called, "降级路径下严禁调用共享下载器下载封面")
        self.assertFalse(shared_dl.streamd.called, "严禁调用共享 streamd")
        self.assertEqual(res.get("images"), [], "降级时未下载的封面与图集严禁混入")
        await req_ctx.close_normal()

    # 40. V2 平台短域名正则匹配与独立上下文构建测试（如 b23.tv 短链）
    async def test_v2_short_domain_regex_matching_isolated_context(self):
        import re

        shared_cfg = MagicMock()
        shared_cfg.source_max_size = 90
        shared_cfg.max_duration = 300
        shared_cfg.cache_dir = Path("/tmp/global_cache")

        class MockDownloader:
            def __init__(self, cfg):
                self.cfg = cfg
                self.cache_dir = getattr(cfg, "cache_dir", None)

        class MockPlatformParser:
            def __init__(self, cfg, downloader):
                self.cfg = cfg
                self.downloader = downloader

            async def parse(self, kw, match):
                res = MagicMock()
                res.title = "B站短链视频"
                res.text = "短链简介"
                res.author = None
                res.extra = None
                res.cover_path = None
                res.img_contents = []
                res.video_contents = []
                return res

        class MockParserPlugin:
            def __init__(self):
                self.downloader = MockDownloader(shared_cfg)
                self.cfg = shared_cfg
                self.parser_map = {
                    "bilibili": MockPlatformParser(shared_cfg, self.downloader),
                }
                # 正则匹配 b23.tv 短链，注意 kw 为 "bilibili"，不在短链 URL 字符中
                self.key_pattern_list = [
                    ("bilibili", re.compile(r"https?://(?:www\.)?(?:bilibili\.com|b23\.tv)/\S+")),
                ]

        plugin_inst = MockParserPlugin()
        class MockStarMetadata:
            star_cls = plugin_inst

        self.context.stars["astrbot_plugin_parser"] = MockStarMetadata()

        task_dir = self.plugin.temp_dir / f"task_short_url_{uuid.uuid4().hex}"
        task_dir.mkdir(parents=True, exist_ok=True)
        req_ctx = RequestContext(task_dir=task_dir)

        # 发送 b23.tv 短链
        res = await self.plugin.parser_bridge.parse_url_with_existing_parser(
            url="https://b23.tv/BV1xx411c7mD",
            max_media_mb=50,
            max_video_duration=180,
            task_dir=task_dir,
            req_ctx=req_ctx,
        )

        self.assertIsNotNone(res)
        self.assertEqual(res.get("title"), "B站短链视频")
        self.assertEqual(res.get("video_error"), "", "短链必须成功命中正则并构建独立解析器，绝不可误判降级")
        await req_ctx.close_normal()

    # 41. V2 下载器协程方法（async def）任务自动拦截与取消测试
    async def test_v2_downloader_coroutine_tasks_tracked_and_cancelled(self):
        from astrbot_plugin_punchline_analyzer.parser_bridge import (
            IsolatedParserContext,
            _instrument_downloader_tasks,
        )

        task_dir = self.plugin.temp_dir / f"task_coro_track_{uuid.uuid4().hex}"
        task_dir.mkdir(parents=True, exist_ok=True)
        req_ctx = RequestContext(task_dir=task_dir)
        iso_ctx = IsolatedParserContext()

        bg_task_cancelled = asyncio.Event()

        class MockAsyncDownloader:
            async def download_video(self):
                try:
                    while True:
                        await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    bg_task_cancelled.set()
                    raise

        dl = MockAsyncDownloader()
        _instrument_downloader_tasks(dl, iso_ctx, req_ctx)

        # 模拟解析器内部通过 asyncio.create_task 启动了后台协程
        spawned_task = asyncio.create_task(dl.download_video())
        await asyncio.sleep(0.01)

        self.assertIn(spawned_task, iso_ctx.tasks, "派生的协程 Task 必须被自动捕获到 iso_ctx.tasks")
        self.assertIn(spawned_task, req_ctx.tasks, "派生的协程 Task 必须被自动捕获到 req_ctx.tasks")

        await req_ctx.cancel_and_cleanup()

        self.assertTrue(spawned_task.done())
        self.assertTrue(bg_task_cancelled.is_set(), "后台协程 Task 必须被 cancel 并等待退出")
        self.assertFalse(task_dir.exists())

    # 42. V2 close_normal 正常清理前等待并取消残留未完成任务
    async def test_v2_close_normal_waits_for_lingering_tasks(self):
        task_dir = self.plugin.temp_dir / f"task_normal_wait_{uuid.uuid4().hex}"
        task_dir.mkdir(parents=True, exist_ok=True)
        req_ctx = RequestContext(task_dir=task_dir)

        task_started = asyncio.Event()
        task_cleaned = asyncio.Event()

        async def lingering_task():
            task_started.set()
            try:
                while True:
                    await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                task_cleaned.set()
                raise

        lingering = asyncio.create_task(lingering_task())
        req_ctx.track_task(lingering)
        await task_started.wait()

        # 正常流程完成时调用 close_normal
        await req_ctx.close_normal()

        self.assertTrue(lingering.done(), "close_normal 必须确保挂起任务退出")
        self.assertTrue(task_cleaned.is_set(), "挂起任务必须收到 CancelledError")
        self.assertFalse(task_dir.exists(), "目录必须已被完全删除")

    # 43. V2 cleanup_disk 支持只读文件属性清除与重试机制
    async def test_v2_cleanup_disk_retries_and_cleans_readonly(self):
        import stat

        task_dir = self.plugin.temp_dir / f"task_readonly_{uuid.uuid4().hex}"
        task_dir.mkdir(parents=True, exist_ok=True)
        readonly_file = task_dir / "readonly.part"
        readonly_file.write_bytes(b"locked data")

        # 将文件设置为只读属性（模拟 Windows 只读或权限锁定状态）
        os.chmod(str(readonly_file), stat.S_IREAD)

        req_ctx = RequestContext(task_dir=task_dir)
        req_ctx.add_temp_file(readonly_file)

        await req_ctx.close_normal()

    # 44. V2 直接 await 下载器协程方法绝不引发主任务自取消与 gather 递归死锁
    async def test_v2_directly_awaited_downloader_no_self_cancel_deadlock(self):
        from astrbot_plugin_punchline_analyzer.parser_bridge import (
            IsolatedParserContext,
            _instrument_downloader_tasks,
        )

        task_dir = self.plugin.temp_dir / f"task_self_deadlock_{uuid.uuid4().hex}"
        task_dir.mkdir(parents=True, exist_ok=True)
        req_ctx = RequestContext(task_dir=task_dir)
        iso_ctx = IsolatedParserContext()

        class MockDownloader:
            async def download_video(self, url: str):
                return "direct_video_path.mp4"

        dl = MockDownloader()
        _instrument_downloader_tasks(dl, iso_ctx, req_ctx)

        # 在主请求协程中直接 await 协程方法（而非放入单独 create_task）
        res = await dl.download_video("http://domain.com/v.mp4")
        self.assertEqual(res, "direct_video_path.mp4")

        # 主任务自身绝不应该被加入后台等待队列
        cur = asyncio.current_task()
        self.assertNotIn(cur, req_ctx.tasks, "请求主任务绝不能被登记为待取消/待等待的后台任务")
        self.assertNotIn(cur, iso_ctx.tasks, "请求主任务绝不能被登记为待取消/待等待的后台任务")

        # 正常清理与异常清理均必须瞬间安全退出，绝无递归或 gather 死锁
        await req_ctx.close_normal()
        self.assertFalse(task_dir.exists())

    # 45. V2 降级路径下既有外部共享缓存文件绝不被加入 temp_files 与误删
    async def test_v2_fallback_existing_cache_files_never_deleted_on_cleanup(self):
        # 模拟外部共享缓存文件（位于任务目录之外）
        external_shared_cover = Path(tempfile.gettempdir()) / f"shared_cover_{uuid.uuid4().hex}.jpg"
        external_shared_cover.write_bytes(b"pre_existing_shared_cover_content")

        shared_dl = MagicMock()
        class MockReadOnlyMetadataPlugin:
            def __init__(self):
                self.key_pattern_list = []
                self.parser_map = {}
                self.downloader = shared_dl

            async def get_metadata(self, url: str):
                res = MagicMock()
                res.title = "只读元数据"
                res.text = ""
                res.author = None
                res.extra = None
                res.cover_path = str(external_shared_cover)
                res.img_contents = []
                res.video_contents = []
                return res

        MockReadOnlyMetadataPlugin.get_metadata.is_metadata_only = True
        plugin_inst = MockReadOnlyMetadataPlugin()
        class MockStarMetadata:
            star_cls = plugin_inst
        self.context.stars["astrbot_plugin_parser"] = MockStarMetadata()

        task_dir = self.plugin.temp_dir / f"task_fallback_cache_{uuid.uuid4().hex}"
        task_dir.mkdir(parents=True, exist_ok=True)
        req_ctx = RequestContext(task_dir=task_dir)

        res = await self.plugin.parser_bridge.parse_url_with_existing_parser(
            url="http://domain.com/fallback_meta_test",
            max_media_mb=50,
            max_video_duration=180,
            task_dir=task_dir,
            req_ctx=req_ctx,
        )

        self.assertIsNotNone(res)
        self.assertEqual(res.get("cover_path"), str(external_shared_cover))
        # 核心断言：降级复用的外部共享缓存文件绝不能进入本请求专属的 temp_files
        self.assertNotIn(str(external_shared_cover), [str(p) for p in req_ctx.temp_files])

        await req_ctx.close_normal()

        # 任务目录被删除，但外部共享文件必须完好无损存在
        self.assertFalse(task_dir.exists())
        self.assertTrue(external_shared_cover.exists(), "外部共享缓存文件绝不能在清理时被误删！")
        try:
            external_shared_cover.unlink()
        except Exception:
            pass

    # 46. V2 IsolatedConfigProxy 字段完备性测试（max_size, data_dir, source_max_minute）
    async def test_v2_isolated_config_proxy_complete_fields(self):
        from astrbot_plugin_punchline_analyzer.parser_bridge import IsolatedConfigProxy

        shared_cfg = MagicMock()
        shared_cfg.source_max_size = 90
        shared_cfg.max_size = 90 * 1024 * 1024
        shared_cfg.max_duration = 300
        shared_cfg.data_dir = Path("/opt/astrbot/data/plugin_data/shared_data")

        task_dir = self.plugin.temp_dir / f"task_cfg_proxy_{uuid.uuid4().hex}"
        iso_cfg = IsolatedConfigProxy(
            orig_cfg=shared_cfg,
            max_media_mb=50,
            max_video_duration=180,
            task_dir=task_dir,
        )

        self.assertEqual(iso_cfg.source_max_size, 50)
        self.assertEqual(iso_cfg.max_size, 50 * 1024 * 1024, "max_size 字节数必须为隔离的 50MB")
        self.assertEqual(iso_cfg.max_duration, 180)
        self.assertEqual(iso_cfg.source_max_minute, 3)
        self.assertEqual(iso_cfg.cache_dir, task_dir)
        self.assertEqual(iso_cfg.data_dir, task_dir, "data_dir 必须重定向至专属任务目录，防止写盘污染")
        self.assertEqual(iso_cfg.download_dir, task_dir)

    # 47. V2 字符串正则与关键词退化匹配对象有效性测试
    async def test_v2_isolated_parser_string_regex_and_keyword_match(self):
        from astrbot_plugin_punchline_analyzer.parser_bridge import create_isolated_parser_context

        class MockDownloader:
            def __init__(self, cfg):
                self.cfg = cfg

        class MockPlatformParser:
            def __init__(self, cfg, downloader):
                self.cfg = cfg
                self.downloader = downloader

            async def parse(self, kw, match):
                # 断言 match 必须具备 group 方法，绝不能是单纯的 bool
                assert hasattr(match, "group"), "match 必须是拥有 group 方法的正则匹配对象"
                res = MagicMock()
                res.title = "解析成功"
                res.text = ""
                res.author = None
                res.extra = None
                res.cover_path = None
                res.video_contents = []
                res.img_contents = []
                return res

        class MockPluginWithStringRegex:
            def __init__(self):
                self.cfg = MagicMock()
                self.downloader = MockDownloader(self.cfg)
                self.parser_map = {
                    "test_kw": MockPlatformParser(self.cfg, self.downloader),
                }
                # 未经预编译的裸字符串正则
                self.key_pattern_list = [
                    ("test_kw", r"test_kw/video/(?P<id>\d+)"),
                ]

        plugin_inst = MockPluginWithStringRegex()
        task_dir = self.plugin.temp_dir / f"task_str_regex_{uuid.uuid4().hex}"
        iso_ctx, kw, m = await create_isolated_parser_context(
            parser_plugin=plugin_inst,
            url="http://domain.com/test_kw/video/123456",
            task_dir=task_dir,
        )

        self.assertIsNotNone(iso_ctx)
        self.assertEqual(kw, "test_kw")
        self.assertIsNotNone(m)
        self.assertTrue(hasattr(m, "group"))
        self.assertEqual(m.group("id"), "123456")

        # 验证调用 parse 不会抛出 'bool' object has no attribute 'group'
        parse_res = await iso_ctx.parser.parse(kw, m)
        self.assertEqual(parse_res.title, "解析成功")
        await iso_ctx.close()

    # 48. V2 超限多视频任务取消时必须显式等待退出
    async def test_v2_duration_limit_cancels_and_awaits_all_video_contents(self):
        task_dir = self.plugin.temp_dir / f"task_dur_multi_{uuid.uuid4().hex}"
        task_dir.mkdir(parents=True, exist_ok=True)
        req_ctx = RequestContext(task_dir=task_dir)

        task1_cleaned = asyncio.Event()
        task2_cleaned = asyncio.Event()

        async def video_download_1():
            try:
                while True:
                    await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                task1_cleaned.set()
                raise

        async def video_download_2():
            try:
                while True:
                    await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                task2_cleaned.set()
                raise

        t1 = asyncio.create_task(video_download_1())
        t2 = asyncio.create_task(video_download_2())

        class MockVC1:
            duration = 300.0  # 超过 180s
            path_task = t1

        class MockVC2:
            duration = 300.0
            path_task = t2

        class MockDownloader:
            def __init__(self, cfg):
                self.cfg = cfg

        class MockParser:
            def __init__(self, cfg, dl):
                pass
            async def parse(self, kw, match):
                res = MagicMock()
                res.title = "超长多视频"
                res.text = ""
                res.author = None
                res.extra = None
                res.cover_path = None
                res.video_contents = [MockVC1(), MockVC2()]
                res.img_contents = []
                return res

        class MockPlugin:
            def __init__(self):
                self.cfg = MagicMock()
                self.downloader = MockDownloader(self.cfg)
                self.parser_map = {"multi_v": MockParser(self.cfg, self.downloader)}
                self.key_pattern_list = [("multi_v", lambda u: True)]

        plugin_inst = MockPlugin()
        class MockStarMetadata:
            star_cls = plugin_inst
        self.context.stars["astrbot_plugin_parser"] = MockStarMetadata()

        res = await self.plugin.parser_bridge.parse_url_with_existing_parser(
            url="http://domain.com/multi_v/test",
            max_media_mb=50,
            max_video_duration=180,
            task_dir=task_dir,
            req_ctx=req_ctx,
        )

        self.assertIsNotNone(res)
        self.assertIn("超过系统上限 180秒", res.get("video_error", ""))
        self.assertTrue(t1.done(), "t1 必须已被取消并等待退出")
        self.assertTrue(t2.done(), "t2 必须已被取消并等待退出")
        self.assertTrue(task1_cleaned.is_set())
        self.assertTrue(task2_cleaned.is_set())
        await req_ctx.close_normal()

    def test_v3_isolated_config_proxy_get_and_getitem_and_save_isolation(self):
        """【审查缺陷 1 回归测试】：断言 IsolatedConfigProxy 的 get、__getitem__、__contains__ 与 save_config 严格隔离，绝不穿透至共享配置或写盘"""
        from astrbot_plugin_punchline_analyzer.parser_bridge import IsolatedConfigProxy

        class MockOrigCfg(dict):
            def __init__(self):
                super().__init__({
                    "cache_dir": Path("/shared/global/cache"),
                    "data_dir": Path("/shared/global/data"),
                    "max_size": 90 * 1024 * 1024,
                    "source_max_size": 90,
                })
                self.save_called = False
            def save_config(self):
                self.save_called = True

        orig = MockOrigCfg()
        task_dir = Path("/tmp/punchline_isolated_task_123")
        proxy = IsolatedConfigProxy(orig, max_media_mb=50, max_video_duration=180, task_dir=task_dir)

        # 属性访问验证
        self.assertEqual(proxy.cache_dir, task_dir)
        self.assertEqual(proxy.max_size, 50 * 1024 * 1024)

        # 字典访问验证 (__getitem__ 与 get)
        self.assertEqual(proxy["cache_dir"], task_dir)
        self.assertEqual(proxy.get("cache_dir"), task_dir)
        self.assertEqual(proxy["max_size"], 50 * 1024 * 1024)
        self.assertEqual(proxy.get("max_size"), 50 * 1024 * 1024)
        self.assertTrue("cache_dir" in proxy)

        # 磁盘保存防御验证
        proxy.save_config()
        self.assertFalse(orig.save_called, "save_config 绝不能污染原配置并写盘")

    async def test_v3_downloader_ytdlp_tasks_instrumented_and_cancelled(self):
        """【审查缺陷 2 回归测试】：断言 ytdlp 等所有派生下载任务被自动拦截并纳入纳管，并在 cancel_and_cleanup 时妥善取消并等待退出"""
        from astrbot_plugin_punchline_analyzer.parser_bridge import (
            IsolatedParserContext,
            RequestContext,
            _instrument_downloader_tasks,
        )

        task_dir = self.plugin.temp_dir / f"task_ytdlp_{uuid.uuid4().hex}"
        task_dir.mkdir(parents=True, exist_ok=True)
        req_ctx = RequestContext(task_dir=task_dir)
        iso_ctx = IsolatedParserContext()

        ytdlp_cancelled = asyncio.Event()

        class MockYtdlpDownloader:
            def __init__(self):
                pass
            async def ytdlp_download_video(self, url, **kwargs):
                try:
                    while True:
                        await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    ytdlp_cancelled.set()
                    raise

            async def ytdlp_download_audio(self, url, **kwargs):
                await asyncio.sleep(0.01)
                return Path("/tmp/audio.flac")

        dl = MockYtdlpDownloader()
        _instrument_downloader_tasks(dl, iso_ctx, req_ctx)

        # 调用 ytdlp_download_video
        dl_coro = dl.ytdlp_download_video("https://youtube.com/watch?v=123")
        ytdlp_task = asyncio.create_task(dl_coro)
        await asyncio.sleep(0.01)

        # 断言 task 已被 req_ctx 与 iso_ctx 纳管
        self.assertIn(ytdlp_task, req_ctx.tasks)
        self.assertIn(ytdlp_task, iso_ctx.tasks)

        # 执行 cancel_and_cleanup
        await req_ctx.cancel_and_cleanup()
        self.assertTrue(ytdlp_task.done())
        self.assertTrue(ytdlp_cancelled.is_set())
        self.assertFalse(task_dir.exists())

    async def test_v3_request_context_cleanup_guaranteed_on_exceptions(self):
        """【审查缺陷 3 回归测试】：断言在 session.close 或任务等待抛出异常时，try...finally 结构确保 cleanup_disk 必定执行，零残留"""
        task_dir = self.plugin.temp_dir / f"task_err_guarantee_{uuid.uuid4().hex}"
        task_dir.mkdir(parents=True, exist_ok=True)
        dummy_file = task_dir / "leftover.tmp"
        dummy_file.write_text("hello", encoding="utf-8")

        req_ctx = RequestContext(task_dir=task_dir)

        # 模拟一个 close 会抛出异常的恶劣 session
        class FaultySession:
            closed = False
            async def close(self):
                self.closed = True
                raise RuntimeError("Simulated network session error during close")

        req_ctx.track_session(FaultySession())

        # 执行 cancel_and_cleanup，即使 session.close 抛出异常，磁盘必须彻底清除
        await req_ctx.cancel_and_cleanup()
        self.assertFalse(task_dir.exists(), "即便发生异常，专属任务目录必须被成功删除")
        self.assertFalse(dummy_file.exists())

    async def test_v3_download_file_with_limit_enforces_timeout_on_passed_session(self):
        """【审查缺陷 4 回归测试】：断言传入外部已存在 session 时，download_file_with_limit 依然严格强制执行 client_timeout"""
        from astrbot_plugin_punchline_analyzer.parser_bridge import download_file_with_limit

        task_dir = self.plugin.temp_dir / f"task_dl_to_{uuid.uuid4().hex}"
        task_dir.mkdir(parents=True, exist_ok=True)
        dest_file = task_dir / "timeout_test.mp4"

        # 传入一个无超时的 session，模拟服务端无限阻塞
        class HangingSession:
            closed = False
            def get(self, url, timeout=None, *args, **kwargs):
                class AsyncContext:
                    async def __aenter__(self):
                        total = getattr(timeout, "total", None) if timeout else None
                        if total and total <= 0.2:
                            raise asyncio.TimeoutError("Simulated socket timeout")
                        await asyncio.sleep(10.0)
                        return MagicMock(status=200)
                    async def __aexit__(self, *args):
                        pass
                return AsyncContext()
            async def close(self):
                self.closed = True

        sess = HangingSession()
        res = await download_file_with_limit(
            url="http://mock.hang/stream.mp4",
            output_path=dest_file,
            max_bytes=1024 * 1024,
            timeout=0.1,  # 0.1s 超时
            session=sess,
        )
        self.assertFalse(res)
        self.assertFalse(dest_file.exists())
        shutil.rmtree(task_dir, ignore_errors=True)

    async def test_v3_video_sampling_short_duration_no_collapse(self):
        """【审查缺陷 5 回归测试】：断言极短视频（<= 1.0s）抽帧时间戳自适应缩放不越界，且生成空文件时即时清理"""
        from astrbot_plugin_punchline_analyzer.parser_bridge import sample_video_frames

        task_dir = self.plugin.temp_dir / f"task_short_sample_{uuid.uuid4().hex}"
        task_dir.mkdir(parents=True, exist_ok=True)
        fake_video = task_dir / "short.mp4"
        fake_video.write_bytes(b"dummy")

        with patch("astrbot_plugin_punchline_analyzer.parser_bridge.get_video_duration", new_callable=AsyncMock) as mock_dur:
            mock_dur.return_value = 0.4  # 极短 0.4 秒视频
            with patch("astrbot_plugin_punchline_analyzer.parser_bridge.run_cancellable_subprocess", new_callable=AsyncMock) as mock_sub:
                captured_cmds = []
                async def fake_sub(*cmd, **kwargs):
                    captured_cmds.append(cmd)
                    out_path = Path(cmd[-1])
                    out_path.write_bytes(b"jpgdata")
                    return 0, b"", b""
                mock_sub.side_effect = fake_sub

                frames = await sample_video_frames(
                    video_path=fake_video,
                    output_dir=task_dir,
                    num_frames=4,
                    max_duration=180,
                    max_media_mb=50,
                )
                self.assertEqual(len(frames), 4)
                # 校验时间戳绝不能超过视频总长度 0.4s
                for cmd in captured_cmds:
                    ss_idx = cmd.index("-ss")
                    ts = float(cmd[ss_idx + 1])
                    self.assertLess(ts, 0.4, f"抽帧时间戳 {ts} 绝不能超过视频总时长 0.4s")
                    self.assertGreaterEqual(ts, 0.0)

        shutil.rmtree(task_dir, ignore_errors=True)

    async def test_v3_duration_limit_cancels_and_awaits_cover_tasks(self):
        """【审查缺陷 6 回归测试】：断言视频超限时，不仅 path_task，所有关联的 cover 后台任务也必须被取消并等待退出"""
        task_dir = self.plugin.temp_dir / f"task_dur_cover_{uuid.uuid4().hex}"
        task_dir.mkdir(parents=True, exist_ok=True)
        req_ctx = RequestContext(task_dir=task_dir)

        cover_cancelled = asyncio.Event()

        async def cover_download():
            try:
                while True:
                    await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                cover_cancelled.set()
                raise

        t_cover = asyncio.create_task(cover_download())

        class MockVC:
            duration = 300.0  # 超限
            path_task = Path("/tmp/dummy.mp4")
            cover = t_cover

        class MockDownloader:
            def __init__(self, cfg):
                self.cfg = cfg

        class MockParser:
            def __init__(self, cfg, dl):
                pass
            async def parse(self, kw, match):
                res = MagicMock()
                res.title = "超长带封面视频"
                res.text = ""
                res.author = None
                res.extra = None
                res.cover_path = None
                res.video_contents = [MockVC()]
                res.img_contents = []
                return res

        class MockPlugin:
            def __init__(self):
                self.cfg = MagicMock()
                self.downloader = MockDownloader(self.cfg)
                self.parser_map = {"cover_v": MockParser(self.cfg, self.downloader)}
                self.key_pattern_list = [("cover_v", lambda u: True)]

        self.context.stars["astrbot_plugin_parser"] = MagicMock(star_cls=MockPlugin())

        res = await self.plugin.parser_bridge.parse_url_with_existing_parser(
            url="http://domain.com/cover_v/test",
            max_media_mb=50,
            max_video_duration=180,
            task_dir=task_dir,
            req_ctx=req_ctx,
        )

        self.assertIsNotNone(res)
        self.assertIn("超过系统上限 180秒", res.get("video_error", ""))
        self.assertTrue(t_cover.done(), "封面任务必须已被取消并等待退出")
        self.assertTrue(cover_cancelled.is_set())
        await req_ctx.close_normal()

    def test_v4_multiline_header_followed_by_content_on_next_line(self):
        """【审查缺陷 1 回归测试】：断言编号头独立成行（如「笑点1：\\n内容」或「1.\\n内容」）时，内容绝不丢失，不误判为无笑点或空编号"""
        from astrbot_plugin_punchline_analyzer.formatter import format_punchline_response, validate_punchline_format

        # Case 1: 笑点1：换行后接内容
        raw1 = (
            "笑点解析：\n"
            "笑点1：\n"
            "程序员写代码，造成系统崩溃。\n"
            "笑点2：\n"
            "测试经理当场破防。\n"
            "令人忍俊不禁"
        )
        res1 = format_punchline_response(raw1)
        ok1, reason1 = validate_punchline_format(res1)
        self.assertTrue(ok1, f"笑点1：换行格式校验失败: {reason1}\n{res1}")
        self.assertIn("笑点1：程序员写代码，造成系统崩溃，很好笑。", res1)
        self.assertIn("笑点2：测试经理当场破防，很好笑。", res1)
        self.assertNotIn("笑点1：很好笑。", res1, "绝不产生空笑点编号项")

        # Case 2: 1. 换行后接内容及多行补充
        raw2 = (
            "1.\n"
            "主治医生忘记拔出听诊器\n"
            "病人家属误以为在做法事\n"
            "2.\n"
            "护士长默默递上说明书\n"
        )
        res2 = format_punchline_response(raw2)
        ok2, reason2 = validate_punchline_format(res2)
        self.assertTrue(ok2, f"1. 换行格式校验失败: {reason2}\n{res2}")
        self.assertIn("笑点1：主治医生忘记拔出听诊器 病人家属误以为在做法事，很好笑。", res2)
        self.assertIn("笑点2：护士长默默递上说明书，很好笑。", res2)

        # Case 3: 无冒号的 笑点1 独立换行
        raw3 = (
            "笑点1\n"
            "猫猫试图抓影子结果撞墙\n"
            "笑点2\n"
            "狗狗在旁边一脸鄙视\n"
        )
        res3 = format_punchline_response(raw3)
        ok3, reason3 = validate_punchline_format(res3)
        self.assertTrue(ok3, f"笑点1无冒号换行格式校验失败: {reason3}\n{res3}")
        self.assertIn("笑点1：猫猫试图抓影子结果撞墙，很好笑。", res3)
        self.assertIn("笑点2：狗狗在旁边一脸鄙视，很好笑。", res3)

    def test_v4_closing_bracket_and_alphabet_numbering(self):
        """【审查缺陷 2 回归测试】：断言半角/全角右括号编号（1) / 1））与英文字母编号（A. / (a)）被彻底剥离且主语数量词不被吞噬"""
        from astrbot_plugin_punchline_analyzer.formatter import format_punchline_response, validate_punchline_format

        # Case 1: 半角与全角右括号
        raw1 = (
            "1) 300个程序员同时提交代码\n"
            "2） 一天吃了三碗面\n"
        )
        res1 = format_punchline_response(raw1)
        ok1, reason1 = validate_punchline_format(res1)
        self.assertTrue(ok1, f"右括号编号格式校验失败: {reason1}\n{res1}")
        self.assertIn("笑点1：300个程序员同时提交代码，很好笑。", res1)
        self.assertIn("笑点2：一天吃了三碗面，很好笑。", res1)
        self.assertNotIn("1)", res1)
        self.assertNotIn("2）", res1)

        # Case 2: 英文字母编号 A. 与 B.
        raw2 = (
            "A. 主角出场极其酷炫\n"
            "B. 结果踩到香蕉皮摔倒\n"
        )
        res2 = format_punchline_response(raw2)
        ok2, reason2 = validate_punchline_format(res2)
        self.assertTrue(ok2, f"字母编号 A. 校验失败: {reason2}\n{res2}")
        self.assertIn("笑点1：主角出场极其酷炫，很好笑。", res2)
        self.assertIn("笑点2：结果踩到香蕉皮摔倒，很好笑。", res2)
        self.assertNotIn("A.", res2)
        self.assertNotIn("B.", res2)

        # Case 3: 括号字母 (a) 与 (b)
        raw3 = (
            "(a) 兔子跳进草丛不见了\n"
            "(b) 乌龟还在起跑线吃草\n"
        )
        res3 = format_punchline_response(raw3)
        ok3, reason3 = validate_punchline_format(res3)
        self.assertTrue(ok3, f"括号字母 (a) 校验失败: {reason3}\n{res3}")
        self.assertIn("笑点1：兔子跳进草丛不见了，很好笑。", res3)
        self.assertIn("笑点2：乌龟还在起跑线吃草，很好笑。", res3)
        self.assertNotIn("(a)", res3)
        self.assertNotIn("(b)", res3)

    def test_v4_question_mark_and_colon_no_comma_glitch(self):
        """【审查缺陷 3 回归测试】：断言问号（？/?）与冒号（：/:）收尾的笑点规范化时绝对不产生「？，」或「：，」标点连用"""
        from astrbot_plugin_punchline_analyzer.formatter import format_punchline_response, validate_punchline_format

        raw = (
            "笑点1：这个程序员居然直接在生产库测试？很好笑。\n"
            "笑点2：老板当场质问：很好笑。\n"
        )
        res = format_punchline_response(raw)
        ok, reason = validate_punchline_format(res)
        self.assertTrue(ok, f"标点连用校验失败: {reason}\n{res}")
        self.assertNotIn("？，", res, "绝不产生 '？，'")
        self.assertNotIn("?，", res, "绝不产生 '?，'")
        self.assertNotIn("：，", res, "绝不产生 '：，'")
        self.assertNotIn(":，", res, "绝不产生 ':，'")
        self.assertIn("笑点1：这个程序员居然直接在生产库测试，很好笑。", res)
        self.assertIn("笑点2：老板当场质问，很好笑。", res)

    def test_v4_multiline_failure_explanation_natural_joining(self):
        """【审查缺陷 4 回归测试】：断言多行失败说明在中文语境下自然拼接，不插入多余西文空格且严格合规"""
        from astrbot_plugin_punchline_analyzer.formatter import format_punchline_response, validate_punchline_format

        raw = (
            "未能读取视频画面。\n"
            "现有信息不足以判断笑点，请补充视频内容或相关台词。\n"
        )
        res = format_punchline_response(raw)
        ok, reason = validate_punchline_format(res)
        self.assertTrue(ok, f"失败多行拼接校验失败: {reason}\n{res}")
        self.assertNotIn("很好笑", res)
        self.assertNotIn("。 现有", res, "中文句号与下文之间不应插入英文空格")
        self.assertIn("未能读取视频画面。现有信息不足以判断笑点，请补充视频内容或相关台词。", res)

    def test_v4_markdown_links_and_contrast_idiom_handling(self):
        """【审查缺陷 5 回归测试】：断言 Markdown 超链接语法 [文本](url) 被无损剥离，且日常成语（如平淡无奇）不被误判为失败说明"""
        from astrbot_plugin_punchline_analyzer.formatter import format_punchline_response, validate_punchline_format

        # Markdown 链接剥离
        raw1 = "笑点1：在[B站视频](https://bilibili.com/video/BV123)中主角被门夹了"
        res1 = format_punchline_response(raw1)
        ok1, reason1 = validate_punchline_format(res1)
        self.assertTrue(ok1, f"Markdown 链接校验失败: {reason1}\n{res1}")
        self.assertIn("笑点1：在B站视频中主角被门夹了，很好笑。", res1)
        self.assertNotIn("https://", res1)

        # 包含日常叙述成语“平淡无奇”的正常笑点（未带反差/反转词汇）不被误杀
        raw2 = "笑点1：主人公故意装作平淡无奇，结果下一秒就被戳穿了西洋镜，笑料百出。"
        res2 = format_punchline_response(raw2)
        ok2, reason2 = validate_punchline_format(res2)
        self.assertTrue(ok2, f"平淡无奇成语误杀校验失败: {reason2}\n{res2}")
        self.assertIn("笑点1：主人公故意装作平淡无奇，结果下一秒就被戳穿了西洋镜，笑料百出，很好笑。", res2)

    async def test_v5_slash_command_group_whitelist_enforcement(self):
        """【复审缺陷 1 回归测试】：斜杠指令与自动触发入口统一群白名单权限校验"""
        plugin = PunchlineAnalyzerPlugin(MockContext(), {"allowed_groups": ["ALLOWED_GRP_100"]})
        seen_handled = []

        async def mock_process_flow(event, user_text, is_history_mode=False, history_count=30):
            seen_handled.append(event.get_group_id())
            yield "HANDLED_RESPONSE"

        plugin._process_flow = mock_process_flow

        # 1. 未授权群：发起 /笑点解析 指令与 @机器人 消息，均必须被拦截
        denied_event = MockAstrMessageEvent(
            messages=[MockPlain("/笑点解析 最近30条")],
            group_id="DENIED_GRP_200",
            sender_id="user_test",
        )
        cmd_denied_results = [r async for r in plugin.cmd_punchline(denied_event, "最近30条")]
        self.assertEqual(cmd_denied_results, [], "被排除的群调用 /笑点解析 必须直接拦截且不返回内容")
        self.assertTrue(denied_event.stopped, "被排除的群指令事件也应调用 stop_event 阻止默认聊天回复")
        self.assertEqual(seen_handled, [], "被排除的群绝不能进入 _process_flow 执行链路")

        denied_mention_event = MockAstrMessageEvent(
            messages=[MockAt("99999"), MockPlain("笑点解析")],
            group_id="DENIED_GRP_200",
            self_id="99999",
            sender_id="user_test",
        )
        auto_denied_results = [r async for r in plugin.on_message(denied_mention_event)]
        self.assertEqual(auto_denied_results, [], "被排除的群 @机器人 也必须直接拦截")
        self.assertEqual(seen_handled, [], "被排除的群无论哪个入口均不能进入执行链路")

        # 2. 授权群：发起指令与自动触发，均正常放行并处理
        allowed_event = MockAstrMessageEvent(
            messages=[MockPlain("/笑点解析 最近30条")],
            group_id="ALLOWED_GRP_100",
            sender_id="user_test",
        )
        cmd_allowed_results = [r async for r in plugin.cmd_punchline(allowed_event, "最近30条")]
        self.assertEqual(cmd_allowed_results, ["HANDLED_RESPONSE"], "授权白名单群正常执行指令")
        self.assertIn("ALLOWED_GRP_100", seen_handled)

        allowed_mention_event = MockAstrMessageEvent(
            messages=[MockAt("99999"), MockPlain("笑点解析")],
            group_id="ALLOWED_GRP_100",
            self_id="99999",
            sender_id="user_test",
        )
        auto_allowed_results = [r async for r in plugin.on_message(allowed_mention_event)]
        self.assertEqual(auto_allowed_results, ["HANDLED_RESPONSE"], "授权白名单群正常响应自动触发")

    async def test_v5_parser_construction_failure_cleans_downloader_session(self):
        """【复审缺陷 2 回归测试】：平台解析器构造失败时，立即释放并纳管下载器网络会话，杜绝泄漏"""
        import types
        import re
        import aiohttp
        from astrbot_plugin_punchline_analyzer.parser_bridge import (
            create_isolated_parser_context,
            RequestContext,
        )

        made_downloaders = []

        class MockDownloaderWithSession:
            def __init__(self, cfg):
                self.cfg = cfg
                self.client = aiohttp.ClientSession()
                made_downloaders.append(self)

            async def close(self):
                await self.client.close()

        class MockBrokenParser:
            def __init__(self, cfg, dl):
                raise RuntimeError("simulated parser constructor explosion")

        shared_plugin = types.SimpleNamespace(
            cfg=types.SimpleNamespace(source_max_size=90),
            downloader=object.__new__(MockDownloaderWithSession),
            parser_map={"broken_plat": object.__new__(MockBrokenParser)},
            key_pattern_list=[("broken_plat", re.compile("broken_plat"))],
        )

        test_dir = self.plugin.temp_dir / f"test_v5_init_fail_{uuid.uuid4().hex}"
        test_dir.mkdir(parents=True, exist_ok=True)
        req_ctx = RequestContext(test_dir)

        # 1. 验证在传入 req_ctx 且平台解析器抛错时：不产生额外后台关闭任务，统一由 req_ctx 纳管
        before_tasks = set(asyncio.all_tasks())
        res = await create_isolated_parser_context(
            parser_plugin=shared_plugin,
            url="https://broken_plat.com/video/123",
            task_dir=test_dir,
            req_ctx=req_ctx,
        )
        after_tasks = set(asyncio.all_tasks())
        self.assertIsNone(res[0], "构造失败应优雅返回 None 并启用降级")
        self.assertEqual(after_tasks - before_tasks, set(), "有 req_ctx 时严禁创建未纳管的额外后台关闭任务")
        self.assertEqual(len(req_ctx.iso_contexts), 1, "下载器即使在解析器构造失败时也必须已登记到 req_ctx")

        # 执行正常清理流程
        await req_ctx.close_normal()
        self.assertTrue(made_downloaders[0].client.closed, "清理结束后下载器网络会话必须已关闭，杜绝 session 泄漏")

        # 2. 验证在未传入 req_ctx (req_ctx=None) 的场景下，构造失败路径异步等待关闭完成再返回
        made_downloaders.clear()
        res_no_req = await create_isolated_parser_context(
            parser_plugin=shared_plugin,
            url="https://broken_plat.com/video/123",
            req_ctx=None,
        )
        self.assertIsNone(res_no_req[0])
        self.assertEqual(len(made_downloaders), 1)
        # 无需任何额外 sleep 等待，函数返回时 session 必须已经由内部等待关闭完成
        self.assertTrue(made_downloaders[0].client.closed, "无 req_ctx 时也必须等待关闭完成再返回，杜绝未等待后台任务")

    async def test_v6_slow_close_ordering_assertion_task_finishes_before_directory_deletion(self):
        """【终审剩余缺陷回归测试】：慢关闭场景顺序断言，确认关闭任务结束早于任务目录删除，且同一上下文无重复关闭"""
        import types
        import re
        import aiohttp
        from astrbot_plugin_punchline_analyzer.parser_bridge import (
            create_isolated_parser_context,
            RequestContext,
        )

        class MockSlowDownloader:
            def __init__(self, cfg):
                self.cfg = cfg
                self.client = aiohttp.ClientSession()
                self.close_calls = 0
                self.timeline = []

            async def close(self):
                self.close_calls += 1
                # 记录开始关闭时目录是否依然存在
                self.timeline.append(("close_start", task_dir.exists()))
                # 模拟慢关闭 / 网络连接注销延时
                await asyncio.sleep(0.05)
                # 记录完成关闭时目录是否依然存在
                self.timeline.append(("close_end", task_dir.exists()))
                await self.client.close()

        class MockBrokenParser:
            def __init__(self, cfg, dl):
                raise RuntimeError("simulated parser constructor explosion")

        slow_dl_holder = []

        class MockSlowDownloaderWithHolder(MockSlowDownloader):
            def __init__(self, cfg):
                super().__init__(cfg)
                slow_dl_holder.append(self)

        shared_plugin = types.SimpleNamespace(
            cfg=types.SimpleNamespace(source_max_size=90),
            downloader=object.__new__(MockSlowDownloaderWithHolder),
            parser_map={"slow_broken": object.__new__(MockBrokenParser)},
            key_pattern_list=[("slow_broken", re.compile("slow_broken"))],
        )

        task_dir = self.plugin.temp_dir / f"test_v6_slow_close_{uuid.uuid4().hex}"
        task_dir.mkdir(parents=True, exist_ok=True)
        test_file = task_dir / "sample.bin"
        test_file.write_bytes(b"temp_media_content")
        req_ctx = RequestContext(task_dir)

        before_tasks = set(asyncio.all_tasks())
        res = await create_isolated_parser_context(
            parser_plugin=shared_plugin,
            url="https://slow_broken.com/video/999",
            task_dir=task_dir,
            req_ctx=req_ctx,
        )
        after_tasks = set(asyncio.all_tasks())

        self.assertIsNone(res[0], "构造失败应优雅返回 None")
        self.assertEqual(after_tasks - before_tasks, set(), "有 req_ctx 时绝不产生悬挂后台任务")
        self.assertEqual(len(slow_dl_holder), 1)
        dl = slow_dl_holder[0]
        self.assertEqual(dl.close_calls, 0, "此时 close 尚未触发，全权交由 req_ctx 在清理阶段统一调度")

        # 执行请求正常收尾清理流程
        await req_ctx.close_normal()

        # 核心顺序断言：
        # 1. 关闭必须仅调用 1 次，严禁同一上下文重复关闭（杜绝 close_call_count=2）
        self.assertEqual(dl.close_calls, 1, "同一上下文绝不应被重复关闭（close_call_count 必须为 1）")
        # 2. 会话必须已彻底关闭
        self.assertTrue(dl.client.closed, "慢关闭下载器的网络会话必须已关闭")
        # 3. 顺序断言：close_start 和 close_end 发生时，任务目录必须都依然完好存在
        self.assertEqual([item[0] for item in dl.timeline], ["close_start", "close_end"])
        self.assertTrue(dl.timeline[0][1], "close 开始时任务目录必须存在")
        self.assertTrue(dl.timeline[1][1], "close 结束时任务目录必须仍然存在（确认任务结束早于目录删除）")
        # 4. 整个清理流程完全结束后，目录才被安全删除
        self.assertFalse(task_dir.exists(), "全部关闭与等待彻底完成后，任务目录才应被移除")

    async def test_v7_overlapping_close_awaits_in_progress_and_protects_directory(self):
        """【终审缺陷 1 回归测试 - 并发关闭】：后到调用必须等待同一关闭任务完成，绝不提前返回导致目录过早删除"""
        import aiohttp
        from astrbot_plugin_punchline_analyzer.parser_bridge import (
            IsolatedParserContext,
            RequestContext,
        )

        td = self.plugin.temp_dir / f"test_v7_overlap_{uuid.uuid4().hex}"
        td.mkdir(parents=True, exist_ok=True)
        started = asyncio.Event()
        release = asyncio.Event()

        class GateDownloader:
            def __init__(self):
                self.client = aiohttp.ClientSession()
                self.calls = 0

            async def close(self):
                self.calls += 1
                started.set()
                await release.wait()
                await self.client.close()

        dl = GateDownloader()
        iso = IsolatedParserContext(downloader=dl)
        req = RequestContext(td)
        req.track_iso_context(iso)

        first_task = asyncio.create_task(iso.close())
        await started.wait()

        # 并发启动 req.close_normal()
        normal_task = asyncio.create_task(req.close_normal())
        await asyncio.sleep(0.02)

        # 核心断言：此时 first_task 仍在执行，normal_task 必须等待它，任务目录必须依然完好存在！
        self.assertFalse(first_task.done(), "first_task 仍在等待释放")
        self.assertFalse(normal_task.done(), "normal_task 绝不能提前返回，必须等待 first_task 完成")
        self.assertTrue(td.exists(), "并发关闭进行中，任务目录严禁被提前删除！")

        # 释放栅栏
        release.set()
        await asyncio.gather(first_task, normal_task)

        self.assertTrue(first_task.done())
        self.assertTrue(normal_task.done())
        self.assertEqual(dl.calls, 1, "同一上下文绝不重复调用 close")
        self.assertTrue(dl.client.closed, "会话必须已关闭")
        self.assertFalse(td.exists(), "全部关闭完全结束后目录才被删除")

    async def test_v7_cancelled_close_allows_retry_and_does_not_mark_closed(self):
        """【终审缺陷 1 回归测试 - 取消后重试】：关闭被外部取消时严禁标记为 _closed=True，必须支持后续安全重试"""
        import aiohttp
        from astrbot_plugin_punchline_analyzer.parser_bridge import IsolatedParserContext

        class CancelGateDownloader:
            def __init__(self):
                self.client = aiohttp.ClientSession()
                self.calls = 0

            async def close(self):
                self.calls += 1
                await asyncio.sleep(10.0)
                await self.client.close()

        dl = CancelGateDownloader()
        iso = IsolatedParserContext(downloader=dl)

        task = asyncio.create_task(iso.close())
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # 核心断言：被取消后严禁标记 _closed=True，底层客户端未关闭
        self.assertFalse(iso._closed, "关闭被取消时绝对不能标记为成功完成(_closed=True)")
        self.assertFalse(dl.client.closed, "会话此时尚未关闭")
        self.assertEqual(dl.calls, 1)

        # 再次调用 close 必须支持正常重试
        async def fast_close():
            dl.calls += 1
            await dl.client.close()

        dl.close = fast_close
        await iso.close()

        self.assertTrue(iso._closed, "重试成功后应标记 _closed=True")
        self.assertTrue(dl.client.closed, "重试后网络会话已成功关闭")
        self.assertEqual(dl.calls, 2, "重试必须触发实际关闭逻辑")

    async def test_v7_create_isolated_parser_context_loop_bound_future_without_req_ctx(self):
        """【终审缺陷 2 回归测试 - 循环绑定 Future 异步关闭】：无 req_ctx 时直接在资源所属循环中 await 关闭，杜绝跨线程跨循环阻塞"""
        import types
        import re
        import aiohttp
        from astrbot_plugin_punchline_analyzer.parser_bridge import create_isolated_parser_context

        made = []

        class LoopBoundDownloader:
            def __init__(self, cfg):
                self.client = aiohttp.ClientSession()
                self.completed = False
                # 绑定当前事件循环的 Future
                self.owner_future = asyncio.get_running_loop().create_future()
                asyncio.get_running_loop().call_later(0.02, self.owner_future.set_result, None)
                made.append(self)

            async def close(self):
                await self.owner_future
                self.completed = True
                await self.client.close()

        class BrokenParser:
            def __init__(self, cfg, dl):
                raise RuntimeError("simulated parser constructor failure")

        shared_plugin = types.SimpleNamespace(
            cfg=types.SimpleNamespace(source_max_size=90),
            downloader=object.__new__(LoopBoundDownloader),
            parser_map={"loop_plat": object.__new__(BrokenParser)},
            key_pattern_list=[("loop_plat", re.compile("loop_plat"))],
        )

        res = await create_isolated_parser_context(
            parser_plugin=shared_plugin,
            url="https://loop_plat.com/video/777",
            req_ctx=None,
        )

        self.assertIsNone(res[0], "构造失败优雅降级返回 None")
        self.assertEqual(len(made), 1)
        dl = made[0]

        # 核心断言：函数返回时，关闭必须已在所属事件循环中完全完成！
        self.assertTrue(dl.completed, "函数返回时下载器 close 必须已彻底完成")
        self.assertTrue(dl.client.closed, "会话必须已关闭")
        self.assertTrue(dl.owner_future.done(), "所属事件循环的 Future 必须已正常解决")

    async def test_v8_waiter_cancelled_does_not_start_second_close_and_waits_for_completion(self):
        """【终审缺陷回归测试 1】：等待者取消绝不启动第二次关闭，请求清理必须等待原关闭彻底完成后才删除目录并按约定传递取消"""
        import aiohttp
        from astrbot_plugin_punchline_analyzer.parser_bridge import (
            IsolatedParserContext,
            RequestContext,
        )

        td = self.plugin.temp_dir / f"test_v8_waiter_cancel_{uuid.uuid4().hex}"
        td.mkdir(parents=True, exist_ok=True)
        started = asyncio.Event()
        release = asyncio.Event()

        class SlowDownloader:
            def __init__(self):
                self.client = aiohttp.ClientSession()
                self.closing = False
                self.calls = 0

            async def close(self):
                self.calls += 1
                if self.closing:
                    return
                self.closing = True
                started.set()
                await release.wait()
                await self.client.close()

        dl = SlowDownloader()
        iso = IsolatedParserContext(downloader=dl)
        req = RequestContext(td)
        req.track_iso_context(iso)

        # 启动第一个关闭任务并等待其进入 close 等待栅栏
        first_task = asyncio.create_task(iso.close())
        await started.wait()

        # 启动等待者执行 req.close_normal()
        waiter_task = asyncio.create_task(req.close_normal())
        await asyncio.sleep(0.02)

        # 此时 first_task 仍在进行，waiter_task 正在等待
        self.assertFalse(first_task.done())
        self.assertFalse(waiter_task.done())
        self.assertTrue(td.exists())
        self.assertEqual(dl.calls, 1)

        # 取消等待者 waiter_task
        waiter_task.cancel()
        await asyncio.sleep(0.02)

        # 关键断言 1：原关闭尚未释放前，等待者绝不能提前返回，目录绝不能提前删除，绝不能触发第二次关闭！
        self.assertFalse(first_task.done(), "原关闭任务此时仍在运行中")
        self.assertFalse(waiter_task.done(), "等待者即使被取消，也必须等待原关闭完成，绝不提前结束")
        self.assertTrue(td.exists(), "原关闭未结束前，任务专属目录绝不能提前被删除！")
        self.assertEqual(dl.calls, 1, "等待者被取消后绝不能启动第二次关闭(calls 必须保持为 1)")

        # 释放原关闭任务
        release.set()
        await asyncio.gather(first_task, return_exceptions=True)
        await asyncio.gather(waiter_task, return_exceptions=True)

        # 关键断言 2：原关闭完成后，目录被删除，等待者按约定传递 CancelledError 结束
        self.assertTrue(first_task.done(), "原关闭任务已顺利完成")
        self.assertTrue(waiter_task.cancelled(), "等待者最终必须按规范进入 cancelled 状态")
        self.assertFalse(td.exists(), "原关闭真正结束后，任务目录才被安全彻底删除")
        self.assertEqual(dl.calls, 1, "全过程仅调用一次 close")
        self.assertTrue(dl.client.closed, "底层的客户端会话已安全关闭")
        self.assertTrue(iso._closed, "上下文标记为已成功关闭")

    async def test_v8_request_context_cancellation_order_and_propagation(self):
        """【终审缺陷回归测试 2】：请求上下文取消与清理顺序实测，保证后台任务先彻底结束再删除目录并传递取消"""
        import aiohttp
        from astrbot_plugin_punchline_analyzer.parser_bridge import RequestContext

        td = self.plugin.temp_dir / f"test_v8_req_cancel_{uuid.uuid4().hex}"
        td.mkdir(parents=True, exist_ok=True)

        subtask_finished = False
        dir_existed_during_subtask = False

        async def slow_bg_task():
            nonlocal subtask_finished, dir_existed_during_subtask
            try:
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                # 模拟后台任务在收到取消后需要一点时间清理缓冲区与释放文件
                await asyncio.sleep(0.05)
                dir_existed_during_subtask = td.exists()
                subtask_finished = True
                raise

        req = RequestContext(td)
        bg_task = asyncio.create_task(slow_bg_task())
        req.track_task(bg_task)

        # 模拟调用 cancel_and_cleanup 的协程本身被取消
        cleanup_task = asyncio.create_task(req.cancel_and_cleanup())
        await asyncio.sleep(0.01)
        cleanup_task.cancel()

        await asyncio.gather(cleanup_task, return_exceptions=True)

        self.assertTrue(bg_task.done(), "后台任务必须已彻底退出")
        self.assertTrue(bg_task.cancelled(), "后台任务必须已取消")
        self.assertTrue(subtask_finished, "后台任务取消清理代码已完整执行")
        self.assertTrue(dir_existed_during_subtask, "后台任务在执行清理退出时，任务目录依然完好存在")
        self.assertFalse(td.exists(), "所有后台任务退出后，任务目录才被最终移除")
        self.assertTrue(cleanup_task.cancelled(), "cleanup_task 最终必须按约定传递 CancelledError")
        self.assertTrue(req._closed, "请求上下文标记为已关闭")



    async def test_v9_cancellation_during_parser_close_session_in_close_normal(self):
        """【终审缺陷回归 1】：close_normal 执行期间在解析器 close_session 内部遭遇取消，关闭操作必须抵御取消完整执行并删除目录、传递取消"""
        import aiohttp
        from astrbot_plugin_punchline_analyzer.parser_bridge import (
            IsolatedParserContext,
            RequestContext,
        )

        td = self.plugin.temp_dir / f"test_v9_owner_normal_{uuid.uuid4().hex}"
        td.mkdir(parents=True, exist_ok=True)
        entered = asyncio.Event()

        class DL:
            def __init__(self):
                self.client = aiohttp.ClientSession()
            async def close(self):
                await self.client.close()

        class Parser:
            def __init__(self):
                self.session = aiohttp.ClientSession()
                self.completed = False
            async def close_session(self):
                entered.set()
                await asyncio.sleep(0.05)
                await self.session.close()
                self.completed = True

        dl = DL()
        parser = Parser()
        iso = IsolatedParserContext(parser=parser, downloader=dl)
        req = RequestContext(td)
        req.track_iso_context(iso)

        cleanup_task = asyncio.create_task(req.close_normal())
        await asyncio.wait_for(entered.wait(), 1.0)
        cleanup_task.cancel()
        await asyncio.wait_for(asyncio.gather(cleanup_task, return_exceptions=True), 2.0)

        self.assertTrue(cleanup_task.cancelled(), "请求清理任务最终必须按规范进入 cancelled 状态")
        self.assertTrue(parser.completed, "解析器 close_session 必须抵御调用者取消并完整完成")
        self.assertTrue(parser.session.closed, "解析器持有的底层 aiohttp Session 必须已彻底关闭")
        self.assertTrue(dl.client.closed, "下载器 ClientSession 必须已关闭")
        self.assertTrue(iso._closed, "独立解析器上下文必须标记为已成功关闭")
        self.assertTrue(req._closed, "请求上下文必须确认完成后标记关闭")
        self.assertFalse(td.exists(), "全部确认关闭完成后，任务目录已被安全彻底删除")
        self.assertEqual(len(req.iso_contexts), 0, "确认关闭后已从请求上下文登记列表中移除")

    async def test_v9_cancellation_during_parser_close_session_in_cancel_and_cleanup(self):
        """【终审缺陷回归 2】：cancel_and_cleanup 执行期间在解析器 close_session 内部遭遇取消，关闭操作必须抵御取消完整执行并删除目录、传递取消"""
        import aiohttp
        from astrbot_plugin_punchline_analyzer.parser_bridge import (
            IsolatedParserContext,
            RequestContext,
        )

        td = self.plugin.temp_dir / f"test_v9_owner_cancel_{uuid.uuid4().hex}"
        td.mkdir(parents=True, exist_ok=True)
        entered = asyncio.Event()

        class DL:
            def __init__(self):
                self.client = aiohttp.ClientSession()
            async def close(self):
                await self.client.close()

        class Parser:
            def __init__(self):
                self.session = aiohttp.ClientSession()
                self.completed = False
            async def close_session(self):
                entered.set()
                await asyncio.sleep(0.05)
                await self.session.close()
                self.completed = True

        dl = DL()
        parser = Parser()
        iso = IsolatedParserContext(parser=parser, downloader=dl)
        req = RequestContext(td)
        req.track_iso_context(iso)

        cleanup_task = asyncio.create_task(req.cancel_and_cleanup())
        await asyncio.wait_for(entered.wait(), 1.0)
        cleanup_task.cancel()
        await asyncio.wait_for(asyncio.gather(cleanup_task, return_exceptions=True), 2.0)

        self.assertTrue(cleanup_task.cancelled(), "请求清理任务最终必须按规范进入 cancelled 状态")
        self.assertTrue(parser.completed, "解析器 close_session 必须抵御调用者取消并完整完成")
        self.assertTrue(parser.session.closed, "解析器持有的底层 aiohttp Session 必须已彻底关闭")
        self.assertTrue(dl.client.closed, "下载器 ClientSession 必须已关闭")
        self.assertTrue(iso._closed, "独立解析器上下文必须标记为已成功关闭")
        self.assertTrue(req._closed, "请求上下文必须确认完成后标记关闭")
        self.assertFalse(td.exists(), "全部确认关闭完成后，任务目录已被安全彻底删除")
        self.assertEqual(len(req.iso_contexts), 0, "确认关闭后已从请求上下文登记列表中移除")


    async def test_v10_config_missing_illegal_and_hard_cap_clamping(self):
        """测试新增面板配置：缺失默认值、非法输入容错、旧配置超限硬上限钳制及允许调低"""
        # 1. 缺失配置 -> 默认值
        c_default = PluginConfig({})
        self.assertTrue(c_default.enable_auto_trigger, "默认开启自动触发")
        self.assertEqual(c_default.user_cooldown_seconds, 15.0, "用户冷却默认 15 秒")
        self.assertEqual(c_default.group_cooldown_seconds, 5.0, "群冷却默认 5 秒")
        self.assertEqual(c_default.max_punchlines, 5, "最多笑点数量默认 5")
        self.assertEqual(c_default.analysis_detail, "standard", "解析详细程度默认 standard")
        self.assertEqual(c_default.max_media_mb, 50, "体积硬上限默认 50MB")
        self.assertEqual(c_default.max_video_duration, 180, "时长硬上限默认 180秒")
        self.assertEqual(c_default.video_sample_frames, 4, "抽帧硬上限默认 4帧")

        # 2. 非法输入容错
        c_illegal = PluginConfig({
            "user_cooldown_seconds": "invalid_val",
            "group_cooldown_seconds": -5.0,
            "max_punchlines": 999,
            "analysis_detail": "unknown_option",
        })
        self.assertEqual(c_illegal.user_cooldown_seconds, 15.0, "非数值回退默认值")
        self.assertEqual(c_illegal.group_cooldown_seconds, 0.0, "负数归零关闭")
        self.assertEqual(c_illegal.max_punchlines, 10, "超过 10 钳制为上限 10")
        self.assertEqual(c_illegal.analysis_detail, "standard", "未知枚举回退 standard")

        # 中文别名映射测试
        c_cn = PluginConfig({"analysis_detail": "简短", "max_punchlines": 0})
        self.assertEqual(c_cn.analysis_detail, "brief", "中文别名映射为 brief")
        self.assertEqual(c_cn.max_punchlines, 1, "小于 1 钳制为下限 1")

        c_cn2 = PluginConfig({"analysis_detail": "详细"})
        self.assertEqual(c_cn2.analysis_detail, "detailed", "中文别名映射为 detailed")

        # 3. 旧配置超限硬上限钳制（严禁突破硬上限）
        c_hard = PluginConfig({
            "max_media_mb": 200,
            "max_video_duration": 600,
            "video_sample_frames": 10,
        })
        self.assertEqual(c_hard.max_media_mb, 50, "体积超过 50MB 自动钳制为 50")
        self.assertEqual(c_hard.max_video_duration, 180, "时长超过 180s 自动钳制为 180")
        self.assertEqual(c_hard.video_sample_frames, 4, "抽帧超过 4 帧自动钳制为 4")

        # 4. 允许用户调低
        c_lower = PluginConfig({
            "max_media_mb": 20,
            "max_video_duration": 60,
            "video_sample_frames": 2,
        })
        self.assertEqual(c_lower.max_media_mb, 20, "允许用户调低体积")
        self.assertEqual(c_lower.max_video_duration, 60, "允许用户调低时长")
        self.assertEqual(c_lower.video_sample_frames, 2, "允许用户调低抽帧数")

    async def test_v10_auto_trigger_disabled_slash_command_still_works(self):
        """测试关闭自动入口后：@机器人 不触发但记录历史，独立指令 /笑点解析 正常工作"""
        plugin = PunchlineAnalyzerPlugin(self.context, config={"enable_auto_trigger": False})
        plugin.context = self.context

        # 1. 模拟群聊中 @机器人 + 笑点解析
        ev_auto = MockAstrMessageEvent(
            messages=[MockAt("99999", "Bot"), MockPlain("笑点解析 这个好好笑")],
            sender_id="user_123",
            group_id="group_456",
        )
        results = [res async for res in plugin.on_message(ev_auto)]
        self.assertEqual(len(results), 0, "关闭自动触发后，on_message 不触发且无任何输出")
        self.assertFalse(ev_auto.stopped, "关闭自动触发时不截断事件")

        # 验证必要的群聊历史记录仍正常记录
        cached_msgs = list(plugin.history_manager._cache.get("group_456", []))
        self.assertEqual(len(cached_msgs), 1, "历史记录继续正常缓存工作")
        self.assertIn("笑点解析", cached_msgs[0].text)

        # 2. 模拟使用独立指令 /笑点解析 最近30条
        ev_cmd = MockAstrMessageEvent(
            messages=[MockPlain("/笑点解析 最近30条")],
            sender_id="user_123",
            group_id="group_456",
        )
        cmd_results = [res async for res in plugin.cmd_punchline(ev_cmd, arg="最近30条")]
        self.assertEqual(len(cmd_results), 1, "独立指令入口仍然可用并输出结果")
        self.assertIn("笑点解析：", cmd_results[0])
        self.assertIn("令人忍俊不禁", cmd_results[0])
        self.assertTrue(ev_cmd.stopped, "独立指令入口截断默认事件")

    async def test_v10_cooldown_isolation_and_concurrency(self):
        """测试用户与群冷却：平台/机器人/群/用户四维隔离、私聊隔离与并发一致性"""
        from astrbot_plugin_punchline_analyzer.main import CooldownManager
        cm = CooldownManager()

        # 1. 用户冷却：User A 在 Group 1 发起请求（冷却 10 秒，群冷却 0）
        ok, rem, old_cd = await cm.acquire_atomic("plat1", "bot1", "grp1", "userA", 10.0, 0.0)
        self.assertTrue(ok)

        # User A 在 Group 1 立即再次发起 -> 拒绝
        ok2, rem2, _ = await cm.acquire_atomic("plat1", "bot1", "grp1", "userA", 10.0, 0.0)
        self.assertFalse(ok2)
        self.assertGreater(rem2, 8.0)

        # User A 在 Group 2 发起 -> 允许（群隔离！）
        ok_grp2, _, _ = await cm.acquire_atomic("plat1", "bot1", "grp2", "userA", 10.0, 0.0)
        self.assertTrue(ok_grp2, "用户冷却按群隔离，不同群互不干扰")

        # User B 在 Group 1 发起 -> 允许（用户隔离！）
        ok_userB, _, _ = await cm.acquire_atomic("plat1", "bot1", "grp1", "userB", 10.0, 0.0)
        self.assertTrue(ok_userB, "用户冷却按用户隔离，同一群不同用户互不干扰")

        # 2. 群冷却：Group 1 开启 10 秒群冷却（用户冷却 0）
        cm_grp = CooldownManager()
        ok_g1, _, _ = await cm_grp.acquire_atomic("plat1", "bot1", "grp1", "userA", 0.0, 10.0)
        self.assertTrue(ok_g1)

        # User B 在 Group 1 立即发起 -> 拒绝（群正在冷却）
        ok_g2, rem_g, _ = await cm_grp.acquire_atomic("plat1", "bot1", "grp1", "userB", 0.0, 10.0)
        self.assertFalse(ok_g2, "群冷却生效，拦截该群所有用户")
        self.assertGreater(rem_g, 8.0)

        # User B 在 Group 2 发起 -> 允许（群隔离，不串群）
        ok_g3, _, _ = await cm_grp.acquire_atomic("plat1", "bot1", "grp2", "userB", 0.0, 10.0)
        self.assertTrue(ok_g3, "群冷却不串群，不同群互不干扰")

        # 3. 私聊隔离：私聊中 group_id 为空
        cm_private = CooldownManager()
        ok_p1, _, _ = await cm_private.acquire_atomic("plat1", "bot1", "", "userA", 10.0, 5.0)
        self.assertTrue(ok_p1)
        # 私聊不参与群冷却，但受私聊用户冷却约束
        ok_p2, _, _ = await cm_private.acquire_atomic("plat1", "bot1", "", "userA", 10.0, 5.0)
        self.assertFalse(ok_p2, "私聊用户冷却正常生效")
        # 同一用户在群里不受私聊冷却限制
        ok_p3, _, _ = await cm_private.acquire_atomic("plat1", "bot1", "grp1", "userA", 10.0, 5.0)
        self.assertTrue(ok_p3, "私聊与群聊互相隔离")

        # 4. 并发一致性：10 个并发请求同时发起，必须且仅能接纳 1 个请求
        cm_conc = CooldownManager()
        async def call_atomic():
            return await cm_conc.acquire_atomic("plat1", "bot1", "grp1", "userA", 10.0, 10.0)

        tasks = [asyncio.create_task(call_atomic()) for _ in range(10)]
        results = await asyncio.gather(*tasks)
        successes = [r for r in results if r[0] is True]
        failures = [r for r in results if r[0] is False]
        self.assertEqual(len(successes), 1, "高并发竞争下，必须原子接纳恰好 1 个请求")
        self.assertEqual(len(failures), 9, "其余 9 个并发请求全部被冷却一致性拦截")

    async def test_v10_cooldown_shared_across_entrances_and_chat_intercepted(self):
        """测试自动入口与斜杠入口共用冷却，冷却拒绝时提示剩余时间并截断聊天流水线"""
        plugin = PunchlineAnalyzerPlugin(
            self.context,
            config={
                "enable_auto_trigger": True,
                "user_cooldown_seconds": 15.0,
                "group_cooldown_seconds": 5.0,
            },
        )
        plugin.context = self.context

        # 1. 用户首先通过自动入口调用
        ev1 = MockAstrMessageEvent(
            messages=[MockAt("99999", "Bot"), MockPlain("笑点解析 这个好好笑"), MockImage("https://example.com/joke.jpg")],
            sender_id="user_cd_1",
            group_id="group_cd_1",
        )
        res1 = [res async for res in plugin.on_message(ev1)]
        self.assertEqual(len(res1), 1)
        self.assertIn("笑点解析：", res1[0])

        # 2. 同一用户立即切换到斜杠指令入口 /笑点解析（共用冷却，防绕过）
        ev2 = MockAstrMessageEvent(
            messages=[MockPlain("/笑点解析")],
            sender_id="user_cd_1",
            group_id="group_cd_1",
        )
        res2 = [res async for res in plugin.cmd_punchline(ev2, arg="")]
        self.assertEqual(len(res2), 1, "冷却拒绝时输出提示")
        self.assertIn("笑点解析冷却中，请等待", res2[0])
        self.assertIn("秒后再试~", res2[0])
        self.assertTrue(ev2.stopped, "冷却拒绝时必须截断事件，避免触发默认聊天")

        # 3. 未被接纳的请求（如非白名单群）不消耗冷却
        plugin_wl = PunchlineAnalyzerPlugin(
            self.context,
            config={"allowed_groups": ["group_allowed"], "user_cooldown_seconds": 15.0},
        )
        plugin_wl.context = self.context
        ev_rejected = MockAstrMessageEvent(
            messages=[MockPlain("/笑点解析")],
            sender_id="user_test",
            group_id="group_not_allowed",
        )
        res_rej = [res async for res in plugin_wl.cmd_punchline(ev_rejected, arg="")]
        self.assertEqual(len(res_rej), 0, "非白名单群直接忽略")
        # 冷却未被消耗：转到白名单群调用应当立即成功接纳
        ev_ok = MockAstrMessageEvent(
            messages=[MockPlain("/笑点解析"), MockImage("https://example.com/joke.png")],
            sender_id="user_test",
            group_id="group_allowed",
        )
        res_ok = [res async for res in plugin_wl.cmd_punchline(ev_ok, arg="")]
        self.assertEqual(len(res_ok), 1)
        self.assertIn("笑点解析：", res_ok[0], "未接纳请求不消耗冷却，白名单调用立即可用")

    async def test_v10_max_punchlines_and_analysis_detail_in_prompt_and_formatting(self):
        """测试最多笑点数量与详细程度：Prompt 约束传导、格式化严格截断与经典格式不变"""
        # 1. Prompt 提示词传导测试
        p_brief = build_user_prompt(
            target_description="群聊文字",
            text_content="测试素材",
            max_punchlines=3,
            analysis_detail="brief",
        )
        self.assertIn("最多笑点数量上限：3 项", p_brief)
        self.assertIn("简短（brief）", p_brief)

        p_detailed = build_user_prompt(
            target_description="群聊文字",
            text_content="测试素材",
            max_punchlines=7,
            analysis_detail="detailed",
        )
        self.assertIn("最多笑点数量上限：7 项", p_detailed)
        self.assertIn("详细（detailed）", p_detailed)

        # 2. 格式化器数量上限严格截断测试
        raw_llm = """笑点解析：
笑点1：第一条，很好笑。
笑点2：第二条，很好笑。
笑点3：第三条，很好笑。
笑点4：第四条，很好笑。
笑点5：第五条，很好笑。
笑点6：第六条，很好笑。
令人忍俊不禁"""
        # 限制为 3 个笑点
        formatted_3 = format_punchline_response(raw_llm, max_punchlines=3)
        self.assertIn("笑点1：", formatted_3)
        self.assertIn("笑点2：", formatted_3)
        self.assertIn("笑点3：", formatted_3)
        self.assertNotIn("笑点4：", formatted_3)
        self.assertNotIn("笑点5：", formatted_3)
        self.assertNotIn("笑点6：", formatted_3)
        ok, reason = validate_punchline_format(formatted_3)
        self.assertTrue(ok, f"格式合规校验通过: {reason}")

        # 3. 失败/降级格式保持纯客观，不带编号，不追加很好笑
        failure_raw = "素材较为日常平淡，未发现明显的幽默反差或笑点。"
        fmt_failure = format_punchline_response(failure_raw, max_punchlines=5)
        self.assertIn("笑点解析：", fmt_failure)
        self.assertIn("令人忍俊不禁", fmt_failure)
        self.assertNotIn("笑点1：", fmt_failure)
        self.assertNotIn("很好笑", fmt_failure)
        ok_fail, reason_fail = validate_punchline_format(fmt_failure)
        self.assertTrue(ok_fail, f"失败格式合规校验通过: {reason_fail}")


    async def test_v11_close_normal_failure_retains_resources_and_retries_successfully(self):
        """测试 76：close_normal 遇到底层关闭异常时保留资源登记、保持未完成与保留目录；故障解除后 ensure_cleanup 重试成功释放"""
        import aiohttp
        from astrbot_plugin_punchline_analyzer.parser_bridge import IsolatedParserContext, RequestContext

        td = self.plugin.temp_dir / f"test_v11_close_normal_{uuid.uuid4().hex}"
        td.mkdir(parents=True, exist_ok=True)

        class FailSession:
            def __init__(self):
                self.real = aiohttp.ClientSession()
                self.fail = True
                self.calls = 0

            @property
            def closed(self):
                return self.real.closed

            async def close(self):
                self.calls += 1
                if self.fail:
                    raise OSError("simulated close failure")
                await self.real.close()

        class Parser:
            def __init__(self):
                self.session = FailSession()

            async def close_session(self):
                await self.session.close()

        parser = Parser()
        iso = IsolatedParserContext(parser=parser)
        req = RequestContext(td)
        req.track_iso_context(iso)

        # 1. 执行 close_normal，此时 fail=True
        await req.close_normal()

        # 断言故障时状态：实际 session 未关闭，iso 与 req 未标记关闭，资源登记保留，任务目录保留
        self.assertFalse(parser.session.closed, "故障时底层 session 绝不能误标为 closed")
        self.assertFalse(iso._closed, "关闭失败时 iso._closed 绝不能误标为 True")
        self.assertFalse(req._closed, "关闭失败时 req._closed 绝不能误标为 True")
        self.assertEqual(len(req.iso_contexts), 1, "未关闭成功的上下文必须保留登记以备重试")
        self.assertTrue(td.exists(), "未确认完全释放前任务目录绝不能删除")

        # 2. 模拟故障解除，调用 ensure_cleanup 重试释放
        parser.session.fail = False
        await req.ensure_cleanup()

        # 断言重试后状态：底层 session 真正关闭，状态标记完成，资源注销，任务目录安全删除
        self.assertTrue(parser.session.closed, "故障解除重试后底层 session 必须彻底关闭")
        self.assertTrue(iso._closed, "重试成功后 iso 必须标记为已关闭")
        self.assertTrue(req._closed, "重试成功后 req 必须标记为已关闭")
        self.assertEqual(len(req.iso_contexts), 0, "重试成功后登记上下文必须注销清空")
        self.assertFalse(td.exists(), "重试成功后任务目录必须彻底删除")

    async def test_v11_cancel_and_cleanup_failure_retains_resources_and_retries_successfully(self):
        """测试 77：cancel_and_cleanup 遇到底层关闭异常时保留资源登记、保持未完成与保留目录；故障解除后 ensure_cleanup 重试成功释放"""
        import aiohttp
        from astrbot_plugin_punchline_analyzer.parser_bridge import IsolatedParserContext, RequestContext

        td = self.plugin.temp_dir / f"test_v11_cancel_cleanup_{uuid.uuid4().hex}"
        td.mkdir(parents=True, exist_ok=True)

        class FailSession:
            def __init__(self):
                self.real = aiohttp.ClientSession()
                self.fail = True
                self.calls = 0

            @property
            def closed(self):
                return self.real.closed

            async def close(self):
                self.calls += 1
                if self.fail:
                    raise OSError("simulated close failure")
                await self.real.close()

        class Parser:
            def __init__(self):
                self.session = FailSession()

            async def close_session(self):
                await self.session.close()

        parser = Parser()
        iso = IsolatedParserContext(parser=parser)
        req = RequestContext(td)
        req.track_iso_context(iso)

        # 1. 执行 cancel_and_cleanup，此时 fail=True
        await req.cancel_and_cleanup()

        # 断言故障时状态：实际 session 未关闭，iso 与 req 未标记关闭，资源登记保留，任务目录保留
        self.assertFalse(parser.session.closed, "故障时底层 session 绝不能误标为 closed")
        self.assertFalse(iso._closed, "关闭失败时 iso._closed 绝不能误标为 True")
        self.assertFalse(req._closed, "关闭失败时 req._closed 绝不能误标为 True")
        self.assertEqual(len(req.iso_contexts), 1, "未关闭成功的上下文必须保留登记以备重试")
        self.assertTrue(td.exists(), "未确认完全释放前任务目录绝不能删除")

        # 2. 模拟故障解除，调用 ensure_cleanup 重试释放
        parser.session.fail = False
        await req.ensure_cleanup()

        # 断言重试后状态：底层 session 真正关闭，状态标记完成，资源注销，任务目录安全删除
        self.assertTrue(parser.session.closed, "故障解除重试后底层 session 必须彻底关闭")
        self.assertTrue(iso._closed, "重试成功后 iso 必须标记为已关闭")
        self.assertTrue(req._closed, "重试成功后 req 必须标记为已关闭")
        self.assertEqual(len(req.iso_contexts), 0, "重试成功后登记上下文必须注销清空")
        self.assertFalse(td.exists(), "重试成功后任务目录必须彻底删除")

    async def test_v12_cooldown_revert_ownership_and_versioning(self):
        """测试 78：CooldownManager.revert() 校验请求归属标识，禁止撤销/覆盖后来请求的冷却记录；覆盖群冷却与同用户更新双场景"""
        from astrbot_plugin_punchline_analyzer.main import CooldownManager
        cm = CooldownManager()

        # 场景 1：A 预接纳；群冷却到期后 B 预接纳；A 随后排队超时回滚；C 仍应受 B 的剩余群冷却限制
        ok_a, rem_a, ticket_a = await cm.acquire_atomic(
            platform_id="plat_v12",
            bot_id="bot_v12",
            group_id="grp_v12_1",
            user_id="user_A",
            user_cooldown_sec=15.0,
            group_cooldown_sec=5.0,
            request_id="req_A",
        )
        self.assertTrue(ok_a, "请求 A 必须成功预接纳")
        self.assertEqual(ticket_a.request_id, "req_A")

        # 模拟 5.1 秒流逝，群冷却到期，用户 A 冷却未到期
        cm._group_records[("plat_v12", "bot_v12", "grp_v12_1")].timestamp -= 5.1

        # 用户 B 在同群发起请求 B 预接纳
        ok_b, rem_b, ticket_b = await cm.acquire_atomic(
            platform_id="plat_v12",
            bot_id="bot_v12",
            group_id="grp_v12_1",
            user_id="user_B",
            user_cooldown_sec=15.0,
            group_cooldown_sec=5.0,
            request_id="req_B",
        )
        self.assertTrue(ok_b, "群冷却到期后请求 B 必须成功预接纳")
        self.assertEqual(ticket_b.request_id, "req_B")

        # 请求 A 随后排队超时调用 revert() 回滚
        await cm.revert(ticket=ticket_a)

        # 断言 1：用户 A 自身未被后来请求覆盖，其个人冷却应被正常撤销
        ok_uA, rem_uA = await cm.check("plat_v12", "bot_v12", "grp_v12_1", "user_A", 15.0, 0.0)
        self.assertTrue(ok_uA, "用户 A 个人冷却应随请求 A 回滚而安全解除")
        self.assertEqual(rem_uA, 0.0)

        # 断言 2：用户 C 在该群发起请求，必须仍受请求 B 的剩余群冷却限制，绝不能被请求 A 误删
        ok_c, rem_c, _ = await cm.acquire_atomic(
            platform_id="plat_v12",
            bot_id="bot_v12",
            group_id="grp_v12_1",
            user_id="user_C",
            user_cooldown_sec=15.0,
            group_cooldown_sec=5.0,
            request_id="req_C",
        )
        self.assertFalse(ok_c, "请求 C 必须仍受请求 B 建立的群冷却拦截")
        self.assertGreater(rem_c, 3.0, f"请求 C 应具有大于 3 秒的剩余群冷却，实测 {rem_c}")

        # 场景 2：同用户记录更新后的旧请求回滚，严禁覆盖新请求的冷却
        cm2 = CooldownManager()
        ok_1, _, ticket_1 = await cm2.acquire_atomic(
            platform_id="plat_v12",
            bot_id="bot_v12",
            group_id="grp_v12_2",
            user_id="user_Same",
            user_cooldown_sec=10.0,
            group_cooldown_sec=0.0,
            request_id="req_Same_1",
        )
        self.assertTrue(ok_1)

        # 模拟用户冷却到期（10.1 秒后）
        cm2._user_records[("plat_v12", "bot_v12", "grp_v12_2", "user_Same")].timestamp -= 10.1

        # 同一用户发起请求 2 预接纳
        ok_2, _, ticket_2 = await cm2.acquire_atomic(
            platform_id="plat_v12",
            bot_id="bot_v12",
            group_id="grp_v12_2",
            user_id="user_Same",
            user_cooldown_sec=10.0,
            group_cooldown_sec=0.0,
            request_id="req_Same_2",
        )
        self.assertTrue(ok_2)

        # 旧请求 1 发生超时回滚
        await cm2.revert(ticket=ticket_1)

        # 断言 3：该用户必须仍处于请求 2 建立的冷却保护中，绝不能被旧请求 1 撤销
        ok_same, rem_same = await cm2.check(
            platform_id="plat_v12",
            bot_id="bot_v12",
            group_id="grp_v12_2",
            user_id="user_Same",
            user_cooldown_sec=10.0,
            group_cooldown_sec=0.0,
        )
        self.assertFalse(ok_same, "同用户新请求的冷却绝不能被旧请求的回滚覆盖或清空")
        self.assertGreater(rem_same, 8.0, f"新请求剩余冷却应大于 8 秒，实测 {rem_same}")

    async def test_v12_queue_lock_cancellation_revokes_preemption_and_preserves_later_requests(self):
        """测试 79：排队等待全局执行锁期间取消，安全撤销冷却预占、等待计数恢复、调用数为0；已执行请求保留冷却语义，且取消回滚不覆盖后来请求"""
        from astrbot_plugin_punchline_analyzer.main import _get_platform_id, _get_bot_id

        plugin = PunchlineAnalyzerPlugin(
            self.context,
            config={
                "enable_auto_trigger": True,
                "user_cooldown_seconds": 15.0,
                "group_cooldown_seconds": 5.0,
                "queue_wait_seconds": 6.0,
            },
        )
        plugin.context = self.context

        extractor_call_count = 0
        real_extract = plugin.extractor.extract_target

        async def tracked_extract(*args, **kwargs):
            nonlocal extractor_call_count
            extractor_call_count += 1
            return await real_extract(*args, **kwargs)

        plugin.extractor.extract_target = tracked_extract

        # 1. 占住执行锁
        await plugin._lock.acquire()
        self.assertTrue(plugin._lock.locked(), "测试主动占住执行锁")

        # 2. 发起请求 A 进入排队
        ev_a = MockAstrMessageEvent(
            messages=[MockAt("99999", "Bot"), MockPlain("笑点解析 这个好好笑"), MockImage("https://example.com/joke_a.jpg")],
            sender_id="user_queue_a",
            group_id="group_queue_1",
        )
        plat_a = _get_platform_id(ev_a)
        bot_a = _get_bot_id(ev_a)

        async def run_flow(ev):
            results = []
            async for res in plugin._process_flow(ev, user_text="笑点解析 这个好好笑"):
                results.append(res)
            return results

        task_a = asyncio.create_task(run_flow(ev_a))
        await asyncio.sleep(0.05)

        # 断言排队状态：等待计数为 1，用户 A 与该群已在冷却管理器中预占
        self.assertEqual(plugin._waiting_count, 1, "请求 A 必须已进入排队且等待计数为 1")
        ok_before_cancel, rem_before = await plugin.cooldown_manager.check(
            platform_id=plat_a,
            bot_id=bot_a,
            group_id="group_queue_1",
            user_id="user_queue_a",
            user_cooldown_sec=15.0,
            group_cooldown_sec=5.0,
        )
        self.assertFalse(ok_before_cancel, "排队中请求 A 的冷却已原子预占")

        # 模拟在请求 A 阻塞排队期间，群冷却到期后请求 B 预接纳
        plugin.cooldown_manager._group_records[(plat_a, bot_a, "group_queue_1")].timestamp -= 5.1
        ok_b, _, ticket_b = await plugin.cooldown_manager.acquire_atomic(
            platform_id=plat_a,
            bot_id=bot_a,
            group_id="group_queue_1",
            user_id="user_queue_b",
            user_cooldown_sec=15.0,
            group_cooldown_sec=5.0,
            request_id="req_later_b",
        )
        self.assertTrue(ok_b, "请求 B 必须成功预接纳并建立最新群冷却")

        # 3. 在排队等待执行锁期间取消请求 A
        task_a.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task_a

        # 断言取消后状态：
        # - 提取与模型调用为 0
        self.assertEqual(extractor_call_count, 0, "排队期间取消，提取/模型调用必须恰好为 0")
        # - 等待计数恢复为 0
        self.assertEqual(plugin._waiting_count, 0, "取消后排队等待计数必须正确恢复为 0")
        # - 执行锁状态保持正确（仍由测试持有）
        self.assertTrue(plugin._lock.locked(), "执行锁仍由测试安全持有，未发生锁泄漏或提前释放")
        # - 用户 A 的个人冷却已安全撤销，无残留冷却
        ok_user_a, rem_user_a = await plugin.cooldown_manager.check(
            platform_id=plat_a,
            bot_id=bot_a,
            group_id="group_queue_1",
            user_id="user_queue_a",
            user_cooldown_sec=15.0,
            group_cooldown_sec=0.0,
        )
        self.assertTrue(ok_user_a, f"请求 A 取消后用户个人冷却已安全撤销，无残留，实测 rem={rem_user_a}")
        self.assertEqual(rem_user_a, 0.0)

        # - 验证取消回滚绝不会覆盖后来请求 B 的群冷却记录！
        ok_grp_check, rem_grp_check = await plugin.cooldown_manager.check(
            platform_id=plat_a,
            bot_id=bot_a,
            group_id="group_queue_1",
            user_id="user_queue_c",
            user_cooldown_sec=15.0,
            group_cooldown_sec=5.0,
        )
        self.assertFalse(ok_grp_check, "请求 A 的取消回滚绝不能覆盖后来请求 B 建立的群冷却")
        self.assertGreater(rem_grp_check, 3.0, f"后来请求 B 的剩余群冷却应大于 3 秒，实测 {rem_grp_check}")

        # 4. 释放测试持有的执行锁
        plugin._lock.release()
        self.assertFalse(plugin._lock.locked(), "测试主动释放执行锁")

        # 5. 验证已取得执行锁并开始执行的请求，取消时保留现有冷却语义
        async def slow_extract(*args, **kwargs):
            nonlocal extractor_call_count
            extractor_call_count += 1
            await asyncio.sleep(0.2)
            return await real_extract(*args, **kwargs)

        plugin.extractor.extract_target = slow_extract

        ev_exec = MockAstrMessageEvent(
            messages=[MockAt("99999", "Bot"), MockPlain("笑点解析 执行中取消"), MockImage("https://example.com/joke_exec.jpg")],
            sender_id="user_exec",
            group_id="group_queue_2",
        )
        plat_exec = _get_platform_id(ev_exec)
        bot_exec = _get_bot_id(ev_exec)

        task_exec = asyncio.create_task(run_flow(ev_exec))
        await asyncio.sleep(0.05)

        # 确认请求已取得锁并开始执行（extractor 已被调用）
        self.assertEqual(extractor_call_count, 1, "请求必须已取得锁并开始执行提取")
        self.assertEqual(plugin._waiting_count, 0, "已开始执行的请求不在等待队列中")
        self.assertTrue(plugin._lock.locked(), "执行阶段锁处于持有状态")

        # 取消执行中的任务
        task_exec.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task_exec

        # 断言已执行任务取消后：锁已释放，但保留消耗冷却语义（防重试刷爆接口）
        self.assertFalse(plugin._lock.locked(), "执行取消后锁必须彻底释放")
        ok_exec_cd, rem_exec_cd = await plugin.cooldown_manager.check(
            platform_id=plat_exec,
            bot_id=bot_exec,
            group_id="group_queue_2",
            user_id="user_exec",
            user_cooldown_sec=15.0,
            group_cooldown_sec=5.0,
        )
        self.assertFalse(ok_exec_cd, "已开始执行的请求取消时，必须保留冷却语义")
        self.assertGreater(rem_exec_cd, 10.0, f"剩余冷却应大于 10 秒，实测 {rem_exec_cd}")



    async def test_v6_group_filter_modes_and_dynamic_provider_selection(self):
        """【第80项测试】：群聊三模过滤（白名单/黑名单/全部群）与模型提供商动态选择"""
        # ==========================================
        # 1. 测试群过滤三模式与向后兼容
        # ==========================================
        # 1.1 白名单模式 (whitelist)
        cfg_white = PluginConfig({"group_filter_mode": "whitelist", "filter_groups": ["111", "222"]})
        self.assertEqual(cfg_white.group_filter_mode, "whitelist")
        self.assertTrue(cfg_white.is_group_allowed("111"))
        self.assertTrue(cfg_white.is_group_allowed("222"))
        self.assertFalse(cfg_white.is_group_allowed("333"))
        self.assertTrue(cfg_white.is_group_allowed(None), "私聊必须放行")

        # 1.2 白名单模式名单为空时默认放行
        cfg_white_empty = PluginConfig({"group_filter_mode": "whitelist", "filter_groups": []})
        self.assertTrue(cfg_white_empty.is_group_allowed("111"), "白名单为空时应放行保障开箱即用")

        # 1.3 黑名单模式 (blacklist)
        cfg_black = PluginConfig({"group_filter_mode": "blacklist", "filter_groups": ["333", "444"]})
        self.assertEqual(cfg_black.group_filter_mode, "blacklist")
        self.assertTrue(cfg_black.is_group_allowed("111"))
        self.assertTrue(cfg_black.is_group_allowed("222"))
        self.assertFalse(cfg_black.is_group_allowed("333"), "黑名单中的群必须拦截")
        self.assertFalse(cfg_black.is_group_allowed("444"), "黑名单中的群必须拦截")
        self.assertTrue(cfg_black.is_group_allowed(None), "私聊必须放行")

        # 1.4 黑名单模式名单为空时默认放行
        cfg_black_empty = PluginConfig({"group_filter_mode": "blacklist", "filter_groups": []})
        self.assertTrue(cfg_black_empty.is_group_allowed("333"))

        # 1.5 全部群模式 (all)
        cfg_all = PluginConfig({"group_filter_mode": "all", "filter_groups": ["333"]})
        self.assertEqual(cfg_all.group_filter_mode, "all")
        self.assertTrue(cfg_all.is_group_allowed("111"))
        self.assertTrue(cfg_all.is_group_allowed("333"), "全部群模式下即使在名单中也放行")
        self.assertTrue(cfg_all.is_group_allowed(None))

        # 1.6 中文模式别名与旧版配置兼容
        cfg_cn_black = PluginConfig({"group_filter_mode": "黑名单", "allowed_groups": ["888"]})
        self.assertEqual(cfg_cn_black.group_filter_mode, "blacklist")
        self.assertFalse(cfg_cn_black.is_group_allowed("888"))
        self.assertTrue(cfg_cn_black.is_group_allowed("999"))

        cfg_cn_all = PluginConfig({"group_filter_mode": "全部群", "allowed_groups": ["888"]})
        self.assertEqual(cfg_cn_all.group_filter_mode, "all")
        self.assertTrue(cfg_cn_all.is_group_allowed("888"))

        # 1.7 纯旧版 allowed_groups 字段且未指定 group_filter_mode 时，默认作为白名单
        cfg_legacy = PluginConfig({"allowed_groups": ["777"]})
        self.assertEqual(cfg_legacy.group_filter_mode, "whitelist")
        self.assertTrue(cfg_legacy.is_group_allowed("777"))
        self.assertFalse(cfg_legacy.is_group_allowed("888"))
        self.assertEqual(cfg_legacy.allowed_groups, {"777"})
        self.assertEqual(cfg_legacy.filter_groups, {"777"})

        # ==========================================
        # 2. 测试模型提供商动态选择与回退
        # ==========================================
        mock_ctx = MockContext()
        # mock_ctx 已提供:
        # providers[0]: deepseek/deepseek-flash (非视觉)
        # providers[1]: Hagimi3.8/gemini-3.8-flash (多模态视觉)

        # 2.1 自动模式 (auto / 空字符串)
        plugin_auto = PunchlineAnalyzerPlugin(mock_ctx, {"model_provider_id": ""})
        dummy_event = MockAstrMessageEvent(messages=[MockPlain("test")], group_id="111")
        
        # 纯文本请求 -> 使用当前会话模型 deepseek
        pid, imgs, note = await plugin_auto._select_model_and_prepare_vision(dummy_event, [])
        self.assertEqual(pid, "deepseek/deepseek-flash")
        self.assertEqual(imgs, [])
        self.assertEqual(note, "")

        # 附带图片请求 -> 自动探测并切换到视觉模型 gemini
        pid_img, imgs_img, note_img = await plugin_auto._select_model_and_prepare_vision(dummy_event, ["data:image/jpeg;base64,mock"])
        self.assertEqual(pid_img, "Hagimi3.8/gemini-3.8-flash", "自动模式应自动升级到可用视觉提供商")
        self.assertEqual(len(imgs_img), 1)
        self.assertEqual(note_img, "")

        # 2.2 显式指定视觉模型
        plugin_gemini = PunchlineAnalyzerPlugin(mock_ctx, {"model_provider_id": "Hagimi3.8/gemini-3.8-flash"})
        pid_g, imgs_g, note_g = await plugin_gemini._select_model_and_prepare_vision(dummy_event, ["data:image/jpeg;base64,mock"])
        self.assertEqual(pid_g, "Hagimi3.8/gemini-3.8-flash")
        self.assertEqual(len(imgs_g), 1)

        # 2.3 显式指定非视觉模型 + 传入图片 -> 明确降级为纯文本，未传入画面
        plugin_ds = PunchlineAnalyzerPlugin(mock_ctx, {"model_provider_id": "deepseek/deepseek-flash"})
        pid_ds, imgs_ds, note_ds = await plugin_ds._select_model_and_prepare_vision(dummy_event, ["data:image/jpeg;base64,mock"])
        self.assertEqual(pid_ds, "deepseek/deepseek-flash")
        self.assertEqual(imgs_ds, [], "非视觉模型必须被清空传入图片以防报错")
        self.assertIn("降级为纯文本分析", note_ds)

        # 2.4 指定不存在的提供商 -> 记录警告并优雅回退到自动选择模式，不抛出异常
        plugin_nonexist = PunchlineAnalyzerPlugin(mock_ctx, {"model_provider_id": "not_exist/provider-999"})
        pid_ne, imgs_ne, note_ne = await plugin_nonexist._select_model_and_prepare_vision(dummy_event, ["data:image/jpeg;base64,mock"])
        self.assertEqual(pid_ne, "Hagimi3.8/gemini-3.8-flash", "不存在的提供商应安全回退至视觉模型")
        self.assertEqual(len(imgs_ne), 1)

        # 2.5 优先检查 modalities 是否包含 "image"（涵盖 deepseek-flash 默认带 image 模态的场景）
        # 2.5.1 provider_config 中带有 image 模态 -> 即便模型名不包含 gemini/vision 关键词，也必须识别为视觉提供商
        prov_ds_modal = MockProvider(
            "deepseek/deepseek-flash",
            "deepseek-flash",
            provider_config={"modalities": ["text", "image", "tool_use"]}
        )
        self.assertTrue(
            plugin_auto._is_visual_provider(prov_ds_modal),
            "modalities 包含 image 时必须优先判定为支持视觉多模态"
        )

        # 2.5.2 显式声明 modalities 但不含 image -> 判定为非视觉
        prov_no_img = MockProvider(
            "custom/text-model",
            "text-model",
            provider_config={"modalities": ["text", "tool_use"]}
        )
        self.assertFalse(
            plugin_auto._is_visual_provider(prov_no_img),
            "modalities 不含 image 时判定为非视觉"
        )

        # 2.5.3 modalities 支持大写 "IMAGE" 或直接属性/元数据形式
        prov_caps = MockProvider("custom/caps", "caps-model", provider_config={"modalities": ["IMAGE"]})
        self.assertTrue(plugin_auto._is_visual_provider(prov_caps))



    async def test_v7_footer_with_user_id_attribution(self):
        """【第81项测试】：输出末尾「令人忍俊不禁。[群内昵称]问的」归属标注、句号分隔与向下兼容"""
        from astrbot_plugin_punchline_analyzer.formatter import (
            format_punchline_response,
            validate_punchline_format,
        )

        sample_llm = (
            "笑点解析：\n"
            "笑点1：群友信誓旦旦说不熬夜结果凌晨三点还在发猫猫表情包，言行反差极大，很好笑。\n"
            "令人忍俊不禁"
        )

        # 1. 正常笑点解析输出：带群内昵称与句号分隔
        res_user_str = format_punchline_response(sample_llm, user_id="小明")
        self.assertTrue(res_user_str.endswith("\n令人忍俊不禁。[小明]问的"))
        ok_u1, reason_u1 = validate_punchline_format(res_user_str)
        self.assertTrue(ok_u1, f"带群内昵称尾行校验失败: {reason_u1}")

        # 2. 昵称为数字或ID时兼容
        res_user_int = format_punchline_response(sample_llm, user_id=987654321)
        self.assertTrue(res_user_int.endswith("\n令人忍俊不禁。[987654321]问的"))
        ok_u2, reason_u2 = validate_punchline_format(res_user_int)
        self.assertTrue(ok_u2, f"数字昵称尾行校验失败: {reason_u2}")

        # 3. 昵称自带方括号防冗余包裹
        res_bracket = format_punchline_response(sample_llm, user_id="[群管张三]")
        self.assertTrue(res_bracket.endswith("\n令人忍俊不禁。[群管张三]问的"))
        self.assertNotIn("[[", res_bracket)

        # 4. 未传入昵称或为空字符串时，保持经典尾行
        res_empty = format_punchline_response(sample_llm, user_id="")
        self.assertTrue(res_empty.endswith("\n令人忍俊不禁"))
        ok_empty, reason_empty = validate_punchline_format(res_empty)
        self.assertTrue(ok_empty, f"无昵称尾行校验失败: {reason_empty}")

        # 5. 失败/降级分支同样支持昵称与句号分隔标注
        fail_raw = "现有信息不足以判断笑点。"
        res_fail = format_punchline_response(fail_raw, user_id="群管理大大")
        self.assertTrue(res_fail.endswith("\n令人忍俊不禁。[群管理大大]问的"))
        ok_fail, reason_fail = validate_punchline_format(res_fail)
        self.assertTrue(ok_fail, f"降级分支尾行校验失败: {reason_fail}")

        # 6. 端到端流程：验证插件从事件中提取群内昵称 sender_name 并附加到尾行
        ev_test = MockAstrMessageEvent(
            messages=[MockPlain("/笑点解析 最近30条")],
            sender_id="special_user_333",
            sender_name="特别搞笑的群友",
            group_id="88888",
        )
        responses = [r async for r in self.plugin.cmd_punchline(ev_test, arg="最近30条")]
        self.assertTrue(len(responses) > 0)
        self.assertTrue(responses[0].endswith("令人忍俊不禁。[特别搞笑的群友]问的"))

    # 82. 免斜杠自动触发开关专项测试
    async def test_v10_no_slash_trigger_setting_switch(self):
        """【第82项测试】：免斜杠指令（如“笑点解析 最近30条”）开关控制测试"""
        # 1. 默认设置开启 (enable_auto_trigger=True)：无斜杠“笑点解析 最近30条”直接触发
        plugin_on = PunchlineAnalyzerPlugin(self.context, config={"enable_auto_trigger": True})
        plugin_on.context = self.context

        # 预设历史消息以供分析
        for i in range(2):
            ev_h = MockAstrMessageEvent(
                messages=[MockPlain(f"群友发言测试第 {i + 1} 条：今天真开心")],
                sender_id=f"u_{i}",
                sender_name=f"群友_{i}",
                group_id="group_switch_test",
                message_id=f"msg_h_{i}",
            )
            plugin_on.history_manager.record_event(ev_h)

        ev_no_slash = MockAstrMessageEvent(
            messages=[MockPlain("笑点解析 最近30条")],
            sender_id="user_test_on",
            sender_name="提问群友",
            group_id="group_switch_test",
        )
        results_on = [res async for res in plugin_on.on_message(ev_no_slash)]
        self.assertTrue(ev_no_slash.stopped, "开启免斜杠触发时，on_message 应拦截事件并触发")
        self.assertEqual(len(results_on), 1, "开启免斜杠触发时，应成功输出解析结果")
        self.assertIn("笑点解析：", results_on[0])
        self.assertIn("令人忍俊不禁。[提问群友]问的", results_on[0])

        # 2. 设置关闭 (enable_auto_trigger=False)：无斜杠“笑点解析 最近30条”不触发，必须使用斜杠指令 /笑点解析
        plugin_off = PunchlineAnalyzerPlugin(self.context, config={"enable_auto_trigger": False})
        plugin_off.context = self.context

        ev_no_slash_off = MockAstrMessageEvent(
            messages=[MockPlain("笑点解析 最近30条")],
            sender_id="user_test_off",
            sender_name="提问群友",
            group_id="group_switch_test",
        )
        results_off = [res async for res in plugin_off.on_message(ev_no_slash_off)]
        self.assertFalse(ev_no_slash_off.stopped, "关闭免斜杠触发后，非斜杠消息不应被拦截")
        self.assertEqual(len(results_off), 0, "关闭免斜杠触发后，非斜杠消息不应输出解析")

        # 但使用斜杠独立指令仍然正常触发
        ev_slash = MockAstrMessageEvent(
            messages=[MockPlain("/笑点解析 最近30条")],
            sender_id="user_test_off",
            sender_name="提问群友",
            group_id="group_switch_test",
        )
        results_slash = [res async for res in plugin_off.cmd_punchline(ev_slash, arg="最近30条")]
        self.assertTrue(ev_slash.stopped, "关闭免斜杠触发后，斜杠指令仍然正常拦截")
        self.assertEqual(len(results_slash), 1, "关闭免斜杠触发后，斜杠指令应正常输出解析")
        self.assertIn("笑点解析：", results_slash[0])
        self.assertIn("令人忍俊不禁。[提问群友]问的", results_slash[0])



if __name__ == "__main__":
    unittest.main()

