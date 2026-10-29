from __future__ import annotations

import datetime
import json
from typing import Any, Callable

from astrbot.api import logger
from astrbot.core.message.components import BaseMessageComponent, Forward, Image, Plain


class ForwardMessageReader:
    """QQ 合并转发记录解析器，支持递归展开嵌套转发与多模态图片提取"""

    def __init__(
        self,
        call_action: Callable[..., Any] | None,
        max_depth: int = 3,
        max_messages: int = 30,
        max_images: int = 10,
    ) -> None:
        self.call_action = call_action
        self.max_depth = max_depth
        self.max_messages = max_messages
        self.max_images = max_images
        self.visited_ids: set[str] = set()
        self.message_count: int = 0
        self.collected_images: list[str] = []

    async def read_forward(self, forward_id: str | int) -> tuple[str, list[str]]:
        """读取指定 forward_id 的转发消息，返回 (格式化文本, 提取的图片列表)"""
        forward_id_str = str(forward_id).strip()
        if not forward_id_str:
            return "（无效的合并转发ID）", []

        nodes = await self._fetch_forward_nodes(forward_id_str)
        if not nodes:
            return "（合并转发记录内容获取失败或为空）", []

        transcript_lines: list[str] = []
        await self._parse_nodes(nodes, transcript_lines, current_depth=1)

        header = f"=== QQ合并转发聊天记录（共解析 {self.message_count} 条消息）==="
        full_transcript = header + "\n" + "\n".join(transcript_lines)
        return full_transcript, self.collected_images

    async def _fetch_forward_nodes(self, forward_id: str) -> list[dict[str, Any]] | None:
        if not self.call_action:
            return None

        # 尝试标准 get_forward_msg
        params_attempts = [
            {"id": forward_id},
            {"message_id": forward_id},
        ]
        if forward_id.isdigit():
            params_attempts.extend([
                {"id": int(forward_id)},
                {"message_id": int(forward_id)},
            ])

        for params in params_attempts:
            try:
                resp = await self.call_action("get_forward_msg", **params)
                nodes = self._extract_nodes_from_resp(resp)
                if nodes:
                    return nodes
            except Exception as e:
                logger.debug(f"[笑点解析] get_forward_msg 尝试 {params} 失败: {e}")

        # 回退尝试 get_msg
        for params in params_attempts:
            try:
                resp = await self.call_action("get_msg", **params)
                if isinstance(resp, dict):
                    data = resp.get("data") if isinstance(resp.get("data"), dict) else resp
                    msg = data.get("message")
                    if isinstance(msg, list):
                        return [{
                            "sender": data.get("sender", {}),
                            "time": data.get("time", 0),
                            "message": msg,
                        }]
            except Exception:
                pass

        return None

    def _extract_nodes_from_resp(self, resp: Any) -> list[dict[str, Any]] | None:
        if isinstance(resp, list) and resp:
            return resp
        if isinstance(resp, dict):
            data = resp.get("data")
            if isinstance(data, list) and data:
                return data
            for target in (data, resp):
                if isinstance(target, dict):
                    for key in ("messages", "message", "nodes", "content"):
                        val = target.get(key)
                        if isinstance(val, list) and val:
                            return val
        return None

    async def _parse_nodes(
        self,
        nodes: list[Any],
        out_lines: list[str],
        current_depth: int,
    ) -> None:
        if current_depth > self.max_depth or self.message_count >= self.max_messages:
            return

        indent = "  " * (current_depth - 1)

        for item in nodes:
            if self.message_count >= self.max_messages:
                out_lines.append(f"{indent}... [超过最大解析条数上限，后续消息已截断]")
                return

            if not isinstance(item, dict):
                continue

            self.message_count += 1

            # 发送人与时间信息
            sender = item.get("sender", {})
            sender_name = ""
            if isinstance(sender, dict):
                sender_name = sender.get("card") or sender.get("nickname") or str(sender.get("user_id", ""))
            if not sender_name:
                sender_name = str(item.get("nickname") or item.get("user_id") or "某群友")

            timestamp = item.get("time", 0)
            time_str = ""
            if timestamp and timestamp > 0:
                try:
                    time_str = datetime.datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
                except Exception:
                    pass

            time_prefix = f"[{time_str}] " if time_str else ""
            line_prefix = f"{indent}{time_prefix}{sender_name}: "

            # 消息内容解析
            msg_content = item.get("message") or item.get("content") or []
            text_parts: list[str] = []

            if isinstance(msg_content, str):
                text_parts.append(msg_content)
            elif isinstance(msg_content, list):
                for seg in msg_content:
                    if isinstance(seg, dict):
                        stype = seg.get("type")
                        sdata = seg.get("data", {})
                        if stype in ("text", "plain"):
                            text_parts.append(str(sdata.get("text", "")))
                        elif stype == "image":
                            img_ref = sdata.get("url") or sdata.get("file")
                            if img_ref and len(self.collected_images) < self.max_images:
                                self.collected_images.append(str(img_ref))
                                img_idx = len(self.collected_images)
                                text_parts.append(f"[图片{img_idx}]")
                            else:
                                text_parts.append("[图片]")
                        elif stype == "forward":
                            nested_id = str(sdata.get("id", "")).strip()
                            if nested_id and nested_id not in self.visited_ids:
                                self.visited_ids.add(nested_id)
                                text_parts.append("\n" + indent + "  【嵌套合并转发】：")
                                nested_nodes = await self._fetch_forward_nodes(nested_id)
                                if nested_nodes:
                                    await self._parse_nodes(nested_nodes, out_lines, current_depth + 1)
                                else:
                                    text_parts.append("（获取嵌套记录失败）")
                        elif stype == "face":
                            text_parts.append(f"[表情:{sdata.get('id', '')}]")
                        elif stype in ("video", "record"):
                            text_parts.append(f"[{stype}]")
                    elif isinstance(seg, str):
                        text_parts.append(seg)

            out_lines.append(line_prefix + "".join(text_parts).strip())
