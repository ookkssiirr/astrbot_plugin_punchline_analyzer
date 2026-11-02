from __future__ import annotations

import os
import re
import uuid
from PIL import Image as PILImage
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.core.message.components import At, BaseMessageComponent, Forward, Image, Json, Plain, Reply, Video
from astrbot.core.platform.astr_message_event import AstrMessageEvent

from .config import PluginConfig
from .forward_reader import ForwardMessageReader
from .history_manager import GroupHistoryManager
from .parser_bridge import (
    ParserBridge,
    RequestContext,
    extract_url_from_json_segment,
    extract_urls_from_text,
    extract_video_info,
    is_direct_video_url,
    is_video_segment,
)

DEICTIC_KEYWORDS = ("这个", "刚才", "刚才那个", "上面那个", "前一条", "这图", "这视频", "这卡片")


@dataclass
class TargetResult:
    target_type: str  # "reply" | "forward" | "media" | "image" | "history" | "recent" | "none"
    description: str
    text_content: str
    images: list[str] = field(default_factory=list)
    supplementary_text: str = ""
    degrade_note: str = ""
    temp_files: list[str] = field(default_factory=list)


def clean_user_instruction_text(raw_text: str) -> str:
    """清理指令中的触发词和多余空白，保留用户的实际补充说明"""
    if not raw_text:
        return ""
    # 移除诸如 /笑点解析、笑点解析、帮我对这个做个、来个 等修饰
    text = re.sub(r"/?(?:笑点解析|笑点分析)", "", raw_text)
    text = re.sub(r"^(?:帮我|请|来个|做个|对这个|解析一下|看看|分析一下|分析|看看这个)+", "", text.strip())
    return text.strip()


class TargetExtractor:
    """笑点解析目标提取与多模态素材准备器"""

    def __init__(
        self,
        config: PluginConfig,
        parser_bridge: ParserBridge,
        history_manager: GroupHistoryManager,
        temp_dir: Path,
    ) -> None:
        self.config = config
        self.parser_bridge = parser_bridge
        self.history_manager = history_manager
        self.temp_dir = temp_dir

    async def extract_target(
        self,
        event: AstrMessageEvent,
        user_plain_text: str,
        is_history_mode: bool = False,
        requested_count: int = 30,
        task_dir: Path | None = None,
        req_ctx: RequestContext | None = None,
    ) -> TargetResult:
        """主入口：根据用户当前输入与上下文判定解析目标"""
        active_task_dir = task_dir or self.temp_dir
        active_task_dir.mkdir(parents=True, exist_ok=True)

        target = await self._do_extract_target(
            event=event,
            user_plain_text=user_plain_text,
            is_history_mode=is_history_mode,
            requested_count=requested_count,
            task_dir=active_task_dir,
            req_ctx=req_ctx,
        )

        if target and target.images:
            norm_imgs, new_temps = await self._normalize_candidate_images(
                target.images,
                task_dir=active_task_dir,
            )
            target.images = norm_imgs
            if new_temps:
                target.temp_files.extend(new_temps)

        return target

    async def _normalize_candidate_images(
        self,
        images: list[str],
        task_dir: Path,
        max_edge: int = 4000,
    ) -> tuple[list[str], list[str]]:
        """
        对传入大模型的候选图片做安全规范化与长图压缩：
        若单边尺寸超过 max_edge（默认 4000 像素），自动等比缩放并转存为标准 Baseline RGB JPEG，
        彻底规避上游 LLM 视觉网关因超高分辨率或 Progressive 格式不兼容引发的 400 unsupported image 报错。
        """
        if not images:
            return [], []

        normalized_images: list[str] = []
        new_temp_files: list[str] = []

        for img_ref in images:
            try:
                from astrbot.core.utils.media_utils import MediaResolver
                async with MediaResolver(img_ref, media_type="image").as_path() as resolved_file:
                    local_path = resolved_file.path
                    if not local_path.exists():
                        normalized_images.append(img_ref)
                        continue

                    with PILImage.open(local_path) as im:
                        w, h = im.size
                        is_progressive = bool(im.info.get("progressive"))
                        needs_resize = max(w, h) > max_edge

                        if needs_resize or (is_progressive and max(w, h) > 2000):
                            scale = 1.0
                            if needs_resize:
                                scale = float(max_edge) / float(max(w, h))
                            new_w = max(1, int(round(w * scale)))
                            new_h = max(1, int(round(h * scale)))

                            im_converted = im
                            if im_converted.mode in ("RGBA", "LA") or (
                                im_converted.mode == "P" and "transparency" in getattr(im_converted, "info", {})
                            ):
                                bg = PILImage.new("RGB", (w, h), (255, 255, 255))
                                bg.paste(im_converted, mask=im_converted.split()[-1])
                                im_converted = bg
                            elif im_converted.mode != "RGB":
                                im_converted = im_converted.convert("RGB")

                            if needs_resize:
                                resample_filter = getattr(PILImage, "Resampling", PILImage).LANCZOS
                                im_converted = im_converted.resize((new_w, new_h), resample_filter)

                            out_name = f"normalized_img_{uuid.uuid4().hex[:8]}.jpg"
                            out_path = task_dir / out_name
                            im_converted.save(
                                str(out_path),
                                format="JPEG",
                                quality=85,
                                progressive=False,
                                optimize=True,
                            )
                            logger.info(
                                f"[笑点解析] 检测到长图/高分辨率图片 ({w}x{h}, progressive={is_progressive})，"
                                f"已自动等比缩放至 ({new_w}x{new_h}) 并转存为标准 Baseline JPEG -> {out_path.name}"
                            )
                            normalized_images.append(str(out_path))
                            new_temp_files.append(str(out_path))
                        else:
                            normalized_images.append(img_ref)
            except Exception as e:
                logger.warning(f"[笑点解析] 图片预处理/尺寸规范化跳过或异常 ({img_ref}): {e}")
                normalized_images.append(img_ref)

        return normalized_images, new_temp_files

    async def _do_extract_target(
        self,
        event: AstrMessageEvent,
        user_plain_text: str,
        is_history_mode: bool = False,
        requested_count: int = 30,
        task_dir: Path | None = None,
        req_ctx: RequestContext | None = None,
    ) -> TargetResult:
        """主入口：根据用户当前输入与上下文判定解析目标"""
        chain = event.get_messages() or []
        cleaned_instruction = clean_user_instruction_text(user_plain_text)
        active_task_dir = task_dir or self.temp_dir
        active_task_dir.mkdir(parents=True, exist_ok=True)

        # 模式 1：最近 30 条群聊历史模式
        if is_history_mode:
            history_text, history_images, count = await self.history_manager.get_recent_history(
                event=event,
                count=requested_count,
                max_images=self.config.max_forward_images,
            )
            return TargetResult(
                target_type="history",
                description=f"本群最近 {count} 条群聊发言",
                text_content=history_text,
                images=history_images,
                supplementary_text=cleaned_instruction,
            )

        # 模式 2：明确引用了某条消息 (Reply)
        reply_seg = next((s for s in chain if isinstance(s, Reply) or type(s).__name__ in ("Reply", "MockReply")), None)
        if reply_seg:
            return await self._handle_reply_target(event, reply_seg, cleaned_instruction, active_task_dir, req_ctx=req_ctx)

        # 模式 3：当前消息附带合并转发组件 (Forward)
        forward_seg = next((s for s in chain if isinstance(s, Forward) or type(s).__name__ in ("Forward", "MockForward")), None)
        if forward_seg:
            forward_reader = ForwardMessageReader(
                call_action=self._get_call_action(event),
                max_depth=self.config.max_forward_depth,
                max_messages=self.config.max_forward_messages,
                max_images=self.config.max_forward_images,
            )
            f_text, f_imgs = await forward_reader.read_forward(forward_seg.id)
            return TargetResult(
                target_type="forward",
                description="当前消息所附带的合并转发聊天记录",
                text_content=f_text,
                images=f_imgs,
                supplementary_text=cleaned_instruction,
            )

        # 模式 4：当前消息直接附带视频 (Video component)
        video_seg = next((s for s in chain if is_video_segment(s)), None)
        if video_seg:
            v_res = await self.parser_bridge.resolve_video_source(
                event=event,
                video_source=video_seg,
                task_dir=active_task_dir,
                max_video_duration=self.config.max_video_duration,
                max_media_mb=self.config.max_media_mb,
                num_frames=self.config.video_sample_frames,
                req_ctx=req_ctx,
            )
            if v_res and v_res.get("has_video"):
                return self._format_video_target(v_res, cleaned_instruction, prefix="用户当前发送的视频")

        # 模式 5：当前消息附带卡片 (Json) 或视频链接
        json_seg = next((s for s in chain if isinstance(s, Json) or type(s).__name__ in ("Json", "MockJson")), None)
        json_url = extract_url_from_json_segment(json_seg.data) if json_seg else None
        text_urls = extract_urls_from_text(user_plain_text)
        media_url = json_url or (text_urls[0] if text_urls else None)

        if media_url:
            # 若属于直接视频直链 (.mp4/.webm 等)，直接走视频源解析与抽帧
            if is_direct_video_url(media_url):
                v_res = await self.parser_bridge.resolve_video_source(
                    event=event,
                    video_ref=media_url,
                    task_dir=active_task_dir,
                    max_video_duration=self.config.max_video_duration,
                    max_media_mb=self.config.max_media_mb,
                    num_frames=self.config.video_sample_frames,
                    req_ctx=req_ctx,
                )
                if v_res and v_res.get("has_video"):
                    return self._format_video_target(v_res, cleaned_instruction, prefix=f"直链视频 ({media_url})")

            parsed_media = await self.parser_bridge.parse_url_with_existing_parser(
                url=media_url,
                max_video_duration=self.config.max_video_duration,
                max_media_mb=self.config.max_media_mb,
                num_frames=self.config.video_sample_frames,
                task_dir=active_task_dir,
                req_ctx=req_ctx,
            )
            if parsed_media:
                return self._format_media_target(parsed_media, cleaned_instruction, media_url)

        # 模式 6：当前消息直接包含图片/表情包 (Image)
        direct_images = self._extract_images_from_chain(chain)
        if direct_images:
            return TargetResult(
                target_type="image",
                description="用户当前发送的图片/表情包",
                text_content="",
                images=direct_images,
                supplementary_text=cleaned_instruction,
            )

        # 模式 7：消解指代（如“刚才那个”、“这个”、或没有任何附件和引用的情况）
        group_id = event.get_group_id()
        self_id = str(event.get_self_id() or "")
        current_msg_id = str(getattr(event.message_obj, "message_id", "") or "")

        candidate = None
        if group_id:
            candidate = self.history_manager.get_most_recent_candidate(
                group_id=group_id,
                self_id=self_id,
                current_msg_id=current_msg_id,
            )

        if candidate:
            target_desc = f"刚才群友 {candidate.nickname} 的发言"
            text_lines = [f"{candidate.nickname}: {candidate.text}"]
            if candidate.reply_quote:
                text_lines.insert(0, candidate.reply_quote)

            # 如果候选消息含视频或链接，尝试做解析
            cand_urls = extract_urls_from_text(candidate.text)
            if cand_urls:
                if is_direct_video_url(cand_urls[0]):
                    cand_v_res = await self.parser_bridge.resolve_video_source(
                        event=event,
                        video_ref=cand_urls[0],
                        task_dir=active_task_dir,
                        max_video_duration=self.config.max_video_duration,
                        max_media_mb=self.config.max_media_mb,
                        num_frames=self.config.video_sample_frames,
                        req_ctx=req_ctx,
                    )
                    if cand_v_res and cand_v_res.get("has_video"):
                        return self._format_video_target(
                            cand_v_res,
                            cleaned_instruction,
                            prefix=f"刚才群友 {candidate.nickname} 分享的视频",
                        )

                cand_media = await self.parser_bridge.parse_url_with_existing_parser(
                    url=cand_urls[0],
                    max_video_duration=self.config.max_video_duration,
                    max_media_mb=self.config.max_media_mb,
                    num_frames=self.config.video_sample_frames,
                    task_dir=active_task_dir,
                    req_ctx=req_ctx,
                )
                if cand_media:
                    return self._format_media_target(
                        cand_media,
                        cleaned_instruction,
                        cand_urls[0],
                        prefix=f"刚才群友 {candidate.nickname} 分享的内容",
                    )

            return TargetResult(
                target_type="recent",
                description=target_desc,
                text_content="\n".join(text_lines),
                images=candidate.images,
                supplementary_text=cleaned_instruction,
            )

        # 模式 8：未能寻找到任何明确目标
        return TargetResult(
            target_type="none",
            description="无有效目标",
            text_content="",
            images=[],
            supplementary_text=cleaned_instruction,
        )

    async def _handle_reply_target(
        self,
        event: AstrMessageEvent,
        reply_seg: Reply,
        supplementary_text: str,
        task_dir: Path,
        req_ctx: RequestContext | None = None,
    ) -> TargetResult:
        """处理通过引用触发的解析目标"""
        quoted_nick = reply_seg.sender_nickname or str(reply_seg.sender_id or "被引用的群友")
        quoted_chain = reply_seg.chain or []
        quoted_str = reply_seg.message_str or ""

        # 1. 被引用消息内含有合并转发
        forward_seg = next((s for s in quoted_chain if isinstance(s, Forward) or type(s).__name__ in ("Forward", "MockForward")), None)
        if forward_seg:
            forward_reader = ForwardMessageReader(
                call_action=self._get_call_action(event),
                max_depth=self.config.max_forward_depth,
                max_messages=self.config.max_forward_messages,
                max_images=self.config.max_forward_images,
            )
            f_text, f_imgs = await forward_reader.read_forward(forward_seg.id)
            return TargetResult(
                target_type="forward",
                description=f"被引用消息中的合并转发记录（由 {quoted_nick} 转发）",
                text_content=f_text,
                images=f_imgs,
                supplementary_text=supplementary_text,
            )

        # 2. 被引用消息直接附带视频组件 (Video component)
        quoted_video = next((s for s in quoted_chain if is_video_segment(s)), None)
        if quoted_video:
            v_res = await self.parser_bridge.resolve_video_source(
                event=event,
                video_source=quoted_video,
                reply_message_id=getattr(reply_seg, "id", ""),
                task_dir=task_dir,
                max_video_duration=self.config.max_video_duration,
                max_media_mb=self.config.max_media_mb,
                num_frames=self.config.video_sample_frames,
                req_ctx=req_ctx,
            )
            if v_res and v_res.get("has_video"):
                return self._format_video_target(
                    v_res,
                    supplementary_text,
                    prefix=f"被引用消息中 {quoted_nick} 发送的视频",
                    quoted_text=quoted_str,
                )

        # 3. 被引用消息内含有视频/分享卡片 (Json) 或视频链接
        json_seg = next((s for s in quoted_chain if isinstance(s, Json) or type(s).__name__ in ("Json", "MockJson")), None)
        json_url = extract_url_from_json_segment(json_seg.data) if json_seg else None
        text_urls = extract_urls_from_text(quoted_str)
        media_url = json_url or (text_urls[0] if text_urls else None)

        if media_url:
            if is_direct_video_url(media_url):
                v_res = await self.parser_bridge.resolve_video_source(
                    event=event,
                    video_ref=media_url,
                    task_dir=task_dir,
                    max_video_duration=self.config.max_video_duration,
                    max_media_mb=self.config.max_media_mb,
                    num_frames=self.config.video_sample_frames,
                    req_ctx=req_ctx,
                )
                if v_res and v_res.get("has_video"):
                    return self._format_video_target(
                        v_res,
                        supplementary_text,
                        prefix=f"被引用消息中 {quoted_nick} 分享的直链视频",
                        quoted_text=quoted_str,
                    )

            parsed_media = await self.parser_bridge.parse_url_with_existing_parser(
                url=media_url,
                max_video_duration=self.config.max_video_duration,
                max_media_mb=self.config.max_media_mb,
                num_frames=self.config.video_sample_frames,
                task_dir=task_dir,
                req_ctx=req_ctx,
            )
            if parsed_media:
                return self._format_media_target(
                    parsed_media,
                    supplementary_text,
                    media_url,
                    prefix=f"被引用消息中 {quoted_nick} 分享的内容",
                )

        # 4. 检查是否引用了短视频（联动专门的引用视频理解插件或 OneBot 历史回溯）
        reply_id = getattr(reply_seg, "id", "")
        if reply_id:
            video_plugin_res = await self.parser_bridge.resolve_video_source(
                event=event,
                reply_message_id=reply_id,
                task_dir=task_dir,
                max_video_duration=self.config.max_video_duration,
                max_media_mb=self.config.max_media_mb,
                num_frames=self.config.video_sample_frames,
                req_ctx=req_ctx,
            )
            if video_plugin_res and video_plugin_res.get("has_video"):
                return self._format_video_target(
                    video_plugin_res,
                    supplementary_text,
                    prefix=f"被引用消息中 {quoted_nick} 分享的视频",
                    quoted_text=quoted_str,
                )

        # 5. 被引用消息含图片/表情包
        quoted_images = self._extract_images_from_chain(quoted_chain)

        # 组装纯文本描述
        content_lines = []
        if quoted_str:
            content_lines.append(f"{quoted_nick}: {quoted_str}")
        elif quoted_images:
            content_lines.append(f"{quoted_nick}: [发送了图片/表情包]")

        return TargetResult(
            target_type="reply",
            description=f"群友 {quoted_nick} 的发言",
            text_content="\n".join(content_lines),
            images=quoted_images,
            supplementary_text=supplementary_text,
        )

    def _format_video_target(
        self,
        video_res: dict[str, Any],
        supplementary_text: str,
        prefix: str,
        quoted_text: str = "",
    ) -> TargetResult:
        cap = video_res.get("caption") or ""
        frames = video_res.get("frames") or []
        error_reason = video_res.get("error_reason") or ""
        cover_path = video_res.get("cover_path")
        has_cover_only = bool(cover_path and not frames)

        lines = [prefix]
        if quoted_text:
            lines.append(f"视频附言: {quoted_text}")
        if cap:
            lines.append(f"【视频理解与转述】：\n{cap}")
        if error_reason:
            lines.append(f"【视频读取失败/限制】：{error_reason}")

        degrade_parts = []
        if error_reason:
            degrade_parts.append(f"【系统失败原因】：{error_reason}")

        if frames:
            degrade_parts.append(f"（已从视频中均匀抽取 {len(frames)} 张代表性画面供视觉分析）")
        elif cap:
            degrade_parts.append("（已结合视频理解插件生成的转述内容进行笑点分析）")
        elif has_cover_only:
            degrade_parts.append(
                "【重要声明】：仅获取到了视频封面图（[图片1]为封面），未能读取或抽取视频内部动态画面。封面图片绝不代表已读取视频画面，严禁根据封面脑补捏造视频具体情节！"
            )
        else:
            degrade_parts.append(
                f"（警告：未能成功获取视频画面，原因：{error_reason or '格式不支持或无法下载'}。封面不代表已读取视频画面，严禁凭空捏造未看见的视频内容）"
            )

        if not frames and not cap:
            lines.append(
                f"【限制提示】：未能成功读取视频画面（{error_reason or '格式不支持或无法下载'}），且封面不代表已读取视频画面。无法在不了解视频内容的情况下编造笑点。请补充视频内容或相关台词。"
            )

        degrade_note = "\n".join(degrade_parts)
        images = frames if frames else ([cover_path] if cover_path else [])

        return TargetResult(
            target_type="media",
            description=prefix,
            text_content="\n".join(lines),
            images=images,
            supplementary_text=supplementary_text,
            degrade_note=degrade_note,
            temp_files=frames,
        )

    def _format_media_target(
        self,
        parsed_media: dict[str, Any],
        supplementary_text: str,
        media_url: str,
        prefix: str = "视频/网页分享内容",
    ) -> TargetResult:
        title = parsed_media.get("title", "")
        text = parsed_media.get("text", "")
        author = parsed_media.get("author", "")
        extra_info = parsed_media.get("extra_info", "")
        images = parsed_media.get("images", [])
        temp_files = parsed_media.get("frame_paths", [])
        video_error = parsed_media.get("video_error", "")
        has_cover_only = bool(parsed_media.get("cover_path") and not temp_files)

        desc_parts = [f"标题: {title}"] if title else []
        if author:
            desc_parts.append(f"作者/UP主: {author}")
        if text:
            desc_parts.append(text)
        if extra_info:
            desc_parts.append(f"AI视频摘要: {extra_info}")
        if video_error:
            desc_parts.append(f"【视频读取失败/限制】：{video_error}")

        degrade_parts = []
        if video_error:
            degrade_parts.append(f"【系统失败原因】：{video_error}")

        if temp_files:
            degrade_parts.append(f"（已成功对视频进行均匀抽帧，共获取 {len(temp_files)} 张关键画面供视觉分析）")
        elif has_cover_only:
            degrade_parts.append(
                "【重要声明】：仅获取到了视频封面图（[图片1]为封面），未能读取或抽取视频内部动态画面。封面图片仅展示视频封面设定，绝不代表已读取视频画面，严禁根据封面脑补捏造视频具体情节！"
            )
        else:
            degrade_parts.append(
                "（未能获取到视频画面抽帧或封面，仅基于标题、简介与视频信息进行分析。封面不代表已读取视频画面，严禁凭空虚构画面细节）"
            )

        degrade_note = "\n".join(degrade_parts)

        return TargetResult(
            target_type="media",
            description=f"{prefix} ({media_url})",
            text_content="\n".join(desc_parts),
            images=images,
            supplementary_text=supplementary_text,
            degrade_note=degrade_note,
            temp_files=temp_files,
        )

    def _extract_images_from_chain(self, chain: list[BaseMessageComponent]) -> list[str]:
        images: list[str] = []
        for seg in chain:
            if isinstance(seg, Image) or type(seg).__name__ in ("Image", "MockImage"):
                ref = getattr(seg, "url", None) or getattr(seg, "file", None) or getattr(seg, "path", None)
                if ref and str(ref) not in images:
                    images.append(str(ref))
        return images

    def _get_call_action(self, event: AstrMessageEvent) -> Any | None:
        try:
            bot = getattr(event, "bot", None)
            if bot and hasattr(bot, "call_action") and callable(bot.call_action):
                return bot.call_action
        except Exception:
            pass
        return None
