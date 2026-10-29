from __future__ import annotations

from typing import Any

# 资源保护硬上限常量（允许用户调低，不允许调高突破上限）
HARD_CAP_MAX_MEDIA_MB = 50
HARD_CAP_MAX_VIDEO_DURATION = 180
HARD_CAP_VIDEO_SAMPLE_FRAMES = 4


class PluginConfig:
    """笑点解析插件配置管理"""

    def __init__(self, raw: dict[str, Any] | None = None) -> None:
        raw = raw or {}

        # 1. 免斜杠自动触发开关（打开后无需斜杠即可触发；关闭后必须使用斜杠指令）
        val_auto = raw.get("enable_auto_trigger")
        if val_auto is None:
            val_auto = raw.get("enable_no_slash_trigger")
        self.enable_auto_trigger: bool = bool(val_auto) if val_auto is not None else True

        # 2. 模型选择（自动读取系统中已配置的模型提供商，留空或 auto 为自动选择）
        raw_prov = str(raw.get("model_provider_id", "auto") or "auto").strip()
        self.model_provider_id: str = "" if raw_prov.lower() in ("auto", "none", "默认", "") else raw_prov

        # 3. 群聊过滤模式与群号列表（支持白名单、黑名单、全部群三种模式）
        raw_mode = str(raw.get("group_filter_mode", "whitelist") or "whitelist").strip().lower()
        if raw_mode in ("blacklist", "黑名单", "black", "block"):
            self.group_filter_mode: str = "blacklist"
        elif raw_mode in ("all", "全部", "全部群", "all_groups", "none"):
            self.group_filter_mode: str = "all"
        else:
            self.group_filter_mode: str = "whitelist"

        # 兼容新版 filter_groups 与旧版 allowed_groups
        groups_raw = raw.get("filter_groups")
        if groups_raw is None:
            groups_raw = raw.get("allowed_groups", [])
        if isinstance(groups_raw, list):
            self.filter_groups: set[str] = {str(g).strip() for g in groups_raw if str(g).strip()}
        else:
            self.filter_groups = set()
        # 兼容保留原有属性名
        self.allowed_groups: set[str] = self.filter_groups

        # 4. 两种冷却配置（秒，0 表示关闭）
        try:
            val_u = float(
                raw.get("user_cooldown_seconds", 15.0)
                if raw.get("user_cooldown_seconds") is not None
                else 15.0
            )
            self.user_cooldown_seconds: float = max(0.0, val_u)
        except (ValueError, TypeError):
            self.user_cooldown_seconds = 15.0

        try:
            val_g = float(
                raw.get("group_cooldown_seconds", 5.0)
                if raw.get("group_cooldown_seconds") is not None
                else 5.0
            )
            self.group_cooldown_seconds: float = max(0.0, val_g)
        except (ValueError, TypeError):
            self.group_cooldown_seconds = 5.0

        # 5. 最多笑点数量（范围 1~10，默认 5）
        try:
            val_p = int(
                raw.get("max_punchlines", 5)
                if raw.get("max_punchlines") is not None
                else 5
            )
            self.max_punchlines: int = max(1, min(10, val_p))
        except (ValueError, TypeError):
            self.max_punchlines = 5

        # 6. 解析详细程度（brief / standard / detailed，默认 standard）
        detail_raw = str(raw.get("analysis_detail", "standard") or "standard").strip().lower()
        if detail_raw in ("brief", "简短", "短"):
            self.analysis_detail: str = "brief"
        elif detail_raw in ("detailed", "详细", "详尽"):
            self.analysis_detail: str = "detailed"
        else:
            self.analysis_detail: str = "standard"

        # 7. 历史条数限制（高级设置）
        try:
            self.max_history_count: int = max(2, min(50, int(raw.get("max_history_count", 30) or 30)))
        except (ValueError, TypeError):
            self.max_history_count = 30

        # 8. 合并转发限制（高级设置）
        try:
            self.max_forward_depth: int = max(1, min(5, int(raw.get("max_forward_depth", 3) or 3)))
        except (ValueError, TypeError):
            self.max_forward_depth = 3

        try:
            self.max_forward_messages: int = max(5, min(100, int(raw.get("max_forward_messages", 30) or 30)))
        except (ValueError, TypeError):
            self.max_forward_messages = 30

        try:
            self.max_forward_images: int = max(1, min(15, int(raw.get("max_forward_images", 10) or 10)))
        except (ValueError, TypeError):
            self.max_forward_images = 10

        # 9. 视频与媒体大小限制（硬上限保护：允许调低，超限自动钳制为上限）
        try:
            val = int(raw.get("max_video_duration", 180) or 180)
            self.max_video_duration: int = max(10, min(HARD_CAP_MAX_VIDEO_DURATION, val))
        except (ValueError, TypeError):
            self.max_video_duration = HARD_CAP_MAX_VIDEO_DURATION

        try:
            val = int(raw.get("max_media_mb", 50) or 50)
            self.max_media_mb: int = max(5, min(HARD_CAP_MAX_MEDIA_MB, val))
        except (ValueError, TypeError):
            self.max_media_mb = HARD_CAP_MAX_MEDIA_MB

        try:
            val = int(raw.get("video_sample_frames", 4) or 4)
            self.video_sample_frames: int = max(1, min(HARD_CAP_VIDEO_SAMPLE_FRAMES, val))
        except (ValueError, TypeError):
            self.video_sample_frames = HARD_CAP_VIDEO_SAMPLE_FRAMES

        # 10. 排队与并发限制（高级设置）
        try:
            self.max_wait_queue: int = max(1, min(10, int(raw.get("max_wait_queue", 2) or 2)))
        except (ValueError, TypeError):
            self.max_wait_queue = 2

        try:
            self.queue_wait_seconds: float = max(1.0, min(30.0, float(raw.get("queue_wait_seconds", 6.0) or 6.0)))
        except (ValueError, TypeError):
            self.queue_wait_seconds = 6.0

        # 11. 全过程超时限制（高级设置）
        try:
            self.timeout_seconds: int = max(15, min(120, int(raw.get("timeout_seconds", 60) or 60)))
        except (ValueError, TypeError):
            self.timeout_seconds = 60

    def is_group_allowed(self, group_id: str | int | None) -> bool:
        """根据群聊过滤模式检查群是否被允许使用本插件"""
        if group_id is None:
            return True
        if self.group_filter_mode == "all":
            return True
        if not self.filter_groups:
            # 名单留空时，无论白名单或黑名单均默认允许所有群
            return True
        gid = str(group_id).strip()
        if self.group_filter_mode == "blacklist":
            return gid not in self.filter_groups
        # whitelist 白名单模式
        return gid in self.filter_groups
