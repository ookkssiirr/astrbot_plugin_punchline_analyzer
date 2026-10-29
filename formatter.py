from __future__ import annotations

import re

HEADER = "笑点解析："
FOOTER = "令人忍俊不禁"

# 识别失败说明、无笑点、素材不足等关键词
FAILURE_INDICATORS = (
    "未能读取",
    "无法读取",
    "未读取到",
    "读取失败",
    "未能识别",
    "无法识别",
    "不足以判断",
    "信息不足",
    "缺乏前置背景",
    "背景不足",
    "无明显笑点",
    "未发现笑点",
    "未发现明显的笑点",
    "没有明显的笑点",
    "没有明显笑点",
    "较平淡",
    "较为日常",
    "日常平淡",
    "平淡普通",
    "未发现明显的幽默反差",
    "无法看到视频",
    "请补充视频内容",
    "无法在不了解",
    "无法判断笑点",
)


def sanitize_media_mentions(text: str) -> str:
    """清理技术截图、抽帧等词汇，严格遵循回复中不提及辅助截图的规则"""
    replacements = [
        (r"辅助截图[中里]?", "画面中"),
        (r"辅助截图", "视频画面"),
        (r"抽帧截图[中里]?", "画面中"),
        (r"抽帧截图", "视频画面"),
        (r"抽帧画面[中里]?", "画面中"),
        (r"抽帧画面", "视频画面"),
        (r"画面抽帧", "视频画面"),
        (r"视频抽帧", "视频画面"),
        (r"关键帧截图[中里]?", "画面中"),
        (r"关键帧截图", "视频画面"),
        (r"关键截图[中里]?", "画面中"),
        (r"从截图中可以看出", "从画面中可以看出"),
        (r"从截图可以看出", "从画面可以看出"),
        (r"从截图中", "从画面中"),
        (r"辅助截屏", "视频画面"),
        (r"截帧", "画面"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    return text


def _clean_markdown(text: str) -> str:
    """移除 Markdown 语法元素，保留文本内容"""
    # 移除代码块标记行（保留内部文本），移除行内代码反引号
    text = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"```", "", text)
    text = re.sub(r"`+([^`]+)`+", r"\1", text)
    text = re.sub(r"`", "", text)
    # 移除链接标记 [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # 移除引用块标记 >
    text = re.sub(r"^\s*>+\s*", "", text, flags=re.MULTILINE)
    # 移除删除线
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    # 移除粗体/斜体
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)([^*]+)\*", r"\1", text)
    text = re.sub(r"[*_]", "", text)
    # 移除标题标记
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    # 移除无序列表符号
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    return text


def _is_failure_text(text: str) -> bool:
    """判断是否属于失败说明、信息不足或无笑点等非正常笑点提取文本"""
    # 明确具有幽默反差、反转、包袱阐述的笑点分析正文，不属于失败说明
    if re.search(r"(反差极大|反差强烈|反差拉满|反差感|巨大反差|极具反差|制造反差|出人意料|反转|包袱)", text):
        return False
    for ind in FAILURE_INDICATORS:
        if ind in text:
            return True
    return False


# 编号笑点前缀匹配正则，兼容各类编号、括号、圆圈数字、中文数字、罗马数字、字母及全半角标点
NUMBERED_PREFIX_PATTERN = re.compile(
    r"^(?:"
    r"【\s*笑点\s*(?:\d+|[一二三四五六七八九十]+)?\s*】|"
    r"\[\s*笑点\s*(?:\d+|[一二三四五六七八九十]+)?\s*\]|"
    r"笑点\s*[（(]\s*(?:\d+|[一二三四五六七八九十]+)\s*[)）]\s*[:：、.·\-—\u3002\uff0e]?\s*|"
    r"笑点\s*(?:\d+|[一二三四五六七八九十]+)\s*[:：、.·\-—\u3002\uff0e]?\s*|"
    r"笑点\s*[:：、.·\-—\u3002\uff0e]\s*|"
    r"第\s*(?:\d+|[一二三四五六七八九十]+)\s*(?:个笑点|个|点|条)?\s*[:：、.·\-—\u3002\uff0e]?\s*|"
    r"[（(]\s*(?:\d+|[一二三四五六七八九十]+|[ivxIVX]+|[a-zA-Z])\s*[)）]\s*[:：、.·\-—\u3002\uff0e]?\s*|"
    r"[①②③④⑤⑥⑦⑧⑨⑩ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]\s*[:：、.·\-—\u3002\uff0e]?\s*|"
    r"(?:\d+|[一二三四五六七八九十]+|[ivxIVX]+|[a-zA-Z])\s*[:：、.·\-—\u3002\uff0e)）]\s*"
    r")\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)

# 独立成行的编号头匹配正则（用于跨行编号解析）
STANDALONE_HEADER_PATTERN = re.compile(
    r"^(?:"
    r"【\s*笑点\s*(?:\d+|[一二三四五六七八九十]+)?\s*】|"
    r"\[\s*笑点\s*(?:\d+|[一二三四五六七八九十]+)?\s*\]|"
    r"笑点\s*[（(]\s*(?:\d+|[一二三四五六七八九十]+)\s*[)）]\s*[:：、.·\-—\u3002\uff0e]?|"
    r"笑点\s*(?:\d+|[一二三四五六七八九十]+)\s*[:：、.·\-—\u3002\uff0e]?|"
    r"笑点\s*[:：、.·\-—\u3002\uff0e]|"
    r"第\s*(?:\d+|[一二三四五六七八九十]+)\s*(?:个笑点|个|点|条)?\s*[:：、.·\-—\u3002\uff0e]?|"
    r"[（(]\s*(?:\d+|[一二三四五六七八九十]+|[ivxIVX]+|[a-zA-Z])\s*[)）]\s*[:：、.·\-—\u3002\uff0e]?|"
    r"[①②③④⑤⑥⑦⑧⑨⑩ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]\s*[:：、.·\-—\u3002\uff0e]?|"
    r"(?:\d+|[一二三四五六七八九十]+|[ivxIVX]+|[a-zA-Z])\s*[:：、.·\-—\u3002\uff0e)）]"
    r")\s*$",
    re.IGNORECASE,
)


def format_punchline_response(
    raw_text: str,
    is_degraded_video: bool = False,
    max_punchlines: int = 5,
    user_id: str | int | None = None,
) -> str:
    """
    将分析结果统一规范化为 clean-master/ai-shen-yang-mei-shi-jia 经典纯文本格式。
    """
    raw_text = (raw_text or "").strip()

    # 降级视频但未获取画面时的强防护：严禁捏造笑点
    if is_degraded_video:
        if _is_failure_text(raw_text) and not re.search(r"笑点\s*\d+\s*[:：]", raw_text):
            failure_content = raw_text
        else:
            failure_content = "未能读取视频画面，现有信息不足以判断笑点，请补充视频内容或相关台词。"
        return _format_failure_response(failure_content, user_id=user_id)

    if not raw_text:
        return _format_failure_response("现有信息不足以判断笑点。", user_id=user_id)

    cleaned = _clean_markdown(raw_text)
    cleaned = sanitize_media_mentions(cleaned)

    # 检查是否为失败说明/无笑点说明（若未带任何编号，且整段为失败说明，直接走失败格式）
    has_numbered_points = bool(NUMBERED_PREFIX_PATTERN.search(cleaned))
    if _is_failure_text(cleaned) and not has_numbered_points:
        return _format_failure_response(cleaned, user_id=user_id)

    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]

    # 预过滤首尾固定行与开场/结束客套话
    content_lines: list[str] = []
    for line in lines:
        if re.match(r"^笑点解析\s*[:：]?$", line):
            continue
        if re.match(r"^令人忍俊不禁.*$", line):
            break
        # 过滤开场白客套话与说明性前缀（必须是独立行或紧随冒号/空白等分隔符，防止误伤“注视”、“注定”等正文词汇）
        if re.match(r"^(?:好的|为您解析|以上就是|分析如下|总结如下|解析如下)(?:[，,。！!~：:\s].*)?$", line):
            continue
        if re.match(r"^(?:总结|点评|神评|评分|注|注意|提示)(?:[:：,，\s]+.*|$)", line):
            continue
        content_lines.append(line)

    if not content_lines:
        if _is_failure_text(cleaned):
            return _format_failure_response(cleaned, user_id=user_id)
        return _format_failure_response("现有信息不足以判断笑点。", user_id=user_id)

    # 检查内容行中是否包含明确的编号指示
    has_explicit_numbers = any(
        bool(NUMBERED_PREFIX_PATTERN.match(l) or STANDALONE_HEADER_PATTERN.match(l))
        for l in content_lines
    )

    items: list[str] = []
    if has_explicit_numbers:
        pending_new_item = False
        for line in content_lines:
            # 1. 检查是否为独立成行的编号头（如「笑点1：」「1.」「1)」「【笑点1】」）
            if STANDALONE_HEADER_PATTERN.match(line):
                pending_new_item = True
                continue

            # 2. 检查是否为同行带内容的编号行
            m = NUMBERED_PREFIX_PATTERN.match(line)
            if m:
                content = m.group(1).strip()
                content = re.sub(r"^[：:、.·\-—\u3002\uff0e)）是为\s]+", "", content).strip()
                if content:
                    items.append(content)
                    pending_new_item = False
                else:
                    pending_new_item = True
            elif line.startswith("笑点") and any(c in line for c in ("：", ":", "、", ".", "。", ")", "）")):
                parts = re.split(r"[:：、.·\u3002\uff0e)）]", line, maxsplit=1)
                if len(parts) > 1 and parts[1].strip():
                    items.append(parts[1].strip())
                    pending_new_item = False
                else:
                    pending_new_item = True
            else:
                if pending_new_item:
                    items.append(line)
                    pending_new_item = False
                elif items:
                    items[-1] = items[-1] + " " + line
                # 若尚未出现编号条目（items 为空且无 pending_new_item），则属于编号前的前导废话/开场白，直接过滤
    else:
        # 当模型未输出任何编号格式时，每一个非空独立段落视作独立的笑点条目
        for line in content_lines:
            items.append(line)

    if not items:
        if _is_failure_text(cleaned):
            return _format_failure_response(cleaned, user_id=user_id)
        return _format_failure_response("现有信息不足以判断笑点。", user_id=user_id)

    # 严格检验是否属于无可确认笑点、背景不足或素材读取失败等非正常笑点提取文本
    # 若所有条目均为失败说明，或整篇为失败说明且无任何具有明确反差/幽默阐述的有效笑点，强制走失败格式
    valid_items = [it for it in items if not _is_failure_text(it)]
    if not valid_items or all(_is_failure_text(it) for it in items):
        combined_explanation = "\n".join(items) if items else cleaned
        return _format_failure_response(combined_explanation, user_id=user_id)

    if max_punchlines > 0 and len(valid_items) > max_punchlines:
        valid_items = valid_items[:max_punchlines]

    uid = str(user_id or "").strip()
    if uid.startswith("[") and uid.endswith("]"):
        uid = uid[1:-1].strip()
    footer = f"{FOOTER}。[{uid}]问的" if uid else FOOTER

    normalized_items: list[str] = []
    for idx, item in enumerate(valid_items, 1):
        item_text = _normalize_punchline_item(item)
        normalized_items.append(f"笑点{idx}：{item_text}")

    return f"{HEADER}\n" + "\n".join(normalized_items) + f"\n{footer}"


def _normalize_punchline_item(text: str) -> str:
    """清理并确保笑点项末尾以「很好笑。」严格收尾"""
    text = text.strip()
    text = re.sub(r"令人忍俊不禁.*$", "", text).strip()

    # 循环移除首部可能遗留的多层编号或前缀符号
    while True:
        prev = text
        text = re.sub(
            r"^(?:"
            r"【\s*笑点\s*(?:\d+|[一二三四五六七八九十]+)?\s*】|"
            r"\[\s*笑点\s*(?:\d+|[一二三四五六七八九十]+)?\s*\]|"
            r"笑点\s*[（(]\s*(?:\d+|[一二三四五六七八九十]+)\s*[)）]\s*[:：、.·\-—\u3002\uff0e是为]?\s*|"
            r"笑点\s*(?:\d+|[一二三四五六七八九十]+)\s*[:：、.·\-—\u3002\uff0e是为]?\s*|"
            r"笑点\s*[:：、.·\-—\u3002\uff0e是为]\s*|"
            r"第\s*(?:\d+|[一二三四五六七八九十]+)\s*(?:个笑点|个|点|条)\s*[:：、.·\-—\u3002\uff0e是为]?\s*|"
            r"[（(]\s*(?:\d+|[一二三四五六七八九十]+|[ivxIVX]+|[a-zA-Z])\s*[)）]\s*[:：、.·\-—\u3002\uff0e是为]?\s*|"
            r"[①②③④⑤⑥⑦⑧⑨⑩ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]\s*[:：、.·\-—\u3002\uff0e是为]?\s*|"
            r"(?:\d+|[一二三四五六七八九十]+|[ivxIVX]+|[a-zA-Z])\s*[:：、.·\-—\u3002\uff0e)）是为]\s*"
            r")",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        text = re.sub(r"^[：:、.·\-—\u3002\uff0e)）是为\s]+", "", text).strip()
        if text == prev:
            break

    # 循环清理末尾所有的标点与已有各种“好笑”修饰词
    while True:
        prev = text
        text = re.sub(r"[。！!~，,；;、…\s\?？:：]+$", "", text).strip()
        text = re.sub(
            r"[，,；;\s\?？:：]*\s*(?:(?:确实|真|真的|特别|非常|十分|极其|太)?\s*(?:很)?\s*好笑|好笑)[。！!~啊吧哈啦了\?？\s]*$",
            "",
            text,
        ).strip()
        if text == prev:
            break

    text = re.sub(r"[。！!~，,；;、…\s\?？:：]+$", "", text).strip()
    if not text:
        return "很好笑。"
    return f"{text}，很好笑。"


def _format_failure_response(raw_explanation: str, user_id: str | int | None = None) -> str:
    """
    格式化失败或无笑点说明，保留首尾统一，但中间严禁带编号且绝对不得追加「很好笑。」
    """
    raw_explanation = _clean_markdown(raw_explanation)
    raw_explanation = sanitize_media_mentions(raw_explanation)

    lines = [l.strip() for l in raw_explanation.split("\n") if l.strip()]
    content_lines = []
    for line in lines:
        if re.match(r"^笑点解析\s*[:：]?$", line):
            continue
        if re.match(r"^令人忍俊不禁.*$", line):
            break
        while True:
            prev = line
            line = re.sub(
                r"^(?:"
                r"【\s*笑点\s*(?:\d+|[一二三四五六七八九十]+)?\s*】|"
                r"\[\s*笑点\s*(?:\d+|[一二三四五六七八九十]+)?\s*\]|"
                r"笑点\s*[（(]\s*(?:\d+|[一二三四五六七八九十]+)\s*[)）]\s*[:：、.·\-—\u3002\uff0e是为]?\s*|"
                r"笑点\s*(?:\d+|[一二三四五六七八九十]+)\s*[:：、.·\-—\u3002\uff0e是为]?\s*|"
                r"笑点\s*[:：、.·\-—\u3002\uff0e是为]\s*|"
                r"第\s*(?:\d+|[一二三四五六七八九十]+)\s*(?:个笑点|个|点|条)\s*[:：、.·\-—\u3002\uff0e是为]?\s*|"
                r"[（(]\s*(?:\d+|[一二三四五六七八九十]+|[ivxIVX]+|[a-zA-Z])\s*[)）]\s*[:：、.·\-—\u3002\uff0e是为]?\s*|"
                r"[①②③④⑤⑥⑦⑧⑨⑩ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]\s*[:：、.·\-—\u3002\uff0e是为]?\s*|"
                r"(?:\d+|[一二三四五六七八九十]+|[ivxIVX]+|[a-zA-Z])\s*[:：、.·\-—\u3002\uff0e)）是为]\s*"
                r")",
                "",
                line,
                flags=re.IGNORECASE,
            ).strip()
            line = re.sub(r"^[：:、.·\-—\u3002\uff0e)）是为\s]+", "", line).strip()
            if line == prev:
                break

        line = re.sub(r"令人忍俊不禁.*$", "", line).strip()
        line = re.sub(
            r"[，,；;\s\?？:：]*\s*(?:(?:确实|真|真的|特别|非常|十分|极其|太)?\s*(?:很)?\s*好笑|好笑)[。！!~啊吧哈啦了\?？\s]*",
            "",
            line,
        ).strip()
        if line:
            content_lines.append(line)

    merged_lines: list[str] = []
    for l in content_lines:
        if not merged_lines:
            merged_lines.append(l)
        else:
            prev = merged_lines[-1]
            if re.search(r"[\u4e00-\u9fa5。！!？?，,；;：:]$", prev) and re.match(r"^[\u4e00-\u9fa5。！!？?，,；;：:]", l):
                if not prev.endswith(("。", "！", "？", "!", "?", "，", ",")):
                    merged_lines[-1] = prev + "，" + l
                else:
                    merged_lines[-1] = prev + l
            else:
                merged_lines[-1] = prev + " " + l

    body = "".join(merged_lines).strip()
    if not body:
        body = "现有信息不足以判断笑点。"
    body = re.sub(r"令人忍俊不禁.*$", "", body).strip()
    body = re.sub(
        r"[，,；;\s\?？:：]*\s*(?:(?:确实|真|真的|特别|非常|十分|极其|太)?\s*(?:很)?\s*好笑|好笑)[。！!~啊吧哈啦了\?？\s]*",
        "",
        body,
    ).strip()
    body = re.sub(r"[，,；;、…\s:：]+$", "", body).strip()
    if not body.endswith(("。", "！", "？", "!", "?")):
        body += "。"

    uid = str(user_id or "").strip()
    if uid.startswith("[") and uid.endswith("]"):
        uid = uid[1:-1].strip()
    footer = f"{FOOTER}。[{uid}]问的" if uid else FOOTER
    return f"{HEADER}\n{body}\n{footer}"


def validate_punchline_format(text: str) -> tuple[bool, str]:
    """
    严格校验输出是否符合经典纯文本规范。
    返回 (是否合规, 失败原因描述)。
    """
    if not isinstance(text, str):
        return False, "输出必须为字符串类型"

    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if len(lines) < 3:
        return False, f"输出行数不足 3 行 (当前 {len(lines)} 行): {text!r}"

    if lines[0] != HEADER:
        return False, f"首行必须且仅能为 '{HEADER}', 当前为 '{lines[0]}'"

    if not (lines[-1] == FOOTER or lines[-1] == f"{FOOTER}。" or re.match(r"^令人忍俊不禁[。]?(?:\s*\[.+?\]问的)?$", lines[-1])):
        return False, f"尾行必须为 '{FOOTER}' 或 '令人忍俊不禁[用户id]问的', 当前为 '{lines[-1]}'"

    middle_lines = lines[1:-1]
    
    # 检查是否包含 Markdown 语法
    for idx, line in enumerate(middle_lines, 1):
        if re.search(r"[*#_`]", line):
            return False, f"第 {idx} 项包含 Markdown 标记: {line}"
        if re.match(r"^[-*+]\s+", line):
            return False, f"第 {idx} 项包含 Markdown 列表符号: {line}"
        if "辅助截图" in line or "抽帧截图" in line:
            return False, f"第 {idx} 项提及了辅助截图: {line}"

    # 判断是正常笑点还是降级/失败说明
    is_numbered = any(re.match(r"^笑点\d+：", l) for l in middle_lines)

    if is_numbered:
        # 正常笑点分支：
        for i, line in enumerate(middle_lines, 1):
            expected_prefix = f"笑点{i}："
            if not line.startswith(expected_prefix):
                return False, f"笑点编号必须连续，第 {i} 项期望 '{expected_prefix}', 当前为 '{line}'"
            if not line.endswith("很好笑。"):
                return False, f"第 {i} 项必须以 '很好笑。' 收尾: '{line}'"
            if re.search(r"[？?：:]\s*，", line):
                return False, f"第 {i} 项包含不合规的标点连用 (如 '？，' 或 '：，'): '{line}'"
            if re.search(r"很好笑.*很好笑", line):
                return False, f"第 {i} 项包含重复的 '很好笑' 叠词: '{line}'"
            item_body = line[len(expected_prefix):]
            if _is_failure_text(item_body):
                return False, f"第 {i} 项为失败/降级说明，严禁编造成编号笑点: '{line}'"
        return True, "OK (正常分析格式)"
    else:
        # 失败/降级/无笑点分支：
        for i, line in enumerate(middle_lines, 1):
            if "很好笑" in line:
                return False, f"失败/降级说明严禁追加 '很好笑。': '{line}'"
        return True, "OK (失败/降级说明格式)"
