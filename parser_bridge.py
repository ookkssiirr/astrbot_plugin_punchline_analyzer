from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import aiohttp
from astrbot.api import logger

# B站 BV号 与 AV号 匹配正则
BV_PATTERN = re.compile(r"(?:^|[^0-9a-zA-Z])(?P<bvid>BV[0-9a-zA-Z]{10})(?:[^0-9a-zA-Z]|$)", re.IGNORECASE)
AV_PATTERN = re.compile(r"(?:^|[^0-9a-zA-Z])(?P<avid>av\d{4,})(?:[^0-9a-zA-Z]|$)", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
DIRECT_VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".m4v", ".flv", ".3gp", ".avi", ".mkv")


def is_direct_video_url(url: str) -> bool:
    """判断是否为直链视频 URL"""
    if not url:
        return False
    clean = url.split("?")[0].lower()
    return clean.endswith(DIRECT_VIDEO_EXTENSIONS)


def is_video_segment(seg: Any) -> bool:
    """判定对象是否为视频消息组件或视频消息段字典"""
    if seg is None:
        return False
    if isinstance(seg, dict):
        return seg.get("type") == "video"
    stype = getattr(seg, "type", None) or type(seg).__name__
    return stype in ("video", "Video", "MockVideo")


def extract_video_info(video_source: Any) -> tuple[str | None, str | None]:
    """
    统一解析提取视频组件中的 (video_ref, file_id)
    全面兼容 AstrBot Video 组件、OneBot v11 原始消息段字典及各种 Mock 对象：
    - url: 远程直链 (http://, https://)
    - path: 本地绝对路径或 file:// URI
    - file: 既可能是 URL、本地路径、file:// URI，也可能是 OneBot file_id
    - file_id / id: 协议端文件标识符
    返回规范化元组: (video_ref, file_id)
    其中 video_ref 保证为有效的远程 URL 或本地路径；file_id 为供协议端 get_file 使用的唯一标识。
    """
    if not video_source:
        return None, None

    raw_url = None
    raw_path = None
    raw_file = None
    raw_file_id = None

    if isinstance(video_source, dict):
        d = video_source.get("data") if isinstance(video_source.get("data"), dict) else video_source
        raw_url = d.get("url")
        raw_path = d.get("path")
        raw_file = d.get("file")
        raw_file_id = d.get("file_id") or d.get("id")
    else:
        raw_url = getattr(video_source, "url", None)
        raw_path = getattr(video_source, "path", None)
        raw_file = getattr(video_source, "file", None)
        raw_file_id = getattr(video_source, "file_id", None) or getattr(video_source, "id", None)

    resolved_ref: str | None = None
    resolved_fid: str | None = None

    # 1. 规范化 raw_url
    if raw_url and isinstance(raw_url, str):
        u_str = raw_url.strip()
        if u_str.startswith(("http://", "https://")):
            resolved_ref = u_str

    # 2. 规范化 raw_path
    if not resolved_ref and raw_path and isinstance(raw_path, (str, Path)):
        p_str = str(raw_path).strip()
        if p_str.startswith("file://"):
            p_str = p_str[7:]
        if p_str and (os.path.isabs(p_str) or os.path.isfile(p_str)):
            resolved_ref = p_str

    # 3. 规范化 raw_file (可能是 url, 可能是 local path, 也可能是 file_id)
    if raw_file and isinstance(raw_file, (str, Path)):
        f_str = str(raw_file).strip()
        if f_str.startswith(("http://", "https://")):
            resolved_ref = resolved_ref or f_str
        elif f_str.startswith("file://"):
            p_str = f_str[7:]
            resolved_ref = resolved_ref or p_str
        elif os.path.isfile(f_str) or f_str.startswith("/") or re.match(r"^[a-zA-Z]:[/\\]", f_str):
            resolved_ref = resolved_ref or f_str
        else:
            resolved_fid = resolved_fid or f_str

    # 4. 规范化 raw_file_id
    if raw_file_id and isinstance(raw_file_id, (str, int)):
        fid_str = str(raw_file_id).strip()
        if fid_str and not resolved_fid:
            resolved_fid = fid_str

    return resolved_ref, resolved_fid


def extract_urls_from_text(text: str) -> list[str]:
    """从纯文本中提取所有可解析的 URL 或 B站视频链接"""
    if not text:
        return []

    urls: list[str] = []
    # 提取完整 http/https 链接
    for match in URL_PATTERN.finditer(text):
        url = match.group(0).rstrip(".,;!?'\")>")
        if url not in urls:
            urls.append(url)

    # 提取独立 BV号并转为标准链接
    for match in BV_PATTERN.finditer(text):
        bvid = match.group("bvid")
        bvid_url = f"https://www.bilibili.com/video/{bvid}"
        if bvid_url not in urls and not any(bvid in u for u in urls):
            urls.append(bvid_url)

    # 提取独立 av号
    for match in AV_PATTERN.finditer(text):
        avid = match.group("avid")
        avid_url = f"https://www.bilibili.com/video/{avid}"
        if avid_url not in urls and not any(avid in u for u in urls):
            urls.append(avid_url)

    return urls


def extract_url_from_json_segment(data: dict | str) -> str | None:
    """从 QQ 小程序 / Ark 卡片等 JSON 消息段提取目标跳转 URL"""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return None

    if not isinstance(data, dict):
        return None

    meta = data.get("meta")
    if not isinstance(meta, dict):
        meta = {}

    priority_paths = (
        ("miniapp", "jumpUrl"),
        ("miniapp", "legacyUrl"),
        ("miniapp", "pcJumpUrl"),
        ("miniapp", "sourceUrl"),
        ("miniapp", "url"),
        ("detail_1", "qqdocurl"),
        ("detail_1", "jumpUrl"),
        ("detail_1", "url"),
        ("detail_1", "sourceUrl"),
        ("news", "jumpUrl"),
        ("news", "url"),
    )

    for outer_key, inner_key in priority_paths:
        sub = meta.get(outer_key)
        if isinstance(sub, dict):
            val = sub.get(inner_key)
            if isinstance(val, str) and val.startswith(("http://", "https://")):
                return val

    # 深度递归查找 http 链接
    def _search(obj: Any) -> str | None:
        if isinstance(obj, str):
            if obj.startswith(("http://", "https://")):
                return obj
        elif isinstance(obj, dict):
            for v in obj.values():
                res = _search(v)
                if res:
                    return res
        elif isinstance(obj, list):
            for item in obj:
                res = _search(item)
                if res:
                    return res
        return None

    return _search(meta)


async def run_cancellable_subprocess(
    *cmd: str,
    timeout: float = 15.0,
) -> tuple[int, bytes, bytes]:
    """
    运行外部子进程，确保在超时或外层 asyncio.CancelledError 时立即终止并等待子进程退出，
    防止僵尸进程或文件占用导致后续清理失败。
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(timeout):
            stdout, stderr = await proc.communicate()
            return proc.returncode or 0, stdout, stderr
    except (asyncio.CancelledError, asyncio.TimeoutError):
        try:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
        except Exception:
            pass
        raise


async def download_file_with_limit(
    url: str,
    output_path: Path | str,
    max_bytes: int,
    timeout: float = 20.0,
    session: aiohttp.ClientSession | None = None,
    req_ctx: RequestContext | None = None,
) -> bool:
    """
    流式下载远程媒体文件，严格执行字节上限与超时控制。
    支持受控流式写入与中途取消清理。一旦超过字节上限或发生取消，立即终止并清理未完成的残留文件。
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    owns_session = session is None
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    active_session = session or aiohttp.ClientSession(timeout=client_timeout)
    if req_ctx:
        req_ctx.track_session(active_session)

    try:
        try:
            get_cm = active_session.get(url, timeout=client_timeout)
        except TypeError:
            get_cm = active_session.get(url)
        async with get_cm as resp:
            if resp.status != 200:
                logger.warning(f"[笑点解析] 下载失败，HTTP 状态码: {resp.status}, url: {url}")
                return False

            # 1. 预检查 Content-Length 响应头（若服务端提供）
            content_len = resp.headers.get("Content-Length")
            if content_len:
                try:
                    total_size = int(content_len)
                    if total_size > max_bytes:
                        logger.warning(
                            f"[笑点解析] 目标文件体积 {total_size / (1024*1024):.1f}MB 超过限制 {max_bytes / (1024*1024):.1f}MB，放弃下载"
                        )
                        return False
                except ValueError:
                    pass

            # 2. 流式写入并实时累加字节数（受控流式写入，字节上限阻断）
            downloaded = 0
            with open(out_p, "wb") as f:
                async for chunk in resp.content.iter_chunked(64 * 1024):
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        logger.warning(
                            f"[笑点解析] 流式下载已达字节上限 {max_bytes / (1024*1024):.1f}MB，强行中止"
                        )
                        f.close()
                        if out_p.exists():
                            out_p.unlink()
                        return False
                    f.write(chunk)

            return out_p.exists() and out_p.stat().st_size > 0

    except (asyncio.CancelledError, Exception) as e:
        if out_p.exists():
            try:
                out_p.unlink()
            except Exception:
                pass
        if isinstance(e, asyncio.CancelledError):
            raise
        logger.warning(f"[笑点解析] 下载异常: {e}, url: {url}")
        return False
    finally:
        if owns_session and not getattr(active_session, "closed", False) and hasattr(active_session, "close"):
            close_ret = active_session.close()
            if asyncio.iscoroutine(close_ret):
                await close_ret



async def get_video_duration(video_path: str | Path) -> float | None:
    """使用 ffprobe 获取视频时长（秒），支持取消安全控制"""
    video_str = str(video_path)
    if not os.path.isfile(video_str):
        return None

    try:
        ret, stdout, _ = await run_cancellable_subprocess(
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_str,
            timeout=5.0,
        )
        if ret == 0:
            dur_str = stdout.decode("utf-8", errors="ignore").strip()
            if dur_str:
                return float(dur_str)
    except Exception as e:
        logger.debug(f"[笑点解析] ffprobe 获取视频时长异常: {e}")
    return None


HARD_CAP_MAX_MEDIA_MB = 50
HARD_CAP_MAX_VIDEO_DURATION = 180
HARD_CAP_VIDEO_SAMPLE_FRAMES = 4


async def sample_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    num_frames: int = 4,
    max_duration: int = 180,
    max_media_mb: int = 50,
    req_ctx: RequestContext | None = None,
) -> list[str]:
    """使用 ffmpeg 均匀抽帧，提取代表性画面，具备体积检测与中途取消保护"""
    max_media_mb = min(HARD_CAP_MAX_MEDIA_MB, max(5, int(max_media_mb)))
    max_duration = min(HARD_CAP_MAX_VIDEO_DURATION, max(10, int(max_duration)))
    num_frames = min(HARD_CAP_VIDEO_SAMPLE_FRAMES, max(1, int(num_frames)))
    video_str = str(video_path)
    if not os.path.isfile(video_str):
        return []

    # 1. 检查物理文件大小，防止过大视频撑爆内存与磁盘
    file_size_mb = os.path.getsize(video_str) / (1024 * 1024)
    if file_size_mb > max_media_mb:
        logger.warning(
            f"[笑点解析] 视频大小 {file_size_mb:.1f}MB 超过配置上限 {max_media_mb}MB，跳过抽帧以保护内存"
        )
        return []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = await get_video_duration(video_str)
    if duration is None or duration <= 0:
        logger.warning(f"[笑点解析] 无法获取视频时长，跳过抽帧: {video_str}")
        return []

    if duration > max_duration:
        logger.info(
            f"[笑点解析] 视频时长 {duration:.1f}s 超过上限 {max_duration}s，为防止占用过大资源不予抽帧"
        )
        return []

    # 选取均匀时间戳（避开最开头的黑屏或片头，自适应支持短视频）
    if duration <= 1.0:
        timestamps = [(i + 0.5) * duration / num_frames for i in range(num_frames)]
    else:
        timestamps = [
            max(0.5, min(duration - 0.5, (i + 1) * duration / (num_frames + 1)))
            for i in range(num_frames)
        ]

    frame_paths: list[str] = []
    unique_run_id = uuid.uuid4().hex[:6]
    for idx, ts in enumerate(timestamps):
        out_file = out_dir / f"frame_{idx + 1}_{int(ts * 1000)}_{unique_run_id}.jpg"
        try:
            ret, _, _ = await run_cancellable_subprocess(
                "ffmpeg",
                "-y",
                "-ss",
                f"{ts:.2f}",
                "-i",
                video_str,
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(out_file),
                timeout=8.0,
            )
            if ret == 0 and out_file.exists() and out_file.stat().st_size > 0:
                frame_paths.append(str(out_file))
                if req_ctx:
                    req_ctx.add_temp_file(str(out_file))
            elif out_file.exists():
                try:
                    out_file.unlink()
                except Exception:
                    pass
        except asyncio.CancelledError:
            # 中途取消时若生成了半截文件，立即清除
            if out_file.exists():
                try:
                    out_file.unlink()
                except Exception:
                    pass
            raise
        except Exception as e:
            logger.warning(f"[笑点解析] 抽帧失败 (ts={ts}): {e}")

    return frame_paths


class RequestContext:
    """
    单次解析请求专属的任务上下文。
    负责管理当前请求创建的所有异步任务（下载、抽帧等）、专属网络 ClientSession、
    独立解析器上下文及专属任务目录。

    生命周期与清理时序规则：
    1. 正常流程：必须等待模型（LLM）完成图片/多模态输入读取并生成回复后，方可删除任务目录与临时文件；
    2. 异常、超时或取消流程：必须先显式取消本请求创建的所有后台异步任务（下载、抽帧等），
       并通过 await asyncio.gather(..., return_exceptions=True) 等待任务彻底退出，
       然后安全关闭本请求专属的 ClientSession 与网络连接，最后彻底清理任务目录与临时文件。
    """

    def __init__(self, task_dir: Path | str):
        self.task_dir = Path(task_dir)
        self.tasks: set[asyncio.Task] = set()
        self.sessions: list[aiohttp.ClientSession] = []
        self.iso_contexts: list[Any] = []
        self.temp_files: list[str | Path] = []
        self._closed: bool = False
        try:
            self.owner_task: asyncio.Task | None = asyncio.current_task()
        except Exception:
            self.owner_task = None

    def track_task(self, task: asyncio.Task | None) -> asyncio.Task | None:
        if task is not None and isinstance(task, (asyncio.Task, asyncio.Future)):
            if self.owner_task is None:
                try:
                    self.owner_task = asyncio.current_task()
                except Exception:
                    pass
            if self.owner_task is not None and task is self.owner_task:
                return task
            self.tasks.add(task)
            task.add_done_callback(lambda t: self.tasks.discard(t))
        return task

    def track_session(self, session: aiohttp.ClientSession | None) -> aiohttp.ClientSession | None:
        if session is not None and session not in self.sessions:
            self.sessions.append(session)
        return session

    def track_iso_context(self, iso_ctx: Any | None) -> Any | None:
        if iso_ctx is not None and iso_ctx not in self.iso_contexts:
            self.iso_contexts.append(iso_ctx)
            if hasattr(iso_ctx, "downloader") and hasattr(iso_ctx.downloader, "client"):
                client = getattr(iso_ctx.downloader, "client", None)
                if isinstance(client, aiohttp.ClientSession):
                    self.track_session(client)
        return iso_ctx

    def add_temp_file(self, filepath: str | Path):
        if filepath and filepath not in self.temp_files:
            self.temp_files.append(filepath)

    async def cancel_and_wait_tasks(self):
        """显式取消本请求创建的所有后台异步任务，并等待彻底退出（严格排除当前正在执行清理的协程与请求主任务，杜绝自身取消递归与 gather 死锁）"""
        current = asyncio.current_task()
        if self.owner_task is None and current is not None:
            self.owner_task = current
        for iso in self.iso_contexts:
            if hasattr(iso, "tasks"):
                for t in list(iso.tasks):
                    if isinstance(t, (asyncio.Task, asyncio.Future)):
                        if t is not current and (self.owner_task is None or t is not self.owner_task):
                            self.tasks.add(t)

        pending = [
            t for t in list(self.tasks)
            if not t.done() and t is not current and (self.owner_task is None or t is not self.owner_task)
        ]
        for t in pending:
            t.cancel()
        if pending:
            cancelled = False
            cur = asyncio.current_task()
            while not all(t.done() for t in pending):
                try:
                    await asyncio.shield(asyncio.gather(*pending, return_exceptions=True))
                except asyncio.CancelledError:
                    if not all(t.done() for t in pending) or (hasattr(cur, "cancelling") and cur.cancelling() > 0):
                        cancelled = True
                    else:
                        break
                except Exception:
                    break
            if cancelled:
                raise asyncio.CancelledError()
        self.tasks.clear()

    async def close_sessions_and_contexts(self):
        """安全关闭本请求专属的 ClientSession 与网络连接及独立解析器上下文（抵御调用者取消）"""
        cancelled = False
        for iso in list(self.iso_contexts):
            if hasattr(iso, "close") and callable(iso.close):
                close_task = asyncio.create_task(iso.close())
                cur = asyncio.current_task()
                while not close_task.done():
                    try:
                        await asyncio.shield(close_task)
                    except asyncio.CancelledError:
                        if not close_task.done() or (hasattr(cur, "cancelling") and cur.cancelling() > 0):
                            cancelled = True
                        else:
                            break
                    except Exception as e:
                        logger.debug(f"[笑点解析] 关闭独立解析器上下文异常: {e}")
                        break

        # 关闭确认完成前，不能清空资源登记
        self.iso_contexts = [iso for iso in self.iso_contexts if not getattr(iso, "_closed", False)]

        for s in list(self.sessions):
            try:
                if not s.closed:
                    close_task = asyncio.create_task(s.close())
                    cur = asyncio.current_task()
                    while not close_task.done():
                        try:
                            await asyncio.shield(close_task)
                        except asyncio.CancelledError:
                            if not close_task.done() or (hasattr(cur, "cancelling") and cur.cancelling() > 0):
                                cancelled = True
                            else:
                                break
                        except Exception:
                            break
            except asyncio.CancelledError:
                cancelled = True
            except Exception as e:
                logger.debug(f"[笑点解析] 关闭专属 ClientSession 异常: {e}")

        # 关闭确认完成前，不能清空资源登记
        self.sessions = [s for s in self.sessions if not s.closed]

        if cancelled:
            raise asyncio.CancelledError()

    def cleanup_disk(self):
        """彻底清理任务目录与临时文件"""
        for f in list(self.temp_files):
            try:
                p = Path(f)
                if p.exists() and p.is_file():
                    try:
                        p.unlink()
                    except PermissionError:
                        import stat
                        os.chmod(str(p), stat.S_IWRITE)
                        p.unlink()
            except Exception:
                pass
        self.temp_files.clear()

        if self.task_dir.exists():
            def _remove_readonly(func, path, exc_info):
                import stat
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except Exception:
                    pass

            for attempt in range(3):
                try:
                    shutil.rmtree(self.task_dir, onerror=_remove_readonly)
                    break
                except Exception as e:
                    if attempt == 2:
                        shutil.rmtree(self.task_dir, ignore_errors=True)
                        logger.debug(f"[笑点解析] 清理专属任务目录最终重试: {e}")
                    else:
                        import time
                        time.sleep(0.05)

    async def cancel_and_cleanup(self):
        """
        异常、超时或取消时执行精确清理时序：
        1. 显式取消所有后台异步任务并等待其彻底退出
        2. 安全关闭 ClientSession 与网络连接及独立解析器上下文（抵御调用者取消）
        3. 关闭确认完成前，绝不能清空资源登记、标记请求完成或删除目录
        4. 确认所有资源关闭后，彻底清理任务目录与临时文件并标记完成
        5. 若等待过程被取消，在完成上述清理后按约定传递 CancelledError
        """
        if self._closed:
            return
        cancelled = False
        try:
            try:
                await self.cancel_and_wait_tasks()
            except asyncio.CancelledError:
                cancelled = True
            except Exception as e:
                logger.debug(f"[笑点解析] cancel_and_wait_tasks 异常: {e}")

            try:
                await self.close_sessions_and_contexts()
            except asyncio.CancelledError:
                cancelled = True
            except Exception as e:
                logger.debug(f"[笑点解析] close_sessions_and_contexts 异常: {e}")
        finally:
            all_isos_closed = len(self.iso_contexts) == 0
            all_sessions_closed = len(self.sessions) == 0
            all_tasks_done = not self.tasks or all(t.done() for t in self.tasks)

            if all_isos_closed and all_sessions_closed and all_tasks_done:
                self.cleanup_disk()
                self._closed = True

            cur = asyncio.current_task()
            if cancelled or (hasattr(cur, "cancelling") and cur.cancelling() > 0):
                raise asyncio.CancelledError()

    async def close_normal(self):
        """
        正常流程完成时执行清理：
        必须等待模型完成多模态输入读取并生成回复后调用。
        首先确保取消并等待任何可能存在的挂起后台任务彻底退出，然后安全关闭网络连接与独立解析器上下文，
        关闭确认完成前绝不删除目录或标记请求完成，最后彻底清理任务目录并按约定传递 CancelledError。
        """
        if self._closed:
            return
        cancelled = False
        try:
            try:
                await self.cancel_and_wait_tasks()
            except asyncio.CancelledError:
                cancelled = True
            except Exception as e:
                logger.debug(f"[笑点解析] cancel_and_wait_tasks 异常: {e}")

            try:
                await self.close_sessions_and_contexts()
            except asyncio.CancelledError:
                cancelled = True
            except Exception as e:
                logger.debug(f"[笑点解析] close_sessions_and_contexts 异常: {e}")
        finally:
            all_isos_closed = len(self.iso_contexts) == 0
            all_sessions_closed = len(self.sessions) == 0
            all_tasks_done = not self.tasks or all(t.done() for t in self.tasks)

            if all_isos_closed and all_sessions_closed and all_tasks_done:
                self.cleanup_disk()
                self._closed = True

            cur = asyncio.current_task()
            if cancelled or (hasattr(cur, "cancelling") and cur.cancelling() > 0):
                raise asyncio.CancelledError()

    async def ensure_cleanup(self):
        """兜底确保一切资源均已安全释放与清理"""
        if not self._closed or self.iso_contexts or self.sessions:
            try:
                await self.cancel_and_cleanup()
            except asyncio.CancelledError:
                pass
        if self._closed:
            self.cleanup_disk()


class IsolatedConfigProxy:
    """针对当前解析请求的只读配置代理，覆盖本请求限额与专属任务目录，绝不污染或修改共享配置单例"""
    def __init__(
        self,
        orig_cfg: Any,
        max_media_mb: int = 50,
        max_video_duration: int = 180,
        task_dir: Path | str | None = None,
    ):
        self._orig = orig_cfg
        max_media_mb = min(HARD_CAP_MAX_MEDIA_MB, max(5, int(max_media_mb)))
        max_video_duration = min(HARD_CAP_MAX_VIDEO_DURATION, max(10, int(max_video_duration)))
        self.source_max_size = max_media_mb
        self.max_size = max_media_mb * 1024 * 1024
        self.max_duration = max_video_duration
        self.source_max_minute = max_video_duration // 60
        if task_dir is not None:
            p_task_dir = Path(task_dir)
            self.cache_dir = p_task_dir
            self.temp_dir = p_task_dir
            self.download_dir = p_task_dir
            self.save_dir = p_task_dir
            self.data_dir = p_task_dir
            self.output_dir = p_task_dir

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if hasattr(self._orig, "get") and callable(self._orig.get):
            return self._orig.get(key, default)
        return getattr(self._orig, key, default)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if hasattr(self._orig, "__getitem__"):
            return self._orig[key]
        return getattr(self._orig, key)

    def __contains__(self, key: str) -> bool:
        if hasattr(self, key):
            return True
        if hasattr(self._orig, "__contains__"):
            return key in self._orig
        return hasattr(self._orig, key)

    def save_config(self) -> None:
        """保护性空实现，绝不向物理磁盘回写修改"""
        pass


class IsolatedParserContext:
    """本请求专属的独立解析与下载上下文，持有独立的下载器与平台解析器实例"""
    def __init__(self, parser: Any = None, downloader: Any = None):
        self.parser = parser
        self.downloader = downloader
        self.tasks: set[asyncio.Task] = set()
        self._closed: bool = False
        self._close_task: asyncio.Task | None = None

    def is_fully_closed(self) -> bool:
        """检查所有关联的底层 ClientSession 与异步任务是否均已确实验证关闭"""
        if self.tasks and not all(t.done() for t in self.tasks):
            return False

        for target in (self.downloader, self.parser):
            if target is not None:
                for attr in ("client", "_client", "session", "_session", "http_session"):
                    sess = getattr(target, attr, None)
                    if sess is not None:
                        if hasattr(sess, "closed") and not getattr(sess, "closed", True):
                            return False
                        if hasattr(sess, "is_closed") and not getattr(sess, "is_closed", True):
                            return False
                if hasattr(target, "closed") and not getattr(target, "closed", True):
                    return False
                if hasattr(target, "is_closed") and not getattr(target, "is_closed", True):
                    return False
        return True

    async def close(self):
        """安全释放本请求独立的网络会话、异步任务与解析器会话"""
        if getattr(self, "_closed", False):
            return

        # 若当前已有执行中的关闭任务，作为等待者等待该任务彻底完成，绝不启动第二次关闭
        if self._close_task is not None and not self._close_task.done():
            if self._close_task is not asyncio.current_task():
                target_task = self._close_task
                cancelled = False
                cur = asyncio.current_task()
                while not target_task.done():
                    try:
                        await asyncio.shield(target_task)
                    except asyncio.CancelledError:
                        if not target_task.done() or (hasattr(cur, "cancelling") and cur.cancelling() > 0):
                            cancelled = True
                        else:
                            break
                    except Exception:
                        break
                # 等待者被取消时，必须等待原关闭真正结束后抛出 CancelledError，绝不启动第二次关闭
                if cancelled:
                    raise asyncio.CancelledError()
                if getattr(self, "_closed", False):
                    return

        self._close_task = asyncio.current_task()
        had_failure = False
        try:
            current = asyncio.current_task()
            pending = [t for t in list(self.tasks) if not t.done() and t is not current]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            self.tasks.clear()

            if self.downloader:
                if hasattr(self.downloader, "close") and callable(self.downloader.close):
                    try:
                        close_ret = self.downloader.close()
                        if asyncio.iscoroutine(close_ret):
                            await close_ret
                    except Exception as e:
                        had_failure = True
                        logger.debug(f"[笑点解析] 关闭独立下载器会话异常: {e}")
                for attr in ("client", "_client", "session", "_session", "http_session"):
                    sess = getattr(self.downloader, attr, None)
                    if sess and hasattr(sess, "close") and not getattr(sess, "closed", False):
                        try:
                            close_ret = sess.close()
                            if asyncio.iscoroutine(close_ret):
                                await close_ret
                        except Exception as e:
                            had_failure = True
                            logger.debug(f"[笑点解析] 关闭独立下载器 {attr} 异常: {e}")

            if self.parser:
                if hasattr(self.parser, "close_session") and callable(self.parser.close_session):
                    try:
                        close_ret = self.parser.close_session()
                        if asyncio.iscoroutine(close_ret):
                            await close_ret
                    except Exception as e:
                        had_failure = True
                        logger.debug(f"[笑点解析] 关闭独立解析器会话异常: {e}")
                sess = getattr(self.parser, "_session", None) or getattr(self.parser, "session", None)
                if sess and hasattr(sess, "close") and not getattr(sess, "closed", False):
                    try:
                        close_ret = sess.close()
                        if asyncio.iscoroutine(close_ret):
                            await close_ret
                    except Exception as e:
                        had_failure = True
                        logger.debug(f"[笑点解析] 关闭独立解析器 session 异常: {e}")

            # 兜底确保 parser/downloader 上的 ClientSession 最终彻底关闭
            for target in (self.downloader, self.parser):
                if target is not None:
                    for attr in ("client", "_client", "session", "_session", "http_session"):
                        sess = getattr(target, attr, None)
                        if sess and hasattr(sess, "close") and not getattr(sess, "closed", False):
                            try:
                                close_ret = sess.close()
                                if asyncio.iscoroutine(close_ret):
                                    await close_ret
                            except Exception:
                                had_failure = True

            # 严格依据实际关闭结果设置完成状态，绝不能仅因流程执行到末尾就标记为成功
            all_closed = self.is_fully_closed()
            if not all_closed:
                self._closed = False
            elif had_failure:
                has_verifiable = False
                for target in (self.downloader, self.parser):
                    if target is not None:
                        for attr in ("client", "_client", "session", "_session", "http_session"):
                            sess = getattr(target, attr, None)
                            if sess is not None and (hasattr(sess, "closed") or hasattr(sess, "is_closed")):
                                has_verifiable = True
                        if hasattr(target, "closed") or hasattr(target, "is_closed"):
                            has_verifiable = True
                self._closed = has_verifiable
            else:
                self._closed = True
        finally:
            self._close_task = None


def _instrument_downloader_tasks(
    downloader: Any,
    iso_ctx: IsolatedParserContext,
    req_ctx: RequestContext | None = None,
):
    """拦截下载器关键异步方法（streamd, download_video, ytdlp_download_video 等），自动追踪派生的 Task 与 Coroutine"""
    target_methods = {
        "streamd",
        "download_video",
        "download_audio",
        "download_file",
        "download_img",
        "download_imgs_without_raise",
        "download_av_and_merge",
        "ytdlp_download_video",
        "ytdlp_download_video_relaxed",
        "ytdlp_download_audio",
    }
    for attr in dir(downloader):
        if (
            attr.startswith(("download_", "stream", "ytdlp_download"))
            or attr in ("streamd",)
        ) and callable(getattr(downloader, attr, None)):
            target_methods.add(attr)

    for method_name in target_methods:
        orig_fn = getattr(downloader, method_name, None)
        if callable(orig_fn):
            def make_wrapper(fn):
                def wrapper(*args, **kwargs):
                    res = fn(*args, **kwargs)
                    if asyncio.iscoroutine(res):
                        async def _coro_wrapper():
                            cur = asyncio.current_task()
                            if cur and (not req_ctx or cur is not req_ctx.owner_task):
                                iso_ctx.tasks.add(cur)
                                cur.add_done_callback(lambda t: iso_ctx.tasks.discard(t))
                                if req_ctx:
                                    req_ctx.track_task(cur)
                            return await res
                        return _coro_wrapper()
                    elif isinstance(res, (asyncio.Task, asyncio.Future)):
                        if not req_ctx or res is not req_ctx.owner_task:
                            iso_ctx.tasks.add(res)
                            res.add_done_callback(lambda t: iso_ctx.tasks.discard(t))
                            if req_ctx:
                                req_ctx.track_task(res)
                    return res
                return wrapper
            setattr(downloader, method_name, make_wrapper(orig_fn))


async def create_isolated_parser_context(
    parser_plugin: Any,
    url: str,
    max_media_mb: int = 50,
    max_video_duration: int = 180,
    task_dir: Path | str | None = None,
    req_ctx: RequestContext | None = None,
) -> tuple[IsolatedParserContext | None, str | None, Any | None]:
    """
    尝试为当前请求构建完全隔离的 (isolated_parser, isolated_downloader) 上下文。
    核心原则：
    1. 绝不修改共享实例 (parser_plugin, downloader, parser_map, cfg) 的任何属性或方法；
    2. 普通万能解析请求与本插件并发执行时，始终保持其原始 90MB 与 300s 限制；
    3. cache_dir 严格指向本请求专属任务目录，不写入共享缓存；
    4. 若当前平台不支持无状态/独立上下文创建，返回 (None, matching_kw, match)，转入安全降级流程。
    """
    key_pattern_list = getattr(parser_plugin, "key_pattern_list", [])
    parser_map = getattr(parser_plugin, "parser_map", {})
    shared_downloader = getattr(parser_plugin, "downloader", None)
    shared_cfg = getattr(parser_plugin, "cfg", None)

    matching_kw = None
    matching_match = None

    if isinstance(key_pattern_list, list):
        for item in key_pattern_list:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                kw, pat = item
                m = None
                if isinstance(pat, str):
                    try:
                        pat = re.compile(pat)
                    except Exception:
                        pass
                if hasattr(pat, "search") and callable(pat.search):
                    m = pat.search(url)
                elif callable(pat):
                    m = pat(url)
                elif kw in url:
                    m = re.search(re.escape(kw), url)
                if m:
                    matching_kw = kw
                    matching_match = m
                    break

    if not matching_kw or matching_kw not in parser_map:
        return None, matching_kw, matching_match

    shared_parser = parser_map[matching_kw]
    parser_cls = getattr(shared_parser, "__class__", None)
    downloader_cls = getattr(shared_downloader, "__class__", None)

    if not parser_cls or not downloader_cls or not shared_cfg:
        return None, matching_kw, matching_match

    isolated_downloader = None
    iso_ctx = None
    try:
        isolated_cfg = IsolatedConfigProxy(
            orig_cfg=shared_cfg,
            max_media_mb=max_media_mb,
            max_video_duration=max_video_duration,
            task_dir=task_dir,
        )
        isolated_downloader = downloader_cls(isolated_cfg)
        if task_dir is not None:
            setattr(isolated_downloader, "cache_dir", Path(task_dir))
            setattr(isolated_downloader, "temp_dir", Path(task_dir))
            setattr(isolated_downloader, "download_dir", Path(task_dir))

        # 立即包装为独立上下文并登记到请求上下文，确保无论后续构造是否抛错，资源均可被安全回收
        iso_ctx = IsolatedParserContext(parser=None, downloader=isolated_downloader)
        if req_ctx:
            req_ctx.track_iso_context(iso_ctx)

        isolated_parser = parser_cls(isolated_cfg, isolated_downloader)
        iso_ctx.parser = isolated_parser
        _instrument_downloader_tasks(isolated_downloader, iso_ctx, req_ctx)

        return iso_ctx, matching_kw, matching_match
    except Exception as e:
        logger.info(f"[笑点解析] 无法为平台 {matching_kw} 构建独立解析器实例: {e}，将启用安全降级策略")
        # 构造失败路径：主动释放已创建的下载器及网络会话资源
        if iso_ctx:
            if req_ctx:
                # 有 req_ctx 时由它在请求生命周期收尾阶段（close_normal / cancel_and_cleanup）统一等待关闭与清理
                pass
            else:
                # 无 req_ctx 时，在资源所属的当前事件循环中直接 await 关闭，杜绝跨事件循环或线程池阻塞
                try:
                    await iso_ctx.close()
                except Exception as close_err:
                    logger.debug(f"[笑点解析] 无 req_ctx 时关闭独立上下文异常: {close_err}")
        return None, matching_kw, matching_match



class ParserBridge:
    """与万能解析器、引用视频理解插件以及远程/直传视频下载的综合适配器"""

    def __init__(self, context: Any) -> None:
        self.context = context

    def get_parser_plugin_instance(self) -> Any | None:
        """获取已加载的 astrbot_plugin_parser 实例"""
        try:
            star_md = self.context.get_registered_star("astrbot_plugin_parser")
            if star_md and star_md.star_cls:
                return star_md.star_cls
        except Exception as e:
            logger.debug(f"[笑点解析] 获取 astrbot_plugin_parser 失败: {e}")
        return None

    def get_video_vision_plugin_instance(self) -> Any | None:
        """获取已加载的引用视频理解插件 astrbot_plugin_video_reference_vision 实例"""
        try:
            star_md = self.context.get_registered_star("astrbot_plugin_video_reference_vision")
            if star_md and star_md.star_cls:
                return star_md.star_cls
        except Exception as e:
            logger.debug(f"[笑点解析] 获取 astrbot_plugin_video_reference_vision 失败: {e}")
        return None

    async def parse_url_with_existing_parser(
        self,
        url: str,
        max_video_duration: int = 180,
        max_media_mb: int = 50,
        num_frames: int = 4,
        task_dir: Path | None = None,
        req_ctx: RequestContext | None = None,
    ) -> dict[str, Any] | None:
        """调用万能解析器解析 URL，并复用其结果（绝不修改共享解析器任何状态）"""
        parser_plugin = self.get_parser_plugin_instance()
        if not parser_plugin:
            logger.debug("[笑点解析] 未找到 astrbot_plugin_parser 实例")
            return None

        result_data: dict[str, Any] = {
            "title": "",
            "text": "",
            "author": "",
            "extra_info": "",
            "cover_path": None,
            "video_path": None,
            "images": [],
            "frame_paths": [],
            "url": url,
            "video_error": "",
        }

        # 1. 尝试为本请求构建专属的独立解析与下载上下文（专属任务目录隔离）
        iso_ctx, matching_kw, matching_match = await create_isolated_parser_context(
            parser_plugin=parser_plugin,
            url=url,
            max_media_mb=max_media_mb,
            max_video_duration=max_video_duration,
            task_dir=task_dir,
            req_ctx=req_ctx,
        )

        parse_res = None
        used_isolated = False

        try:
            if iso_ctx and iso_ctx.parser and matching_kw and matching_match:
                try:
                    # 独立上下文路径：使用专属解析器与专属下载器，全局共享单例不受任何影响
                    parse_coro = iso_ctx.parser.parse(matching_kw, matching_match)
                    if asyncio.iscoroutine(parse_coro):
                        parse_task = asyncio.create_task(parse_coro)
                        if req_ctx:
                            req_ctx.track_task(parse_task)
                        parse_res = await parse_task
                    else:
                        parse_res = parse_coro
                    used_isolated = True
                except Exception as e:
                    err_msg = str(e)
                    if "SizeLimit" in type(e).__name__ or "超过" in err_msg:
                        result_data["video_error"] = f"视频体积超过本插件 {max_media_mb}MB 限额，已在下载前/下载中阻断"
                    elif "Duration" in type(e).__name__ or "时长" in err_msg:
                        result_data["video_error"] = f"视频时长超过本插件 {max_video_duration} 秒上限，已在下载前阻断"
                    else:
                        logger.warning(f"[笑点解析] 独立上下文解析异常: {e}")
                        result_data["video_error"] = f"独立解析异常: {e}，已降级受限处理（封面不代表已读取视频画面）"
                        parse_res = None

            # 2. 降级路径：严格禁止调用共享 parse_url() 或任何可能在后台创建媒体下载任务的接口
            if parse_res is None and not result_data.get("video_error"):
                # 仅允许调用已确认不会下载媒体的元数据接口（必须显式标注 is_metadata_only is True）
                readonly_meta_fn = None
                for fn_name in ("get_metadata", "parse_metadata", "get_info", "parse_info"):
                    cand = getattr(parser_plugin, fn_name, None)
                    if callable(cand) and getattr(cand, "is_metadata_only", False) is True:
                        readonly_meta_fn = cand
                        break

                if readonly_meta_fn:
                    try:
                        meta_res = readonly_meta_fn(url)
                        if asyncio.iscoroutine(meta_res):
                            parse_res = await meta_res
                        else:
                            parse_res = meta_res
                    except Exception as e:
                        logger.warning(f"[笑点解析] 只读元数据解析异常: {e}")
                        parse_res = None

                if parse_res is None:
                    # 第三方解析器未暴露纯元数据接口，明确返回受限提示并降级说明，绝不在共享下载器上发起未受限下载
                    result_data["video_error"] = "共享解析器未提供独立请求隔离或只读元数据接口，已降级受限处理（封面不代表已读取视频画面）"
                    logger.info(f"[笑点解析] {result_data['video_error']}")

            if parse_res is None:
                return result_data if result_data.get("video_error") else None

            result_data["title"] = getattr(parse_res, "title", None) or ""
            result_data["text"] = getattr(parse_res, "text", None) or ""
            result_data["author"] = parse_res.author.name if getattr(parse_res, "author", None) else ""
            result_data["extra_info"] = parse_res.extra.get("info") if getattr(parse_res, "extra", None) else ""
            result_data["url"] = getattr(parse_res, "url", None) or url

            # 获取封面图
            try:
                cp = getattr(parse_res, "cover_path", None)
                if not used_isolated:
                    # 降级路径：仅当封面路径已是本地存在的合法文件路径时直接复用，严禁在共享下载器上发起下载
                    if isinstance(cp, (str, Path)) and os.path.exists(str(cp)):
                        cover_path = str(cp)
                    else:
                        cover_path = None
                else:
                    if asyncio.iscoroutine(cp) or isinstance(cp, asyncio.Future):
                        if req_ctx and isinstance(cp, asyncio.Task):
                            req_ctx.track_task(cp)
                        cover_path = await cp
                    elif callable(cp):
                        res_cp = cp()
                        cover_path = await res_cp if asyncio.iscoroutine(res_cp) else res_cp
                    else:
                        cover_path = cp
                if cover_path and os.path.exists(str(cover_path)):
                    result_data["cover_path"] = str(cover_path)
                    result_data["images"].append(str(cover_path))
                    if req_ctx and used_isolated:
                        req_ctx.add_temp_file(str(cover_path))
            except Exception as e:
                logger.debug(f"[笑点解析] 获取封面路径异常: {e}")

            # 获取图集中的图片
            img_contents = getattr(parse_res, "img_contents", [])
            for ic in img_contents[:6]:
                try:
                    if used_isolated:
                        if hasattr(ic, "path_task") and isinstance(ic.path_task, asyncio.Task) and req_ctx:
                            req_ctx.track_task(ic.path_task)
                        ipath = await ic.get_path()
                    else:
                        # 降级路径：严禁调用共享下载器下载图片，仅读取已存在的本地文件
                        ipath = getattr(ic, "path", None) or getattr(ic, "local_path", None)
                    if ipath and os.path.exists(str(ipath)):
                        result_data["images"].append(str(ipath))
                        if req_ctx and used_isolated:
                            req_ctx.add_temp_file(str(ipath))
                except Exception as e:
                    logger.debug(f"[笑点解析] 获取图片内容异常: {e}")

            # 视频处理
            video_contents = getattr(parse_res, "video_contents", [])
            if video_contents and not result_data.get("video_error"):
                if req_ctx and used_isolated:
                    for vc_item in video_contents:
                        if hasattr(vc_item, "path_task") and isinstance(vc_item.path_task, asyncio.Task):
                            req_ctx.track_task(vc_item.path_task)

                vc = video_contents[0]
                dur_raw = getattr(vc, "duration", None)
                dur = None
                try:
                    if dur_raw is not None:
                        dur = float(dur_raw)
                except (ValueError, TypeError):
                    dur = None
                if dur is not None and dur > max_video_duration:
                    if used_isolated:
                        for vc_item in video_contents:
                            path_task = getattr(vc_item, "path_task", None)
                            if isinstance(path_task, asyncio.Task) and not path_task.done():
                                path_task.cancel()
                                with contextlib.suppress(asyncio.CancelledError, Exception):
                                    await path_task
                            cover_task = getattr(vc_item, "cover", None)
                            if isinstance(cover_task, asyncio.Task) and not cover_task.done():
                                cover_task.cancel()
                                with contextlib.suppress(asyncio.CancelledError, Exception):
                                    await cover_task
                    result_data["video_error"] = f"视频时长 ({dur:.1f}秒) 超过系统上限 {max_video_duration}秒，已在下载前阻断"
                    logger.info(f"[笑点解析] {result_data['video_error']}")
                elif used_isolated:
                    # 独立上下文路径：此时 vc.get_path() 运行在专属的 isolated_downloader 中
                    try:
                        get_path_coro = vc.get_path()
                        if asyncio.iscoroutine(get_path_coro):
                            gp_task = asyncio.create_task(get_path_coro)
                            if req_ctx:
                                req_ctx.track_task(gp_task)
                            v_path = await asyncio.wait_for(gp_task, timeout=15.0)
                        else:
                            v_path = get_path_coro

                        if v_path and os.path.exists(str(v_path)):
                            v_size_mb = os.path.getsize(str(v_path)) / (1024 * 1024)
                            if v_size_mb > max_media_mb:
                                result_data["video_error"] = f"解析器缓存视频体积 ({v_size_mb:.1f}MB) 超过本插件 50MB 限额"
                                logger.warning(f"[笑点解析] {result_data['video_error']}")
                            else:
                                result_data["video_path"] = str(v_path)
                                if req_ctx:
                                    req_ctx.add_temp_file(str(v_path))
                                if task_dir:
                                    frames = await sample_video_frames(
                                        video_path=v_path,
                                        output_dir=task_dir,
                                        num_frames=num_frames,
                                        max_duration=max_video_duration,
                                        max_media_mb=max_media_mb,
                                        req_ctx=req_ctx,
                                    )
                                    result_data["frame_paths"] = frames
                                    if frames:
                                        result_data["images"].extend(frames)
                                    elif not result_data.get("video_error"):
                                        result_data["video_error"] = "未能从解析器视频中成功抽取代表性画面"
                        elif not result_data.get("video_error"):
                            result_data["video_error"] = "未能从解析器获取到有效视频文件"
                    except asyncio.TimeoutError:
                        if isinstance(gp_task, asyncio.Task) and not gp_task.done():
                            gp_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError, Exception):
                                await gp_task
                        result_data["video_error"] = "视频下载等待超时（超过 15 秒）"
                    except Exception as e:
                        err_msg = str(e)
                        if "SizeLimit" in type(e).__name__ or "超过" in err_msg:
                            result_data["video_error"] = f"视频体积超过本插件 {max_media_mb}MB 限额，已在下载前/下载中阻断"
                        elif "Duration" in type(e).__name__ or "时长" in err_msg:
                            result_data["video_error"] = f"视频时长超过本插件 {max_video_duration} 秒上限，已在下载前阻断"
                        else:
                            result_data["video_error"] = f"视频获取或抽帧异常: {e}"
                        logger.warning(f"[笑点解析] {result_data['video_error']}")
                else:
                    # 降级路径：当前接口未能隔离，为保护全局共享解析器，不调用共享 downloader 下载视频
                    result_data["video_error"] = "共享解析器未提供独立请求隔离接口，已降级为基于视频标题与封面分析，避免影响全局并发请求"
                    logger.info(f"[笑点解析] {result_data['video_error']}")

        finally:
            if iso_ctx and not req_ctx:
                await iso_ctx.close()

        return result_data

    async def resolve_quoted_video_via_plugin(
        self,
        event: Any,
        reply_message_id: str | int | None = None,
        video_source: Any | None = None,
        video_ref: str | None = None,
        file_id: str | None = None,
        task_dir: Path | None = None,
        max_video_duration: int = 180,
        max_media_mb: int = 50,
        num_frames: int = 4,
        req_ctx: RequestContext | None = None,
    ) -> dict[str, Any] | None:
        """兼容层：联动引用视频理解插件或 OneBot 历史解析被引用视频"""
        return await self.resolve_video_source(
            event=event,
            video_source=video_source,
            video_ref=video_ref,
            file_id=file_id,
            reply_message_id=reply_message_id,
            task_dir=task_dir,
            max_video_duration=max_video_duration,
            max_media_mb=max_media_mb,
            num_frames=num_frames,
            req_ctx=req_ctx,
        )

    async def resolve_video_source(
        self,
        event: Any,
        video_source: Any | None = None,
        video_ref: str | None = None,
        file_id: str | None = None,
        reply_message_id: str | int | None = None,
        task_dir: Path | None = None,
        max_video_duration: int = 180,
        max_media_mb: int = 50,
        num_frames: int = 4,
        req_ctx: RequestContext | None = None,
    ) -> dict[str, Any] | None:
        """
        全量视频源解析入口：
        综合支持：
        1. 引用视频插件缓存 (VideoMessageCache & VideoCaptionContextCache)
        2. 未缓存的历史引用视频（通过 OneBot get_msg + get_file 补拉）
        3. 远程直接视频直链（HTTP/HTTPS 流式下载并严格限制字节）
        4. 当前消息直接发送的视频（直传文件或 OneBot file_id）
        5. 本地文件已存在路径
        """
        umo = getattr(event, "unified_msg_origin", "")
        bot = getattr(event, "bot", None)
        task_d = task_dir or Path(tempfile.gettempdir())
        task_d.mkdir(parents=True, exist_ok=True)

        caption_text = ""
        resolved_video_path = None
        frame_paths: list[str] = []
        error_reason = ""

        # 统一处理传入的 video_source (Video 组件 / dict)
        if video_source is not None:
            v_r, f_i = extract_video_info(video_source)
            video_ref = video_ref or v_r
            file_id = file_id or f_i

        # 通道 1：如果指定了 reply_message_id，先尝试联动引用视频插件
        if reply_message_id:
            mid = str(reply_message_id).strip()
            video_plugin = self.get_video_vision_plugin_instance()
            if video_plugin:
                # 读转述缓存
                try:
                    c_cache = getattr(video_plugin, "caption_context_cache", None)
                    if c_cache and hasattr(c_cache, "_items"):
                        item = c_cache._items.get((umo, mid))
                        if item and hasattr(item, "caption_text"):
                            caption_text = item.caption_text
                except Exception:
                    pass

                # 读视频条目
                try:
                    v_cache = getattr(video_plugin, "video_cache", None)
                    if v_cache and hasattr(v_cache, "get"):
                        cached_msg = v_cache.get(session_id=umo, message_id=mid)
                        if cached_msg and hasattr(cached_msg, "videos") and cached_msg.videos:
                            entry = cached_msg.videos[0]
                            v_r, f_i = extract_video_info(entry)
                            video_ref = video_ref or v_r
                            file_id = file_id or f_i
                except Exception:
                    pass

            # 若引用视频未在插件缓存中（未缓存/较早历史），尝试 OneBot get_msg 补拉
            if not video_ref and not file_id and bot and hasattr(bot, "call_action"):
                try:
                    target_mid = int(mid) if mid.isdigit() else mid
                    raw_resp = await bot.call_action("get_msg", message_id=target_mid)
                    if isinstance(raw_resp, dict):
                        data = raw_resp.get("data") if isinstance(raw_resp.get("data"), dict) else raw_resp
                        msg_chain = data.get("message")
                        if isinstance(msg_chain, list):
                            for seg in msg_chain:
                                if is_video_segment(seg):
                                    v_r, f_i = extract_video_info(seg)
                                    video_ref = video_ref or v_r
                                    file_id = file_id or f_i
                                    break
                except Exception as e:
                    logger.debug(f"[笑点解析] OneBot get_msg 补拉未缓存视频异常: {e}")

        # 核心判定：只有实际检测到视频源（链接/file_id/转述缓存），才作为视频解析
        has_video_source = bool(video_ref or file_id or caption_text)
        if not has_video_source:
            return None

        # 通道 2：解析 video_ref 或 file_id 落地为真实视频文件
        # 2.1 本地现有文件
        if video_ref and os.path.isfile(video_ref):
            sz_mb = os.path.getsize(video_ref) / (1024 * 1024)
            if sz_mb > max_media_mb:
                error_reason = f"视频大小 ({sz_mb:.1f}MB) 超过系统上限 {max_media_mb}MB"
                logger.warning(f"[笑点解析] {error_reason}")
            else:
                resolved_video_path = video_ref

        # 2.2 通过 OneBot get_file 拉取直传文件
        if not resolved_video_path and file_id and bot and hasattr(bot, "call_action"):
            try:
                res = await bot.call_action("get_file", file=file_id)
                if isinstance(res, dict):
                    data = res.get("data") if isinstance(res.get("data"), dict) else res
                    loc_ref, _ = extract_video_info(data)
                    if loc_ref:
                        if loc_ref.startswith(("http://", "https://")):
                            video_ref = loc_ref
                        elif os.path.isfile(loc_ref):
                            sz_mb = os.path.getsize(loc_ref) / (1024 * 1024)
                            if sz_mb > max_media_mb:
                                error_reason = f"视频大小 ({sz_mb:.1f}MB) 超过上限 {max_media_mb}MB"
                            else:
                                resolved_video_path = loc_ref
            except Exception as e:
                logger.debug(f"[笑点解析] OneBot get_file 解析失败: {e}")
                if not error_reason:
                    error_reason = f"通过协议端拉取视频失败: {e}"

        # 2.3 远程直链视频下载（带字节上限防护与专属任务目录隔离）
        if not resolved_video_path and video_ref and video_ref.startswith(("http://", "https://")):
            dest_video_file = task_d / f"remote_{uuid.uuid4().hex[:8]}.mp4"
            max_bytes = max_media_mb * 1024 * 1024
            dl_coro = download_file_with_limit(
                url=video_ref,
                output_path=dest_video_file,
                max_bytes=max_bytes,
                timeout=18.0,
                req_ctx=req_ctx,
            )
            dl_task = asyncio.create_task(dl_coro)
            if req_ctx:
                req_ctx.track_task(dl_task)
            ok = await dl_task
            if ok and dest_video_file.exists():
                resolved_video_path = str(dest_video_file)
                if req_ctx:
                    req_ctx.add_temp_file(dest_video_file)
            else:
                if not error_reason:
                    error_reason = f"视频下载失败或体积超过 {max_media_mb}MB 上限"

        # 通道 3：抽帧（专属任务目录输出）
        if resolved_video_path and os.path.isfile(resolved_video_path):
            try:
                frame_coro = sample_video_frames(
                    video_path=resolved_video_path,
                    output_dir=task_d,
                    num_frames=num_frames,
                    max_duration=max_video_duration,
                    max_media_mb=max_media_mb,
                    req_ctx=req_ctx,
                )
                frame_task = asyncio.create_task(frame_coro)
                if req_ctx:
                    req_ctx.track_task(frame_task)
                frames = await frame_task
                if frames:
                    frame_paths = frames
                    if req_ctx:
                        for f in frames:
                            req_ctx.add_temp_file(f)
                elif not error_reason:
                    dur = await get_video_duration(resolved_video_path)
                    if dur and dur > max_video_duration:
                        error_reason = f"视频时长 ({dur:.1f}秒) 超过系统上限 {max_video_duration}秒"
                    else:
                        error_reason = f"未能从视频中成功抽取画面（时长可能超过 {max_video_duration} 秒或格式不支持）"
            except Exception as e:
                logger.warning(f"[笑点解析] 视频抽帧失败: {e}")
                if not error_reason:
                    error_reason = f"视频抽帧异常: {e}"

        if not frame_paths and not caption_text and not error_reason:
            error_reason = "未能获取到可供分析的视频画面或转述内容"

        return {
            "has_video": True,
            "caption": caption_text,
            "video_path": resolved_video_path,
            "frames": frame_paths,
            "url": video_ref or "",
            "error_reason": error_reason,
            "cover_path": None,
        }

