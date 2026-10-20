# Comfy-SSE server routes, used by web/sse_preview.js:
#   /sse/clip_info, /sse/clip_frame — live previews on the load nodes while
#       the artist edits widgets, before anything is queued ("expand=0" uses
#       SSE Load Still semantics: a specific file is NOT expanded to its
#       sequence; folders / patterns / videos are indexable)
#   /sse/browse — server-side directory listing for the Browse dialog
#   /sse/upload — receives drag-and-dropped files into the ComfyUI input dir
import asyncio
import collections
import os
import re
import string
import threading

import cv2
import numpy as np
from aiohttp import web
from server import PromptServer

from .core.color import convert
from .core.io import read_frame
from .core.paths import IMAGE_EXTS, VIDEO_EXTS, _FRAME_RE, _scan_sequences, resolve_source

PREVIEW_WIDTH = 960
MAX_OPEN = 4  # recently previewed sources kept resolved (and videos kept open)

_open = collections.OrderedDict()
_open_lock = threading.Lock()


def _shrink(img):
    h, w = img.shape[:2]
    if w <= PREVIEW_WIDTH:
        return img
    return cv2.resize(img, (PREVIEW_WIDTH, max(1, round(h * PREVIEW_WIDTH / w))), interpolation=cv2.INTER_AREA)


class _Clip:
    def __init__(self, source_path, expand):
        self.src = resolve_source(source_path, expand_sequence=expand)
        self.lock = threading.Lock()
        self.cap = None
        self.next_index = 0
        if self.src["kind"] == "video":
            self.cap = cv2.VideoCapture(self.src["path"])
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video: {self.src['path']}")
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 24.0)
        else:
            self.frame_count = len(self.src["files"])
            self.fps = 24.0

    def read(self, index):
        """Source frame `index` as float32 RGB, at most PREVIEW_WIDTH wide."""
        if self.cap is None:
            return _shrink(np.ascontiguousarray(read_frame(self.src["files"][index])[0]))
        with self.lock:
            # the player asks for frames in order, so short forward gaps
            # (every_nth) are cheaper to grab through than to seek
            if index < self.next_index or index - self.next_index > 48:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                self.next_index = index
            while self.next_index < index:
                self.cap.grab()
                self.next_index += 1
            ok, frame = self.cap.read()
            self.next_index += 1
        if not ok:
            raise IndexError(f"Could not read frame {index}")
        return cv2.cvtColor(_shrink(frame), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def release(self):
        if self.cap is not None:
            with self.lock:
                self.cap.release()


def _clip(source_path, fresh=False, expand=True):
    key = (bool(expand), source_path)
    with _open_lock:
        clip = _open.pop(key, None)
        if clip is not None and fresh:
            clip.release()
            clip = None
        if clip is None:
            clip = _Clip(source_path, expand)
        _open[key] = clip
        while len(_open) > MAX_OPEN:
            _open.popitem(last=False)[1].release()
    return clip


def _frame_jpeg(source_path, index, input_colorspace, output_colorspace, expand=True):
    rgb = convert(_clip(source_path, expand=expand).read(index), input_colorspace, output_colorspace)
    bgr = cv2.cvtColor((np.clip(rgb[..., :3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError(f"Could not encode preview of frame {index}")
    return buf.tobytes()


def _browse(dir_str):
    """Directory listing for the Browse dialog. Frame sequences collapse into
    one pattern entry (name.####.ext) that both load nodes understand."""
    d = os.path.normpath(str(dir_str or "").strip() or os.path.expanduser("~"))
    if not os.path.isdir(d):
        raise ValueError(f"Not a folder: {dir_str}")

    dirs = []
    files = []
    seq_members = set()
    seqs = _scan_sequences(d)
    seq_entries = []
    for (prefix, sep, ext), frames in sorted(seqs.items()):
        if len(frames) < 2:
            continue
        for _, p in frames:
            seq_members.add(os.path.basename(p))
        m = _FRAME_RE.match(os.path.basename(frames[0][1]))
        pad = len(m.group(3)) if m else 4
        pattern = f"{prefix}{sep}{'#' * pad}{ext}"
        seq_entries.append({
            "name": f"{pattern}  ({len(frames)} frames, {frames[0][0]}-{frames[-1][0]})",
            "path": os.path.join(d, pattern),
            "type": "seq",
        })

    try:
        it = sorted(os.scandir(d), key=lambda e: e.name.lower())
    except OSError as err:
        raise ValueError(str(err))
    for e in it:
        if e.name.startswith("."):
            continue
        try:
            if e.is_dir():
                dirs.append({"name": e.name, "path": e.path, "type": "dir"})
                continue
        except OSError:
            continue
        ext = os.path.splitext(e.name)[1].lower()
        if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
            continue
        if e.name in seq_members:
            continue
        kind = "video" if ext in VIDEO_EXTS else "file"
        files.append({"name": e.name, "path": e.path, "type": kind})

    parent = os.path.dirname(d)
    drives = []
    if os.name == "nt":
        drives = [f"{c}:\\" for c in string.ascii_uppercase if os.path.exists(f"{c}:\\")]
    return {
        "dir": d,
        "parent": parent if parent and parent != d else None,
        "drives": drives,
        "entries": dirs + seq_entries + files,
    }


@PromptServer.instance.routes.get("/sse/clip_info")
async def clip_info(request):
    q = request.query
    expand = q.get("expand", "1") != "0"
    try:
        clip = await asyncio.to_thread(_clip, q.get("path", ""), True, expand)
    except (ValueError, OSError) as err:
        return web.json_response({"error": str(err)}, status=404)
    return web.json_response({"kind": clip.src["kind"], "frame_count": clip.frame_count, "fps": clip.fps})


@PromptServer.instance.routes.get("/sse/clip_frame")
async def clip_frame(request):
    q = request.query
    try:
        jpeg = await asyncio.to_thread(
            _frame_jpeg, q.get("path", ""), int(q.get("index", "0")),
            q.get("input_colorspace", "Raw_Passthrough"), q.get("output_colorspace", "Raw_Passthrough"),
            q.get("expand", "1") != "0",
        )
    except (ValueError, OSError, IndexError) as err:
        return web.Response(status=404, text=str(err))
    return web.Response(body=jpeg, content_type="image/jpeg", headers={"Cache-Control": "no-store"})


@PromptServer.instance.routes.get("/sse/browse")
async def sse_browse(request):
    try:
        listing = await asyncio.to_thread(_browse, request.query.get("dir", ""))
    except ValueError as err:
        return web.json_response({"error": str(err)}, status=404)
    return web.json_response(listing)


@PromptServer.instance.routes.post("/sse/upload")
async def sse_upload(request):
    data = await request.post()
    f = data.get("file")
    if f is None or not getattr(f, "filename", ""):
        return web.json_response({"error": "no file in upload"}, status=400)
    import folder_paths
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", os.path.basename(f.filename)) or "dropped"
    dest_dir = os.path.join(folder_paths.get_input_directory(), "sse_dropped")
    os.makedirs(dest_dir, exist_ok=True)
    base, ext = os.path.splitext(safe)
    dest = os.path.join(dest_dir, safe)
    n = 1
    while os.path.exists(dest):
        dest = os.path.join(dest_dir, f"{base}_{n}{ext}")
        n += 1

    def _write():
        with open(dest, "wb") as out:
            while True:
                chunk = f.file.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)

    await asyncio.to_thread(_write)
    return web.json_response({"path": dest})
