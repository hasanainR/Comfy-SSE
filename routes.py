# Comfy-SSE preview routes: the Load Clip player asks for clip info and single
# display frames while the artist edits widgets, before anything is queued.
import asyncio
import collections
import threading

import cv2
import numpy as np
from aiohttp import web
from server import PromptServer

from .core.color import convert
from .core.io import read_frame
from .core.paths import resolve_source

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
    def __init__(self, source_path):
        self.src = resolve_source(source_path, expand_sequence=True)
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


def _clip(source_path, fresh=False):
    with _open_lock:
        clip = _open.pop(source_path, None)
        if clip is not None and fresh:
            clip.release()
            clip = None
        if clip is None:
            clip = _Clip(source_path)
        _open[source_path] = clip
        while len(_open) > MAX_OPEN:
            _open.popitem(last=False)[1].release()
    return clip


def _frame_jpeg(source_path, index, input_colorspace, output_colorspace):
    rgb = convert(_clip(source_path).read(index), input_colorspace, output_colorspace)
    bgr = cv2.cvtColor((np.clip(rgb[..., :3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError(f"Could not encode preview of frame {index}")
    return buf.tobytes()


@PromptServer.instance.routes.get("/sse/clip_info")
async def clip_info(request):
    try:
        clip = await asyncio.to_thread(_clip, request.query.get("path", ""), True)
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
        )
    except (ValueError, OSError, IndexError) as err:
        return web.Response(status=404, text=str(err))
    return web.Response(body=jpeg, content_type="image/jpeg", headers={"Cache-Control": "no-store"})
