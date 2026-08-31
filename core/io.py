# Comfy-SSE low-level IO. All functions work on numpy float32 arrays,
# shape (H, W, 3), RGB, values as stored in the file (no colour conversion
# here — that's sse_color's job).
import os
import subprocess

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

FLOAT_EXTS = {".exr"}  # stored linear float


def _require(mod, name, what):
    if mod is None:
        raise RuntimeError(f"{name} is required to {what}. Install it in the ComfyUI environment.")


def _read_exr_openexr(path):
    """Read an EXR via the OpenEXR python package (pip install OpenEXR)."""
    import OpenEXR
    import Imath
    f = OpenEXR.InputFile(path)
    hdr = f.header()
    dw = hdr["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    names = list(hdr["channels"].keys())

    def chan(c):
        return np.frombuffer(f.channel(c, pt), np.float32).reshape(h, w)

    def pick(want):
        for n in names:
            if n == want or n.endswith("." + want):
                return n
        return None

    r, g, b = pick("R"), pick("G"), pick("B")
    if r and g and b:
        rgb = np.stack([chan(r), chan(g), chan(b)], axis=-1)
    else:
        first = chan(names[0])
        rgb = np.stack([first] * 3, axis=-1)
    a = pick("A")
    alpha = chan(a) if a else None
    return np.ascontiguousarray(rgb, np.float32), alpha


def read_frame(path):
    """Read one image file -> float32 (H, W, 3) RGB, plus alpha (H, W) or None."""
    ext = os.path.splitext(path)[1].lower()
    if ext in FLOAT_EXTS or ext in (".tif", ".tiff", ".dpx"):
        raw = None
        if cv2 is not None:
            raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None and ext in FLOAT_EXTS:
            try:
                return _read_exr_openexr(path)
            except ImportError:
                raise RuntimeError(
                    f"Could not read {path}: this OpenCV build lacks EXR support. "
                    "Run: pip install OpenEXR"
                )
        if raw is None:
            raise ValueError(f"Could not read image: {path}")
        raw = raw.astype(np.float32)
        if raw.ndim == 2:
            raw = raw[..., None].repeat(3, axis=2)
        alpha = raw[..., 3] if raw.shape[2] >= 4 else None
        rgb = raw[..., :3][..., ::-1].copy()  # BGR -> RGB
        if ext not in FLOAT_EXTS and rgb.max() > 1.5:
            # integer-encoded tif/dpx read as 8/16-bit values
            scale = 65535.0 if rgb.max() > 255.0 else 255.0
            rgb = rgb / scale
            if alpha is not None:
                alpha = alpha / scale
        return rgb, alpha
    _require(Image, "Pillow", f"read {ext} files")
    im = Image.open(path)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA" if "A" in im.mode else "RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0
    alpha = arr[..., 3] if arr.shape[-1] == 4 else None
    return arr[..., :3], alpha


def read_video(path, start_frame=0, frame_limit=0, every_nth=1):
    """Read a video -> (list of float32 RGB frames, fps)."""
    _require(cv2, "opencv-python", "read video files")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    idx = 0
    kept = 0
    start = max(0, int(start_frame))
    nth = max(1, int(every_nth))
    limit = max(0, int(frame_limit))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= start and (idx - start) % nth == 0:
            frames.append(frame[..., ::-1].astype(np.float32) / 255.0)
            kept += 1
            if limit and kept >= limit:
                break
        idx += 1
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from video (start_frame past the end?): {path}")
    return frames, float(fps)


def _write_exr_raw(path, rgb):
    """Dependency-free EXR writer: single-part, scanline, float32, uncompressed."""
    import struct
    rgb = np.ascontiguousarray(rgb[..., :3], np.float32)
    h, w = rgb.shape[:2]

    def attr(name, typ, data):
        return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(data)) + data

    # channels sorted alphabetically (B, G, R); FLOAT=2
    ch = b""
    for c in ("B", "G", "R"):
        ch += c.encode() + b"\0" + struct.pack("<i", 2) + struct.pack("<B", 0) + b"\0\0\0" + struct.pack("<ii", 1, 1)
    ch += b"\0"
    box = struct.pack("<iiii", 0, 0, w - 1, h - 1)
    header = b""
    header += attr("channels", "chlist", ch)
    header += attr("compression", "compression", struct.pack("<B", 0))
    header += attr("dataWindow", "box2i", box)
    header += attr("displayWindow", "box2i", box)
    header += attr("lineOrder", "lineOrder", struct.pack("<B", 0))
    header += attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    header += attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
    header += attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    header += b"\0"

    magic = struct.pack("<i", 20000630) + struct.pack("<i", 2)
    table_pos = len(magic) + len(header)
    data_start = table_pos + 8 * h
    line_bytes = w * 3 * 4
    chunk_bytes = 8 + line_bytes

    with open(path, "wb") as f:
        f.write(magic)
        f.write(header)
        for y in range(h):
            f.write(struct.pack("<Q", data_start + y * chunk_bytes))
        # scanlines: y, size, then per-channel rows in B, G, R order
        b_ = rgb[..., 2]
        g_ = rgb[..., 1]
        r_ = rgb[..., 0]
        for y in range(h):
            f.write(struct.pack("<ii", y, line_bytes))
            f.write(b_[y].tobytes())
            f.write(g_[y].tobytes())
            f.write(r_[y].tobytes())


def write_exr(path, rgb):
    rgb = rgb[..., :3].astype(np.float32)
    if cv2 is not None:
        try:
            if cv2.imwrite(path, rgb[..., ::-1]):
                return
        except Exception:
            pass
    try:
        import OpenEXR
        import Imath
        h, w = rgb.shape[:2]
        hdr = OpenEXR.Header(w, h)
        ftype = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        hdr["channels"] = {"R": ftype, "G": ftype, "B": ftype}
        out = OpenEXR.OutputFile(path, hdr)
        out.writePixels({
            "R": np.ascontiguousarray(rgb[..., 0]).tobytes(),
            "G": np.ascontiguousarray(rgb[..., 1]).tobytes(),
            "B": np.ascontiguousarray(rgb[..., 2]).tobytes(),
        })
        out.close()
        return
    except ImportError:
        pass
    _write_exr_raw(path, rgb)  # last resort: built-in uncompressed writer


def write_png(path, rgb):
    _require(cv2, "opencv-python", "write PNG files")
    data = np.clip(rgb, 0.0, 1.0)
    ok = cv2.imwrite(path, (data[..., ::-1] * 65535.0 + 0.5).astype(np.uint16))
    if not ok:
        raise ValueError(f"Could not write PNG: {path}")


def write_h264(path, frames, fps, ffmpeg, crf=18, max_width=0):
    """Write display-encoded (0-1) RGB frames to an H.264 MP4 via ffmpeg."""
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found — needed to write MP4. Add it to PATH.")
    h, w = frames[0].shape[:2]
    out_w, out_h = w, h
    if max_width and w > max_width:
        out_w = max_width
        out_h = int(round(h * max_width / w / 2)) * 2
    out_w -= out_w % 2
    out_h -= out_h % 2
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}",
        "-i", "-",
        "-vf", f"scale={out_w}:{out_h}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for f in frames:
            proc.stdin.write((np.clip(f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        if proc.wait() != 0:
            raise ValueError(f"ffmpeg failed: {err[:400]}")
    finally:
        if proc.poll() is None:
            proc.kill()
