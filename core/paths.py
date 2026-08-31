# Comfy-SSE path resolution: accept whatever an artist (or Nautilus) throws at
# source_path and turn it into an ordered frame list or a video file.
#
# Accepted forms:
#   O:/shot/plate.mov                          -> video
#   O:/shot/exr/                               -> largest frame sequence inside
#   O:/shot/exr/plate.1001.exr                 -> whole sequence it belongs to
#   O:/shot/exr/plate.####.exr                 -> hash pattern
#   O:/shot/exr/plate.%04d.exr                 -> printf pattern
import os
import re
import shutil

VIDEO_EXTS = {".mov", ".mp4", ".mkv", ".avi", ".webm", ".mxf", ".m4v"}
IMAGE_EXTS = {".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dpx", ".webp", ".bmp"}

_FRAME_RE = re.compile(r"^(.*?)([._])(\d{2,})(\.[^.]+)$")


def _scan_sequences(directory):
    """Group frame-numbered files per (prefix, ext). Returns dict key->list[(frame, path)]."""
    groups = {}
    try:
        names = os.listdir(directory)
    except OSError:
        return groups
    for name in names:
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMAGE_EXTS:
            continue
        m = _FRAME_RE.match(name)
        if not m:
            continue
        key = (m.group(1), m.group(2), ext)
        groups.setdefault(key, []).append((int(m.group(3)), os.path.join(directory, name)))
    for v in groups.values():
        v.sort(key=lambda t: t[0])
    return groups


def resolve_source(source_path, expand_sequence=True):
    """Returns {"kind": "video", "path": p} or {"kind": "frames", "files": [p...]}.

    Raises ValueError with an artist-readable message when nothing matches.
    """
    p = str(source_path or "").strip().strip('"')
    if not p:
        raise ValueError("source_path is empty")
    p = os.path.normpath(p)

    # printf / hash patterns
    m = re.match(r"^(.*?)(#+|%0?(\d+)d)(\.[^.]+)$", os.path.basename(p))
    if m and (("#" in m.group(2)) or ("%" in m.group(2))):
        directory = os.path.dirname(p)
        prefix = m.group(1)
        ext = m.group(4).lower()
        groups = _scan_sequences(directory)
        for (g_prefix, g_sep, g_ext), frames in groups.items():
            if (g_prefix + g_sep) == prefix and g_ext == ext and frames:
                return {"kind": "frames", "files": [f for _, f in frames]}
        # prefix may omit the separator
        for (g_prefix, g_sep, g_ext), frames in groups.items():
            if g_prefix == prefix.rstrip("._") and g_ext == ext and frames:
                return {"kind": "frames", "files": [f for _, f in frames]}
        raise ValueError(f"No frames matching pattern: {source_path}")

    if os.path.isdir(p):
        groups = _scan_sequences(p)
        if groups:
            best = max(groups.values(), key=len)
            return {"kind": "frames", "files": [f for _, f in best]}
        # loose images (unnumbered)
        loose = sorted(
            os.path.join(p, n) for n in os.listdir(p)
            if os.path.splitext(n)[1].lower() in IMAGE_EXTS
        )
        if loose:
            return {"kind": "frames", "files": loose}
        raise ValueError(f"No image sequence found in folder: {source_path}")

    if os.path.isfile(p):
        ext = os.path.splitext(p)[1].lower()
        if ext in VIDEO_EXTS:
            return {"kind": "video", "path": p}
        if ext in IMAGE_EXTS:
            if expand_sequence:
                m2 = _FRAME_RE.match(os.path.basename(p))
                if m2:
                    groups = _scan_sequences(os.path.dirname(p))
                    key = (m2.group(1), m2.group(2), ext)
                    frames = groups.get(key)
                    if frames and len(frames) > 1:
                        return {"kind": "frames", "files": [f for _, f in frames]}
            return {"kind": "frames", "files": [p]}
        raise ValueError(f"Unsupported file type: {source_path}")

    raise ValueError(f"Path not found: {source_path}")


def apply_range(files, start_frame=0, frame_limit=0, every_nth=1):
    """0-based start, limit 0 = all, nth >= 1. Mirrors the Nautilus range fields."""
    start = max(0, int(start_frame))
    nth = max(1, int(every_nth))
    limit = max(0, int(frame_limit))
    out = files[start::nth]
    if limit:
        out = out[:limit]
    return out


def find_ffmpeg():
    """ffmpeg executable, or None. PATH first, then imageio-ffmpeg if present."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
