# Comfy-SSE loaders. Conventions (frozen — Nautilus binds to these):
#   source_path : file / folder / name.####.ext / name.%04d.ext / first frame
#   start_frame : 0-based
#   frame_limit : 0 = all
#   every_nth   : 1 = every frame
import os
import re
import time

import numpy as np
import torch

from ..core.color import COLORSPACES, convert
from ..core.io import read_frame, read_video, write_h264, write_png
from ..core.paths import apply_range, find_ffmpeg, resolve_source

PREVIEW_MAX_FRAMES = 240


def _preview_ui(frames, src_colorspace, fps, tag):
    """Write a display-encoded preview into ComfyUI's temp dir and return the
    ui dict that makes it show on the node. Never raises."""
    try:
        import folder_paths
        tdir = folder_paths.get_temp_directory()
        os.makedirs(tdir, exist_ok=True)
        safe = (re.sub(r"[^a-zA-Z0-9_-]+", "_", tag) or "sse")[:48]
        stamp = format(int(time.time() * 1000) % 0xFFFFFFFF, "x")
        disp = [convert(f, src_colorspace, "Rec709_Display") for f in frames[:PREVIEW_MAX_FRAMES]]
        ffmpeg = find_ffmpeg()
        if len(disp) > 1 and ffmpeg:
            name = f"{safe}_{stamp}.mp4"
            write_h264(os.path.join(tdir, name), disp, fps or 24.0, ffmpeg, crf=28, max_width=640)
            return {"gifs": [{"filename": name, "subfolder": "", "type": "temp",
                              "format": "video/h264-mp4", "frame_rate": float(fps or 24.0)}]}
        name = f"{safe}_{stamp}.png"
        write_png(os.path.join(tdir, name), disp[0])
        return {"images": [{"filename": name, "subfolder": "", "type": "temp"}]}
    except Exception as err:
        print(f"[Comfy-SSE] load preview failed: {err}")
        return {}


def _stack(frames_np):
    ref_h, ref_w = frames_np[0].shape[:2]
    fixed = []
    for f in frames_np:
        if f.shape[:2] != (ref_h, ref_w):
            raise ValueError("Frames in the sequence have mismatched resolutions")
        fixed.append(f)
    return torch.from_numpy(np.stack(fixed, axis=0))


class SSE_LoadClip:
    CATEGORY = "SSE/IO"
    FUNCTION = "load"
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("image", "frame_count", "fps")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_path": ("STRING", {"default": "", "tooltip": "File, folder, name.####.ext, name.%04d.ext or any frame of a sequence"}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 2 ** 31, "tooltip": "0-based frame to start at"}),
                "frame_limit": ("INT", {"default": 0, "min": 0, "max": 2 ** 31, "tooltip": "Max frames to load (0 = all)"}),
                "every_nth": ("INT", {"default": 1, "min": 1, "max": 10000}),
                "input_colorspace": (COLORSPACES, {"default": "ACEScg", "tooltip": "Colourspace of the source files"}),
                "output_colorspace": (COLORSPACES, {"default": "sRGB_Display", "tooltip": "Colourspace handed to the graph (models expect sRGB_Display)"}),
            },
            "optional": {
                "fps_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "tooltip": "0 = source fps (sequences default to 24)"}),
                "show_preview": ("BOOLEAN", {"default": True, "tooltip": "Show a preview of the loaded clip on the node"}),
            },
        }

    def load(self, source_path, start_frame, frame_limit, every_nth, input_colorspace, output_colorspace, fps_override=0.0, show_preview=True):
        src = resolve_source(source_path, expand_sequence=True)
        if src["kind"] == "video":
            frames, fps = read_video(src["path"], start_frame, frame_limit, every_nth)
        else:
            files = apply_range(src["files"], start_frame, frame_limit, every_nth)
            if not files:
                raise ValueError(f"start_frame {start_frame} is past the end of the sequence ({len(src['files'])} frames)")
            frames = [read_frame(f)[0] for f in files]
            fps = 24.0
        if fps_override and fps_override > 0:
            fps = float(fps_override)
        frames = [convert(f, input_colorspace, output_colorspace) for f in frames]
        images = _stack([np.ascontiguousarray(f, dtype=np.float32) for f in frames])
        ui = _preview_ui(frames, output_colorspace, fps, os.path.basename(str(source_path))) if show_preview else {}
        return {"ui": ui, "result": (images, images.shape[0], float(fps))}


class SSE_LoadStill:
    CATEGORY = "SSE/IO"
    FUNCTION = "load"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_path": ("STRING", {"default": "", "tooltip": "A specific image file loads as-is; a folder / #### pattern / video uses frame_index"}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 2 ** 31, "tooltip": "0-based frame to pick when the source is a folder, pattern or video"}),
                "input_colorspace": (COLORSPACES, {"default": "ACEScg"}),
                "output_colorspace": (COLORSPACES, {"default": "sRGB_Display"}),
                "mask_from": (["luminance", "alpha", "none"], {"default": "luminance"}),
            },
            "optional": {
                "show_preview": ("BOOLEAN", {"default": True}),
            },
        }

    def load(self, source_path, frame_index, input_colorspace, output_colorspace, mask_from, show_preview=True):
        # A specific file loads AS-IS (no sequence expansion) — frame_index
        # applies when the source is a folder, a ####/%04d pattern, or a video.
        src = resolve_source(source_path, expand_sequence=False)
        alpha = None
        if src["kind"] == "video":
            frames, _ = read_video(src["path"], start_frame=frame_index, frame_limit=1)
            rgb = frames[0]
        else:
            files = src["files"]
            if frame_index >= len(files):
                raise ValueError(f"frame_index {frame_index} out of range (sequence has {len(files)} frames)")
            rgb, alpha = read_frame(files[frame_index])
        rgb = convert(rgb, input_colorspace, output_colorspace)
        rgb = np.ascontiguousarray(rgb, dtype=np.float32)
        image = torch.from_numpy(rgb[None, ...])

        if mask_from == "alpha" and alpha is not None:
            mask_np = np.clip(alpha, 0.0, 1.0)
        elif mask_from == "luminance":
            lin = rgb[..., :3]
            mask_np = np.clip(0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2], 0.0, 1.0)
        else:
            mask_np = np.zeros(rgb.shape[:2], dtype=np.float32)
        mask = torch.from_numpy(np.ascontiguousarray(mask_np, dtype=np.float32)[None, ...])
        ui = _preview_ui([rgb], output_colorspace, 24.0, os.path.basename(str(source_path))) if show_preview else {}
        return {"ui": ui, "result": (image, mask)}


NODE_CLASS_MAPPINGS = {
    "SSE_LoadClip": SSE_LoadClip,
    "SSE_LoadStill": SSE_LoadStill,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SSE_LoadClip": "SSE Load Clip",
    "SSE_LoadStill": "SSE Load Still",
}
