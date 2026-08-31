# Comfy-SSE writer. Writes finals to any absolute path AND reports a preview
# into ComfyUI history so frontends (Nautilus, the Comfy UI) can show results.
import os
import re

import numpy as np

from ..core.color import COLORSPACES, convert
from ..core.io import write_exr, write_h264, write_png
from ..core.paths import find_ffmpeg

FORMATS = [
    "EXR Sequence (ACEScg)",
    "EXR Sequence (Linear Rec.709)",
    "PNG Sequence (sRGB)",
    "MP4 H.264 (Rec.709)",
]


def _target_space(file_format):
    return {
        "EXR Sequence (ACEScg)": "ACEScg",
        "EXR Sequence (Linear Rec.709)": "Linear_Rec709",
        "PNG Sequence (sRGB)": "sRGB_Display",
        "MP4 H.264 (Rec.709)": "Rec709_Display",
    }[file_format]


def _unique_prefix(prefix):
    """If files for this prefix already exist, version up: prefix, prefix_v002…"""
    directory = os.path.dirname(prefix) or "."
    base = os.path.basename(prefix)
    try:
        names = os.listdir(directory)
    except OSError:
        return prefix
    if not any(n.startswith(base + ".") or n.startswith(base + "_") or n == base + ".mp4" for n in names):
        return prefix
    v = 2
    while True:
        cand = f"{base}_v{v:03d}"
        if not any(n.startswith(cand) for n in names):
            return os.path.join(directory, cand)
        v += 1


class SSE_WriteClip:
    CATEGORY = "SSE/IO"
    FUNCTION = "write"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("written_path",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_path": ("STRING", {"default": "", "tooltip": "Full path prefix, absolute paths welcome — e.g. O:/shot/comp/BIB_051_0010_matte"}),
                "file_format": (FORMATS, {"default": "MP4 H.264 (Rec.709)"}),
                "input_colorspace": (COLORSPACES, {"default": "sRGB_Display", "tooltip": "Colourspace of the images coming from the graph"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}),
                "start_number": ("INT", {"default": 1001, "min": 0, "max": 2 ** 31, "tooltip": "First frame number for sequences"}),
                "version_up": ("BOOLEAN", {"default": True, "tooltip": "Never overwrite: version up if files exist"}),
                "save_preview": ("BOOLEAN", {"default": True, "tooltip": "Also write an H.264 preview into the ComfyUI output folder so the result shows in the UI"}),
            },
        }

    def write(self, images, output_path, file_format, input_colorspace, fps, start_number, version_up, save_preview):
        prefix = str(output_path or "").strip().strip('"')
        if not prefix:
            raise ValueError("output_path is empty")
        prefix = os.path.normpath(prefix)
        # strip an accidental extension or frame pattern from the prefix
        prefix = re.sub(r"(\.(#+|%0?\d*d))?\.(exr|png|mp4|mov)$", "", prefix, flags=re.I)
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        if version_up:
            prefix = _unique_prefix(prefix)

        frames = images.cpu().numpy().astype(np.float32)
        target = _target_space(file_format)
        written = []

        if file_format.startswith("MP4"):
            display = [convert(f, input_colorspace, target) for f in frames]
            out_file = prefix + ".mp4"
            write_h264(out_file, display, fps, find_ffmpeg())
            written.append(out_file)
        else:
            enc = [convert(f, input_colorspace, target) for f in frames]
            ext = ".exr" if file_format.startswith("EXR") else ".png"
            for i, f in enumerate(enc):
                out_file = f"{prefix}.{start_number + i:04d}{ext}"
                if ext == ".exr":
                    write_exr(out_file, f)
                else:
                    write_png(out_file, f)
                written.append(out_file)

        ui = {}
        if save_preview:
            try:
                ui = self._write_preview(frames, input_colorspace, fps, os.path.basename(prefix))
            except Exception as err:  # preview must never fail the render
                print(f"[Comfy-SSE] preview failed: {err}")
        summary = written[0] if len(written) == 1 else f"{written[0]}  (+{len(written) - 1} more)"
        return {"ui": ui, "result": (summary,)}

    def _write_preview(self, frames, input_colorspace, fps, base_name):
        import folder_paths
        out_dir = folder_paths.get_output_directory()
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", base_name) or "sse"
        display = [convert(f, input_colorspace, "Rec709_Display") for f in frames]

        if len(display) == 1:
            name = self._unique_name(out_dir, safe, ".png")
            write_png(os.path.join(out_dir, name), display[0])
            return {"images": [{"filename": name, "subfolder": "", "type": "output"}]}

        name = self._unique_name(out_dir, safe, ".mp4")
        write_h264(os.path.join(out_dir, name), display, fps, find_ffmpeg(), crf=23, max_width=1280)
        return {"gifs": [{
            "filename": name, "subfolder": "", "type": "output",
            "format": "video/h264-mp4", "frame_rate": float(fps),
        }]}

    @staticmethod
    def _unique_name(directory, base, ext):
        n = 0
        while True:
            name = f"{base}_preview{'' if n == 0 else '_' + str(n)}{ext}"
            if not os.path.exists(os.path.join(directory, name)):
                return name
            n += 1


NODE_CLASS_MAPPINGS = {
    "SSE_WriteClip": SSE_WriteClip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SSE_WriteClip": "SSE Write Clip",
}
