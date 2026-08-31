# Comfy-SSE colour transform node.
import numpy as np
import torch

from ..core.color import COLORSPACES, convert


class SSE_ColorTransform:
    CATEGORY = "SSE/IO"
    FUNCTION = "transform"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "from_colorspace": (COLORSPACES, {"default": "ACEScg"}),
                "to_colorspace": (COLORSPACES, {"default": "sRGB_Display"}),
            },
        }

    def transform(self, image, from_colorspace, to_colorspace):
        arr = image.cpu().numpy().astype(np.float32)
        out = convert(arr, from_colorspace, to_colorspace)
        return (torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32)),)


NODE_CLASS_MAPPINGS = {
    "SSE_ColorTransform": SSE_ColorTransform,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SSE_ColorTransform": "SSE Color Transform",
}
