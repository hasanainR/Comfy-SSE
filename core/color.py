# Comfy-SSE colour: fixed, dependency-free conversions between the pipeline
# colourspaces. All conversions route through linear Rec.709 primaries.
#
# ENUM NAMES ARE FROZEN — frontends (Nautilus) bind to these strings.
import numpy as np

COLORSPACES = [
    "ACEScg",           # linear, AP1 primaries (pipeline working space)
    "Linear_Rec709",    # linear, Rec.709/sRGB primaries
    "sRGB_Display",     # sRGB-encoded display-referred (what AI models expect)
    "Rec709_Display",   # BT.709 OETF-encoded display-referred
    "Raw_Passthrough",  # no conversion
]

# ACEScg (AP1) <-> linear Rec.709, D65, Bradford-adapted
AP1_TO_709 = np.array([
    [1.70505, -0.62179, -0.08326],
    [-0.13026, 1.14080, -0.01055],
    [-0.02400, -0.12897, 1.15297],
], dtype=np.float64)

M709_TO_AP1 = np.array([
    [0.61313, 0.33953, 0.04734],
    [0.07012, 0.91639, 0.01345],
    [0.02058, 0.10957, 0.86985],
], dtype=np.float64)


def _matmul(img, m):
    # img: (..., 3+) float array; matrix applies to the first 3 channels
    out = img.copy()
    rgb = img[..., :3]
    out[..., :3] = rgb @ m.T.astype(img.dtype)
    return out


def _srgb_encode(x):
    x = np.clip(x, 0.0, None)
    a = 0.055
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * np.power(np.clip(x, 1e-12, None), 1 / 2.4) - a)


def _srgb_decode(x):
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, np.power(np.clip((x + a) / (1 + a), 1e-12, None), 2.4))


def _rec709_oetf(x):
    x = np.clip(x, 0.0, None)
    return np.where(x < 0.018, x * 4.5, 1.099 * np.power(np.clip(x, 1e-12, None), 0.45) - 0.099)


def _rec709_eotf(x):
    return np.where(x < 0.081, x / 4.5, np.power(np.clip((x + 0.099) / 1.099, 1e-12, None), 1 / 0.45))


def _apply_rgb(img, fn):
    out = img.copy()
    out[..., :3] = fn(img[..., :3])
    return out


def to_linear709(img, colorspace):
    """Decode an image (float array, first 3 channels RGB) into linear Rec.709."""
    if colorspace in ("Linear_Rec709", "Raw_Passthrough"):
        return img
    if colorspace == "ACEScg":
        return _matmul(img, AP1_TO_709)
    if colorspace == "sRGB_Display":
        return _apply_rgb(img, _srgb_decode)
    if colorspace == "Rec709_Display":
        return _apply_rgb(img, _rec709_eotf)
    raise ValueError(f"Unknown colorspace: {colorspace}")


def from_linear709(img, colorspace):
    """Encode a linear Rec.709 image into the target colourspace."""
    if colorspace in ("Linear_Rec709", "Raw_Passthrough"):
        return img
    if colorspace == "ACEScg":
        out = _matmul(img, M709_TO_AP1)
        out[..., :3] = np.clip(out[..., :3], 0.0, None)
        return out
    if colorspace == "sRGB_Display":
        return _apply_rgb(img, lambda x: np.clip(_srgb_encode(x), 0.0, 1.0))
    if colorspace == "Rec709_Display":
        return _apply_rgb(img, lambda x: np.clip(_rec709_oetf(x), 0.0, 1.0))
    raise ValueError(f"Unknown colorspace: {colorspace}")


def convert(img, src, dst):
    """Convert float image array between any two pipeline colourspaces."""
    if src == dst or src == "Raw_Passthrough" or dst == "Raw_Passthrough":
        return img
    return from_linear709(to_linear709(img, src), dst)
