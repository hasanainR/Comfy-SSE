# Comfy-SSE — pipeline IO nodes for Sinking Ship Entertainment.
#
# Node modules live in nodes/ and are discovered automatically: any module
# there that defines NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS is
# registered. Shared code (colour, IO, path resolution) lives in core/.
#
# Contract is frozen: input names, enum strings and 0-based frame conventions
# are depended on by Nautilus. Add new inputs — never rename existing ones.
import importlib
import os
import pkgutil
import traceback

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

from . import nodes as _nodes_pkg  # noqa: E402

for _mod_info in pkgutil.iter_modules(_nodes_pkg.__path__):
    try:
        _mod = importlib.import_module(f".nodes.{_mod_info.name}", __name__)
        NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    except Exception:
        print(f"[Comfy-SSE] failed to load nodes.{_mod_info.name}:")
        traceback.print_exc()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
