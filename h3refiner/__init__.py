"""The refiner, as a package.

`nodes.py` is the only module that imports ComfyUI, and it does so inside the
functions that need it. Everything under it — the harness, H3's prompting, the
Context-IR assembly, the two backends' request building — is ordinary data and
runs, and is tested, with neither torch nor a ComfyUI on the path.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
