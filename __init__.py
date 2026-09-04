"""What ComfyUI loads. The pack itself is `h3refiner/`.

Two mappings and nothing else: the node ids saved workflows name, and the names
the search menu shows. There is no `WEB_DIRECTORY` — the refiner upstream was a
panel on a large custom node with a frontend of its own, and this is three
ordinary nodes with ordinary widgets, so there is no javascript to serve.
"""

from .h3refiner import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
