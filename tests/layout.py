"""Import the pack from a directory ComfyUI clones under whatever name it likes.

`Comfy-H3-PromptRefiner` is not an identifier, so `import h3refiner` finds
nothing from a checkout, and the hyphen is not something to rename away: it is
the repository name and the folder people end up with under `custom_nodes/`.
So the package is registered by hand, once, under the name its own modules use
for each other — which is also how ComfyUI reaches it, through the root
`__init__.py`'s relative import.

Nothing here needs torch or a ComfyUI. The modules that do (`local`, `remote`'s
credential file, `nodes`) import theirs inside the functions that use them, so a
suite that never calls those never loads them.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "h3refiner"


def load(*names):
    """The named submodules, as attributes of one object.

    The package is registered *without* being executed, because its `__init__`
    imports `nodes` and `nodes` is the one module that reaches for ComfyUI. The
    submodules only ever import each other, so an unexecuted entry with the
    right `__path__` is the whole of what they need to find one another.
    """
    if NAME not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            NAME, ROOT / NAME / "__init__.py",
            submodule_search_locations=[str(ROOT / NAME)])
        sys.modules[NAME] = importlib.util.module_from_spec(spec)

    loaded = types.SimpleNamespace()
    for name in names:
        setattr(loaded, name, importlib.import_module(f"{NAME}.{name}"))
    return loaded
