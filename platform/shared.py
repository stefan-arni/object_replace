"""Shared SD components for the unified platform.

Loaded ONCE at app startup, imported by every tab. Avoids 5GB-per-tab
duplication and ~30s of re-load time when switching tabs.

Usage from a teammate's module:
    from platform.shared import get_components, get_editor
    c = get_components()
    img = my_function(image, c=c, ...)
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sd_components import SDComponents, load_sd  # noqa: E402

_components: SDComponents | None = None
_editor = None


def get_components() -> SDComponents:
    global _components
    if _components is None:
        print("loading SD components (one-time at app startup)...")
        _components = load_sd()
        print(f"  loaded on {_components.device}")
    return _components


def get_editor():
    """Object-replacement Editor (Stefan's module)."""
    global _editor
    if _editor is None:
        from editor import Editor
        _editor = Editor(get_components())
    return _editor
