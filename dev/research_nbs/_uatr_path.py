"""Add the UATR repository root to ``sys.path`` so ``import dev.*`` works from notebooks.

Keep this file at ``dev/research_nbs/_uatr_path.py`` relative to the repo root. The
location is used to infer the root; do not move it without updating the path math below.
"""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (the directory that contains the ``dev/`` package)."""
    here = Path(__file__).resolve()
    root = here.parent.parent.parent
    marker = root / "dev" / "utils.py"
    if not marker.is_file():
        raise RuntimeError(
            f"Inferred repo root {root} from {here}, but {marker} is not a file."
        )
    return root


def ensure_repo_on_sys_path() -> Path:
    """Prepend the repo root to ``sys.path`` if needed. Safe to call more than once."""
    root = repo_root()
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root
