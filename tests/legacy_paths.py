"""Where the old repos are, for the regression tests; each such test skips when its file is absent.

``OLD_REPO`` / ``ECHO2`` come from ``tools/legacy/old_repo.py`` (``$R2S_OLD_REPO`` /
``$R2S_OLD_ECHO2``), so ``R2S_OLD_REPO=/nonexistent pytest tests`` runs the suite
without the old checkout.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("r2s_legacy_old_repo", REPO / "tools" / "legacy" / "old_repo.py")
old_repo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old_repo)

OLD_REPO: Path = old_repo.OLD
ECHO2: Path = old_repo.ECHO2


def old_file(*parts: str) -> Path:
    """``OLD_REPO/<parts>``, or skip the calling test when it is not there."""
    path = OLD_REPO.joinpath(*parts)
    if not path.exists():
        pytest.skip(f"old repo file not available: {path}")
    return path
