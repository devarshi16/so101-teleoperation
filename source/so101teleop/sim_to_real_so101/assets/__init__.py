"""Resolve simulation assets from the pinned upstream workshop submodule."""

from __future__ import annotations

import os
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_ASSETS_ROOT = (
    _REPOSITORY_ROOT
    / "third_party"
    / "Sim-to-Real-SO-101-Workshop"
    / "source"
    / "sim_to_real_so101"
    / "assets"
)

# An override is useful for containers that mount the upstream assets elsewhere.
ASSETS_ROOT = Path(os.environ.get("SO101_ASSETS_ROOT", _DEFAULT_ASSETS_ROOT)).expanduser().resolve()


def require_assets() -> Path:
    """Return the asset root or explain how to initialize it."""
    if not (ASSETS_ROOT / "usd" / "SO-ARM101-USD.usd").is_file():
        raise FileNotFoundError(
            f"SO-101 assets were not found at {ASSETS_ROOT}. "
            "Run `git submodule update --init --recursive` from the repository root, "
            "or set SO101_ASSETS_ROOT to the upstream asset directory."
        )
    return ASSETS_ROOT

