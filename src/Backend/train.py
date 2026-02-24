"""Backward-compatibility shim.

All training logic now lives in :pymod:`src.Backend.temporal_gnn`.
This module re-exports ``tensor_make_finite_`` so existing imports
continue to work, but prints a deprecation warning on first use.

.. deprecated:: 2025-06
   Use ``from src.Backend.temporal_gnn import tensor_make_finite_`` instead.
"""

from __future__ import annotations

import warnings as _warnings

from src.Backend.temporal_gnn import tensor_make_finite_  # noqa: F401

_warnings.warn(
    "src.Backend.train is deprecated — import from src.Backend.temporal_gnn instead.",
    DeprecationWarning,
    stacklevel=2,
)
