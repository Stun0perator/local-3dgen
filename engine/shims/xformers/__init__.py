"""Minimal stand-in for the parts of xformers that TRELLIS calls, built on torch SDPA.

Part of engine/shims. Real xformers wheels pin (and downgrade) torch and don't ship sm_120 kernels for RTX 50-series
cards, so the TRELLIS backend puts this package first on sys.path instead.
"""
from . import ops  # noqa: F401
