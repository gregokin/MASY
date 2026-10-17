"""MAPLE-SYRUP (MASY) wrapper namespace.

MASY-01 provides only a read-only current-source compatibility preflight
(`tools/masy_preflight.py`). It contains no transport entry point and never
reports a combined model as ready to run.
"""

__all__ = ["__version__"]

__version__ = "0.0.0.masy01"
