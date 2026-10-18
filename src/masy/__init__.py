"""MAPLE-SYRUP (MASY) wrapper namespace.

MASY-01 provides a read-only current-source compatibility preflight
(`tools/masy_preflight.py`); MASY-03 adds a locked CPU runtime prepare/verify
(`tools/masy_runtime.py`) that delivers source/runtime identity and native
interfaces only. It contains no transport entry point and never reports a
combined model as ready to run (`transport_ready` is always false).

`__version__` must equal the static `[project].version` in pyproject.toml.
"""

__all__ = ["__version__"]

__version__ = "0.0.3.dev0"
