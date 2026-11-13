#!/usr/bin/env python3
"""Isolated launcher for the read-only daily scenario inspector (stdlib; no native model).

    /home/okin/MAPLE/.venv/bin/python -I -B tools/masy_daily.py inspect SCENARIO.csv

Refuses unless run with -I and -B and with no preloaded scientific package, then
puts this tree's `src` first on `sys.path` and hands over to `masy.daily_cli`.
Exit codes: 0 inspected, 2 refused, 1 internal error.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main(argv):
    root = Path(__file__).resolve().parent.parent
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise ValueError('use python -I -B')
    forbidden = ('masy', 'maple', 'maple_syrup', 'numpy', 'numba', 'jax', 'cupy', 'scipy')
    if any(name.partition('.')[0] in forbidden for name in sys.modules):
        raise ValueError('preloaded package/provider')
    sys.path.insert(0, str(root / 'src'))
    from masy.daily_cli import main as cli_main
    return cli_main(argv)


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as error:  # noqa: BLE001 - launcher boundary
        print(type(error).__name__ + ': ' + str(error)[:4000], file=sys.stderr)
        sys.exit(1)
