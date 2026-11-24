#!/usr/bin/env python3
"""Local input generation only; use the existing MAPLE CPU interpreter."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from masy.moab_hybrid_inputs import main
if __name__ == '__main__':
    raise SystemExit(main())
