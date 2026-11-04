#!/usr/bin/env python3
"""Read-only environment and input validation for the campaign."""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the ThermoRDF campaign environment and NetCDF.")
    parser.add_argument("netcdf", type=Path, nargs="?", default=Path("thermordf_master.nc"))
    args = parser.parse_args()
    try:
        versions = {}
        for name in ("numpy", "pandas", "xarray", "netCDF4", "matplotlib", "torch"):
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        import torch
        import thermordf_common as common
        data = common.load_data(args.netcdf)
        print("STATUS: READY")
        print(f"Python: {platform.python_version()}")
        for name, version in versions.items():
            print(f"{name}: {version}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"States: {data.temperature.size}")
        print(f"Channels: {list(common.CHANNELS)}")
        print(f"Radial bins: {data.radial.size}")
        print(f"Grid-derived pressure weight: {data.lambda_pressure_grid:.12g}")
        return 0
    except Exception as exc:
        print("STATUS: FAILED", file=sys.stderr)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
