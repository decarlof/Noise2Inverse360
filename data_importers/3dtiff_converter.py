#!/usr/bin/env python3
"""Split 3D TIFF stacks into per-slice directories for Noise2Inverse360.

Usage:
    python 3dtiff_converter.py <input_dir> [output_dir]

    input_dir  : directory containing delta_even.tiff, delta_odd.tiff, delta_all.tiff
    output_dir : where to write per-slice subdirectories (defaults to input_dir)
"""
import sys
import tifffile
from pathlib import Path

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

base = Path(sys.argv[1])
out_base = Path(sys.argv[2]) if len(sys.argv) > 2 else base
out_base.mkdir(parents=True, exist_ok=True)

for fname, outdir in [
    ("beta_even.tiff", "beta_even"),
    ("beta_odd.tiff",  "beta_odd"),
    ("beta_all.tiff",  "beta_all"),
]:
    vol = tifffile.imread(base / fname)  # [Z, H, W]
    out = out_base / outdir
    out.mkdir(exist_ok=True)
    print(f"{fname}: {vol.shape} -> {out}")
    for i, sl in enumerate(vol):
        tifffile.imwrite(out / f"{i:05d}.tiff", sl)
