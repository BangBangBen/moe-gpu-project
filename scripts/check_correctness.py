#!/usr/bin/env python3
"""Check v0/v1 correctness only; no timing or performance measurements."""

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from moe.format import read_input, read_output, write_input, write_output
from moe.reference import reference, reference_fp64
from moe.workloads import generate

CASES = [
    ("balanced", dict(t=37, e=5, k=13, n=7)),
    ("uniform", dict(t=41, e=7, k=17, n=11, routing="uniform")),
    ("hotspot", dict(t=67, e=8, k=31, n=19, routing="hotspot")),
    ("empty_experts", dict(t=9, e=5, k=3, n=8, counts=[0, 4, 0, 5, 0])),
    ("single_active", dict(t=17, e=4, k=9, n=5, counts=[0, 0, 17, 0])),
    ("fewer_tokens", dict(t=3, e=8, k=7, n=13)),
    ("one_expert", dict(t=11, e=1, k=5, n=17, routing="hotspot")),
    ("scalar", dict(t=1, e=1, k=1, n=1)),
    ("zero_tokens", dict(t=0, e=4, k=5, n=3)),
    ("larger_reduction", dict(t=29, e=6, k=257, n=33, routing="hotspot")),
]


def compare(actual, expected, label):
    """Check shape, finite values, and numerical tolerance; return maximum absolute error."""
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise AssertionError(f"{label}: invalid shape or non-finite output")
    error = float(np.max(np.abs(actual.astype(np.float64) - expected))) if actual.size else 0.0
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4, err_msg=label)
    return error


def check_case(binary, directory, name, kwargs, device="cuda"):
    """Generate one fixture, run both versions, and check them against CPU FP64."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    input_path = directory / f"{name}.input.bin"
    v0_path, v1_path = directory / f"{name}.v0.bin", directory / f"{name}.v1.bin"
    write_input(input_path, generate(**kwargs, seed=618))
    workload = read_input(input_path)
    write_output(v0_path, reference(workload, device))
    subprocess.run([str(Path(binary).resolve()), str(input_path), str(v1_path)], check=True)
    v0, v1 = read_output(v0_path), read_output(v1_path)
    fp64 = reference_fp64(workload)
    errors = (compare(v1, v0, "v1 vs v0"), compare(v0, fp64, "v0 vs FP64"), compare(v1, fp64, "v1 vs FP64"))
    print(f"PASS {name}: max_abs_error v1/v0={errors[0]:.3e}, v0/FP64={errors[1]:.3e}, v1/FP64={errors[2]:.3e}")


def main():
    """Run the correctness cases selected by the command-line configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/moe_v1"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/correctness"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error(f"v1 binary not found: {args.binary}; build it with CMake first")
    for name, kwargs in CASES:
        check_case(args.binary, args.artifacts, name, kwargs, args.device)
    print(f"All {len(CASES)} correctness cases passed. No performance measurements collected.")


if __name__ == "__main__":
    main()
