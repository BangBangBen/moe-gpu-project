"""Versioned little-endian interchange shared by Python and C++."""

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np

# '<' fixes little-endian encoding with no padding between header fields.
INPUT_HEADER = struct.Struct("<8sIQQQQ")
OUTPUT_HEADER = struct.Struct("<8sIQQ")
INPUT_MAGIC = b"MOEIN001"
OUTPUT_MAGIC = b"MOEOUT01"
VERSION = 1
INT_MAX = 2**31 - 1


def dimensions(*values: int, allow_zero_first: bool = True) -> None:
    """Check cuBLAS int32 dimension limits; optionally allow the first to be zero."""
    for i, value in enumerate(values):
        if not isinstance(value, (int, np.integer)) or not (
            (0 if i == 0 and allow_zero_first else 1) <= value <= INT_MAX
        ):
            raise ValueError("dimensions must fit int32; only T may be zero")


@dataclass
class Workload:
    """Packed inputs: X[T,K], W[E,K,N], and expert row boundaries offsets[E+1]."""

    x: np.ndarray
    w: np.ndarray
    offsets: np.ndarray

    def validate(self) -> None:
        """Raise ValueError for invalid shapes, offsets, dtypes, or non-finite data."""
        if self.x.ndim != 2 or self.w.ndim != 3:
            raise ValueError("expected X[T,K] and W[E,K,N]")
        t, k = self.x.shape
        e, wk, n = self.w.shape
        dimensions(t, e, k, n)
        if k != wk:
            raise ValueError("X and W disagree on K")
        if self.offsets.shape != (e + 1,) or self.offsets.dtype.kind != "i" or self.offsets.dtype.itemsize != 8:
            raise ValueError("offsets must be int64[E+1]")
        if self.offsets[0] != 0 or self.offsets[-1] != t or np.any(self.offsets[1:] < self.offsets[:-1]):
            raise ValueError("offsets must be nondecreasing, start at 0 and end at T")
        for a in (self.x, self.w):
            if a.dtype.kind != "f" or a.dtype.itemsize != 4 or not a.flags.c_contiguous:
                raise ValueError("X and W must be contiguous FP32")
            if not np.isfinite(a).all():
                raise ValueError("input contains non-finite values")


def _header(file, header: struct.Struct, magic: bytes):
    """Read a header, validate its identity and dimensions, and return its shape."""
    raw = file.read(header.size)
    if len(raw) != header.size:
        raise ValueError("truncated header")
    actual_magic, version, *shape = header.unpack(raw)
    if actual_magic != magic or version != VERSION:
        raise ValueError("unsupported magic or version")
    dimensions(*shape)
    return shape


def _array(file, dtype, count, shape):
    """Read count elements into a writable array with the requested shape."""
    return np.frombuffer(file.read(count * np.dtype(dtype).itemsize), dtype=dtype).copy().reshape(shape)


def write_input(path, workload: Workload) -> None:
    """Validate and save packed inputs in the binary format shared with v1."""
    workload.validate()
    t, k = workload.x.shape
    e, _, n = workload.w.shape
    with Path(path).open("wb") as file:
        file.write(INPUT_HEADER.pack(INPUT_MAGIC, VERSION, t, e, k, n))
        for a, dtype in ((workload.offsets, "<i8"), (workload.x, "<f4"), (workload.w, "<f4")):
            file.write(a.astype(dtype, copy=False).tobytes(order="C"))


def read_input(path) -> Workload:
    """Load a packed workload, rejecting malformed files and invalid tensors."""
    with Path(path).open("rb") as file:
        t, e, k, n = _header(file, INPUT_HEADER, INPUT_MAGIC)
        # Verify the full payload size before reading arrays from the file.
        expected = INPUT_HEADER.size + (e + 1) * 8 + (t * k + e * k * n) * 4
        if Path(path).stat().st_size != expected:
            raise ValueError("input payload length mismatch (truncated or surplus bytes)")
        offsets = _array(file, "<i8", e + 1, (e + 1,))
        x = _array(file, "<f4", t * k, (t, k))
        w = _array(file, "<f4", e * k * n, (e, k, n))
    result = Workload(x, w, offsets)
    result.validate()
    return result


def write_output(path, y: np.ndarray) -> None:
    """Save a finite FP32 output matrix with its versioned shape header."""
    if y.ndim != 2 or y.dtype.kind != "f" or y.dtype.itemsize != 4 or not np.isfinite(y).all():
        raise ValueError("output must be finite FP32[T,N]")
    dimensions(*y.shape)
    with Path(path).open("wb") as file:
        file.write(OUTPUT_HEADER.pack(OUTPUT_MAGIC, VERSION, *y.shape))
        file.write(y.astype("<f4", copy=False).tobytes(order="C"))


def read_output(path) -> np.ndarray:
    """Load Y[T,N], checking the header, exact file size, and finite values."""
    with Path(path).open("rb") as file:
        t, n = _header(file, OUTPUT_HEADER, OUTPUT_MAGIC)
        if Path(path).stat().st_size != OUTPUT_HEADER.size + t * n * 4:
            raise ValueError("output payload length mismatch (truncated or surplus bytes)")
        y = _array(file, "<f4", t * n, (t, n))
    if not np.isfinite(y).all():
        raise ValueError("output contains non-finite values")
    return y
