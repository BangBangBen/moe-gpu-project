"""Deterministic packed synthetic workloads; routing itself is not computed."""

import numpy as np

from .format import Workload, dimensions


def generate(t=37, e=5, k=13, n=7, *, routing="balanced", seed=0, counts=None):
    """Create packed FP32 inputs for T tokens, E experts, and a K-to-N GEMM.

    A fixed seed reproduces the arrays. Explicit counts override routing;
    otherwise routing selects balanced, uniform-random, or hotspot loads.
    """
    dimensions(t, e, k, n)
    rng = np.random.default_rng(seed)
    if counts is not None:
        counts = np.asarray(counts)
        if counts.shape != (e,) or counts.dtype.kind not in "iu" or np.any(counts < 0) or sum(map(int, counts)) != t:
            raise ValueError("counts must be E nonnegative integers summing to T")
        counts = counts.astype(np.int64)
    elif routing == "balanced":
        counts = np.full(e, t // e, dtype=np.int64)
        counts[: t % e] += 1
    elif routing == "uniform":
        counts = np.bincount(rng.integers(e, size=t), minlength=e)
    elif routing == "hotspot":
        # Reserve half the tokens (rounded up) for expert zero.
        counts = np.zeros(e, dtype=np.int64)
        counts[0] = t if e == 1 else (t + 1) // 2
        if e > 1:
            counts[1:] = np.bincount(rng.integers(e - 1, size=t - counts[0]), minlength=e - 1)
    else:
        raise ValueError(f"unknown routing: {routing}")
    # Prefix sums turn token counts into each expert's [start, end) row range.
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
    x = rng.standard_normal((t, k), dtype=np.float32)
    # Scale weights to keep output magnitudes similar as K changes.
    w = rng.standard_normal((e, k, n), dtype=np.float32) * np.float32(1 / np.sqrt(k))
    result = Workload(x, w, offsets)
    result.validate()
    return result
