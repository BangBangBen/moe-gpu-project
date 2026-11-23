"""FP32 PyTorch expert loop and independent small CPU FP64 reference."""

import numpy as np
import torch

from .format import Workload


def run_experts(x, w, y, offsets) -> None:
    """Write each active expert's GEMM into its slice of a preallocated output."""
    for expert, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        if start != end:
            torch.mm(x[start:end], w[expert], out=y[start:end])


def reference(workload: Workload, device="cuda") -> np.ndarray:
    """Compute each expert's FP32 GEMM and return packed Y[T,N] on the CPU."""
    workload.validate()
    # Restore process state so importing/using the reference has no lasting effect.
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode():
            x = torch.tensor(workload.x, device=device)
            w = torch.tensor(workload.w, device=device)
            y = torch.empty((x.shape[0], w.shape[2]), dtype=torch.float32, device=device)
            run_experts(x, w, y, workload.offsets)
            return y.cpu().numpy()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def reference_fp64(workload: Workload) -> np.ndarray:
    """Compute a small CPU FP64 cross-check using one token at a time."""
    workload.validate()
    x, w = workload.x.astype(np.float64), workload.w.astype(np.float64)
    y = np.empty((x.shape[0], w.shape[2]), dtype=np.float64)
    # Per-token expert lookup avoids reproducing the packed expert slicing loop.
    for token in range(x.shape[0]):
        # Skip repeated boundaries from empty experts to find the token's owner.
        expert = np.searchsorted(workload.offsets[1:], token, side="right")
        y[token] = x[token] @ w[expert]
    return y
