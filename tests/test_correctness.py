import struct
import subprocess

import numpy as np
import pytest
import torch

from check_correctness import CASES, check_case, compare
from moe.format import (INPUT_HEADER, INPUT_MAGIC, INT_MAX, read_input,
                        read_output, write_input, write_output)
from moe.reference import reference, reference_fp64
from moe.workloads import generate


@pytest.mark.parametrize("name,kwargs", CASES, ids=[case[0] for case in CASES])
def test_python_reference_and_roundtrip(tmp_path, name, kwargs):
    """Check binary round trips and the CPU PyTorch reference against FP64."""
    workload = generate(**kwargs, seed=618)
    path = tmp_path / "input.bin"
    write_input(path, workload)
    loaded = read_input(path)
    for original, recovered in ((workload.x, loaded.x), (workload.w, loaded.w), (workload.offsets, loaded.offsets)):
        np.testing.assert_array_equal(original, recovered)
    y = reference(loaded, "cpu")
    compare(y, reference_fp64(loaded), name)
    output = tmp_path / "output.bin"
    write_output(output, y)
    np.testing.assert_array_equal(y, read_output(output))


@pytest.mark.parametrize("name,kwargs", CASES, ids=[case[0] for case in CASES])
def test_v1_correctness(v1_binary, tmp_path, name, kwargs):
    """Check a native v1 result against the PyTorch and FP64 references."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_case(v1_binary, tmp_path, name, kwargs, device)


def test_workload_contract():
    """Check reproducible generation, routing counts, and invalid explicit counts."""
    for routing in ("balanced", "uniform", "hotspot"):
        first = generate(routing=routing, seed=10)
        second = generate(routing=routing, seed=10)
        np.testing.assert_array_equal(first.x, second.x)
        np.testing.assert_array_equal(first.w, second.w)
        np.testing.assert_array_equal(first.offsets, second.offsets)
        counts = np.diff(first.offsets)
        if routing == "balanced":
            assert counts.max() - counts.min() <= 1
        if routing == "hotspot":
            assert counts[0] == 19
    for counts in ([1, 2], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 38], [1.0] * 5):
        with pytest.raises(ValueError, match="counts"):
            generate(counts=counts)


def malformed(kind, valid):
    """Corrupt one part of an input file and return the expected error text."""
    data = bytearray(valid)
    if kind == "header":
        return data[:10], "truncated"
    if kind == "payload":
        return data[:-1], "payload length"
    if kind == "surplus":
        return data + b"extra", "payload length"
    if kind == "magic":
        data[0] = 0
        return data, "magic or version"
    if kind == "version":
        struct.pack_into("<I", data, 8, 2)
        return data, "magic or version"
    if kind == "dimension":
        struct.pack_into("<Q", data, 12, INT_MAX + 1)
        return data, "dimensions"
    if kind == "zero_dimension":
        struct.pack_into("<Q", data, 28, 0)
        return data, "dimensions"
    if kind == "overflow":
        return INPUT_HEADER.pack(INPUT_MAGIC, 1, 1, INT_MAX, INT_MAX, INT_MAX), "overflow"
    if kind == "start":
        struct.pack_into("<q", data, INPUT_HEADER.size, 1)
        return data, "offsets"
    if kind == "end":
        struct.pack_into("<q", data, INPUT_HEADER.size + 3 * 8, 8)
        return data, "offsets"
    if kind == "decreasing":
        struct.pack_into("<qqqq", data, INPUT_HEADER.size, 0, 7, 2, 9)
        return data, "offsets"
    if kind == "nan":
        struct.pack_into("<f", data, INPUT_HEADER.size + 4 * 8, float("nan"))
        return data, "non-finite"
    raise AssertionError(kind)


BAD_CASES = ["header", "payload", "surplus", "magic", "version", "dimension",
             "zero_dimension", "start", "end", "decreasing", "nan"]


@pytest.mark.parametrize("kind", BAD_CASES)
def test_python_rejects_bad_input(tmp_path, kind):
    """Check that the Python reader rejects a corrupted input file."""
    path = tmp_path / "bad.bin"
    write_input(path, generate(9, 3, 5, 7))
    data, message = malformed(kind, path.read_bytes())
    path.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        read_input(path)


@pytest.mark.parametrize("kind", BAD_CASES + ["overflow"])
def test_cpp_rejects_bad_input(v1_binary, tmp_path, kind):
    """Check that v1 rejects a bad file without creating an output."""
    path, output = tmp_path / "bad.bin", tmp_path / "output.bin"
    write_input(path, generate(9, 3, 5, 7))
    data, message = malformed(kind, path.read_bytes())
    path.write_bytes(data)
    result = subprocess.run([str(v1_binary), str(path), str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert message in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("kind", ["truncate", "surplus", "nan", "magic", "version", "dimension"])
def test_output_validation(tmp_path, kind):
    """Check that the output reader rejects malformed headers and payloads."""
    path = tmp_path / "output.bin"
    write_output(path, np.ones((3, 5), dtype=np.float32))
    data = bytearray(path.read_bytes())
    if kind == "truncate":
        data = data[:-1]
    elif kind == "surplus":
        data += b"x"
    elif kind == "nan":
        struct.pack_into("<f", data, 28, float("nan"))
    elif kind == "magic":
        data[0] = 0
    elif kind == "version":
        struct.pack_into("<I", data, 8, 2)
    else:
        struct.pack_into("<Q", data, 20, 0)
    path.write_bytes(data)
    with pytest.raises(ValueError):
        read_output(path)


def test_reference_restores_math_mode():
    """Check that the reference preserves the caller's TF32 setting."""
    previous = torch.backends.cuda.matmul.allow_tf32
    reference(generate(3, 2, 5, 7), "cpu")
    assert torch.backends.cuda.matmul.allow_tf32 == previous
