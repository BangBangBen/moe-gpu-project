from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def pytest_addoption(parser):
    """Register the optional path to the compiled v1 executable."""
    parser.addoption("--v1-binary", type=Path, default=None, help="Enable C++ executable tests")


@pytest.fixture
def v1_binary(request):
    """Return the validated executable path, or skip C++ tests when it is omitted."""
    binary = request.config.getoption("--v1-binary")
    if binary is None:
        pytest.skip("pass --v1-binary to enable C++ tests")
    binary = binary.resolve()
    if not binary.is_file():
        pytest.fail(f"v1 binary not found: {binary}")
    return binary
