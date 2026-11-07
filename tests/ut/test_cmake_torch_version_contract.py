from pathlib import Path


def test_cmake_reads_torch_version_without_importing_device_backend():
    cmake = (Path(__file__).parents[2] / "CMakeLists.txt").read_text()

    assert "from importlib.metadata import version" in cmake
    assert "version('torch').split('+', 1)[0]" in cmake
    assert "import torch; print(torch.__version__)" not in cmake
    assert 'if(NOT "${TORCH_VERSION}" VERSION_EQUAL "2.10.0")' in cmake
