import pytest

#add parameters
def pytest_addoption(parser):
    parser.addoption(
        "--weight_path",action="store",default="",
        help="weight_dir"
    )

@pytest.fixture
def weight_path(request):
    return request.config.getoption("--weight_path")
