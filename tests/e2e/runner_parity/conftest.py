# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from pathlib import Path

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("runner-parity")
    group.addoption("--parity-root", help="Collected run directory containing v1/ and v2/")
    group.addoption("--upstream-utils", help="Pinned vLLM tests/models/utils.py")
    group.addoption("--control-root", help="Separate same-host deterministic control run")
    group.addoption("--feature-root", help="Full paired MTP/boundary/actual-generation parser capture")
    group.addoption("--expected-rounds", type=int, default=1, help="Explicit number of captured paired serving rounds")


@pytest.fixture
def parity_root(request):
    value = request.config.getoption("--parity-root")
    if value is None:
        pytest.skip("Requires collected full-model NPU artifacts")
    return Path(value)


@pytest.fixture
def upstream_utils(request):
    value = request.config.getoption("--upstream-utils")
    if value is None:
        pytest.skip("Pass the pinned upstream numerical helper explicitly")
    path = Path(value)
    assert path.is_file()
    return path


@pytest.fixture
def expected_rounds(request):
    value = request.config.getoption("--expected-rounds")
    assert value > 0
    return value


@pytest.fixture
def control_root(request):
    value = request.config.getoption("--control-root")
    if value is None:
        pytest.skip("Requires a separate deterministic full-model control capture")
    return Path(value)


@pytest.fixture
def feature_root(request):
    value = request.config.getoption("--feature-root")
    if value is None:
        pytest.skip("Requires paired full-model MTP, boundary and actual-generation artifacts")
    return Path(value)
