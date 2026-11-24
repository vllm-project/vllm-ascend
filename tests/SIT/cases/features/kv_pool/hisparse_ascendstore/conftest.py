# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project


def pytest_addoption(parser):
    group = parser.getgroup("hisparse-ascendstore")
    group.addoption("--hisparse-replay-url", default="", help="URL of the A5 HiSparse + AscendStore test server")
    group.addoption("--hisparse-replay-reference-url", default="", help="URL of the matching A5 HiSparse-only server")
