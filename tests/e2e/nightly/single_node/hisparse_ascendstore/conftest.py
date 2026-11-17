# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project


def pytest_addoption(parser):
    group = parser.getgroup("hisparse-ascendstore")
    group.addoption("--hisparse-sdk", action="store_true", help="Run the MemCache SDK test on NPU 0")
    group.addoption("--hisparse-replay-url", default="", help="URL of an idle A5 dev-mode test server")
    group.addoption("--hisparse-replay-mode", choices=("only", "store", "plain-store", "plain"), default="store")
    group.addoption("--hisparse-replay-mtp", action="store_true", help="Require MTP=3 with async scheduling")
    group.addoption("--hisparse-replay-long", action="store_true", help="Include 128K prompts")
    group.addoption("--hisparse-replay-output-tokens", type=int, default=32)
    group.addoption("--hisparse-replay-logprob-atol", type=float, default=0.0)
