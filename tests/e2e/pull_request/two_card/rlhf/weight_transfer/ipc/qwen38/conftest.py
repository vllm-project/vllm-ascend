# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in for the local Qwen3.8 PLE/MRoPE IPC case."""


def pytest_addoption(parser):
    group = parser.getgroup("qwen38-ipc")
    group.addoption("--qwen38-model-dir", help="Four-layer PLE/MRoPE descriptor created by qwen38_manifest")
    group.addoption("--qwen38-enable-npu", action="store_true", default=False, help="Explicitly allow the TP2 NPU case")
