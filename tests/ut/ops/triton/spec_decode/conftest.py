# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest

KernelCall = tuple[str, tuple[int, ...], tuple[object, ...], dict[str, object]]


class RecordingKernel:
    def __init__(self, name: str, calls: list[KernelCall]):
        self.name = name
        self.calls = calls

    def __getitem__(self, grid: tuple[int, ...]):
        def launch(*args: object, **kwargs: object) -> None:
            self.calls.append((self.name, grid, args, kwargs))

        return launch


@pytest.fixture
def recording_kernel():
    return RecordingKernel
