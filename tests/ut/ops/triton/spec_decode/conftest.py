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


@pytest.fixture
def suffix_options():
    return {
        "local_scan": {"MAX_DEPTH": 3, "BLOCK": 256},
        "finalize": {
            "HAS_CORPUS": True,
            "ADAPTIVE": True,
            "HASH_MUL": -7046029254386353131,
            "NUM_BUCKETS": 64,
            "SLOTS": 8,
            "MAX_DEPTH": 3,
            "DEPTH_PO2": 4,
            "K": 1,
            "K_PO2": 1,
            "NB_PO2": 8,
        },
        "ingest": {"HASH_MUL": -7046029254386353131, "NUM_BUCKETS": 64, "SLOTS": 8, "BLOCK": 1024},
        "observe": {"K": 3, "K_PO2": 4},
    }
