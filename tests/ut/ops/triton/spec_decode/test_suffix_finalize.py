# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_ascend.ops.triton.spec_decode import suffix_finalize as ops


def test_finalize_uses_active_mapping_strides_and_internal_padding(monkeypatch):
    kernel = MagicMock()
    monkeypatch.setattr(ops, "_suffix_finalize_kernel", kernel)
    token_ids = SimpleNamespace(shape=(128, 2048), stride=lambda axis: 2056)
    mapping = SimpleNamespace(shape=(7,))
    scratch = SimpleNamespace(shape=(128, 8), stride=lambda axis: 12)
    args = (token_ids, 2048, mapping, object(), object(), object(), scratch, 8, 8, 2049, *[object() for _ in range(9)])
    options = dict(
        HAS_CORPUS=True,
        ADAPTIVE=True,
        HASH_MUL=-7046029254386353131,
        NUM_BUCKETS=64,
        SLOTS=8,
        MAX_DEPTH=3,
        DEPTH_PO2=4,
        K=1,
        K_PO2=1,
        NB_PO2=8,
    )

    ops.launch_suffix_finalize(*args, **options)

    kernel.__getitem__.assert_called_once_with((7,))
    expected_args = (token_ids, 2056, *args[2:7], 12, *args[8:])
    kernel.__getitem__.return_value.assert_called_once_with(*expected_args, **(options | {"K_PO2": 2}))
