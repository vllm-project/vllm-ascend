# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

pytest.importorskip("vllm")

from vllm_ascend.patch.worker.patch_multimodal_merge_310 import (
    merge_multimodal_embeddings_310,
)


def test_merge_multimodal_embeddings_310() -> None:
    inputs_embeds = torch.arange(48, dtype=torch.float32).reshape(2, 6, 4)
    expected = inputs_embeds.clone()
    is_multimodal = torch.tensor(
        [
            [False, True, True, False, False, True],
            [True, False, False, True, False, False],
        ]
    )
    mm_embeds = torch.tensor(
        [
            [10.0, 11.0, 12.0, 13.0],
            [20.0, 21.0, 22.0, 23.0],
            [30.0, 31.0, 32.0, 33.0],
            [40.0, 41.0, 42.0, 43.0],
            [50.0, 51.0, 52.0, 53.0],
        ]
    )
    expected[is_multimodal] = mm_embeds

    result = merge_multimodal_embeddings_310(
        inputs_embeds.clone(),
        [mm_embeds],
        is_multimodal,
    )
    torch.testing.assert_close(result, expected)
