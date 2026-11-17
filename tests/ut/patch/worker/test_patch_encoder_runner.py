# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

from vllm.v1.worker.gpu.mm import encoder_runner

import vllm_ascend.patch.worker.patch_encoder_runner  # noqa: F401


def test_encoder_h2d_uses_pageable_host_memory():
    runner = object.__new__(encoder_runner.EncoderRunner)
    runner.device = "npu"
    runner.cudagraph_manager = None
    runner.model = MagicMock()
    runner.model.embed_multimodal.return_value = ["encoded"]
    mm_item = ("image", MagicMock())

    with (
        patch.object(
            encoder_runner,
            "group_and_batch_mm_kwargs",
            return_value=[("image", 1, {"pixel_values": "features"})],
        ) as group_and_batch,
        patch.object(encoder_runner, "sanity_check_mm_encoder_outputs"),
    ):
        assert runner.execute_mm_encoder([mm_item]) == ["encoded"]

    group_and_batch.assert_called_once_with([mm_item], device="npu", pin_memory=False)
    runner.model.embed_multimodal.assert_called_once_with(pixel_values="features")
