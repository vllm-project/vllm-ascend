#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import Gemma4Speculator

from vllm_ascend.worker.v2 import spec_decode as v2_spec_decode
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.gemma4.speculator import (
    AscendGemma4Speculator,
)


def _speculative_config():
    config = MagicMock()
    config.method = "mtp"
    config.use_gemma4_mtp.return_value = True
    config.use_step3p5_mtp.return_value = False
    config.use_dspark.return_value = False
    config.use_dflash.return_value = False
    config.use_eagle.return_value = True
    return config


def test_routes_gemma4_mtp_to_ascend_gemma4_speculator():
    vllm_config = SimpleNamespace(speculative_config=_speculative_config())
    expected = object()
    with patch(
        "vllm_ascend.worker.v2.spec_decode.gemma4.speculator.AscendGemma4Speculator",
        return_value=expected,
    ) as speculator_cls:
        result = v2_spec_decode.init_speculator(vllm_config, torch.device("cpu"))

    assert result is expected
    speculator_cls.assert_called_once()


def test_eagle_speculator_is_not_reached_for_gemma4():
    vllm_config = SimpleNamespace(speculative_config=_speculative_config())
    with patch(
        "vllm_ascend.worker.v2.spec_decode.gemma4.speculator.AscendGemma4Speculator",
        return_value=object(),
    ):
        v2_spec_decode.init_speculator(vllm_config, torch.device("cpu"))

    vllm_config.speculative_config.use_eagle.assert_not_called()


def test_gemma4_speculator_mro():
    assert AscendGemma4Speculator.__mro__[:4] == (
        AscendGemma4Speculator,
        AscendAutoRegressiveSpeculator,
        Gemma4Speculator,
        AutoRegressiveSpeculator,
    )
    assert AscendGemma4Speculator.advance_draft_positions is Gemma4Speculator.advance_draft_positions
    assert AscendGemma4Speculator._setup_gemma4_kv_sharing is Gemma4Speculator._setup_gemma4_kv_sharing
    assert AscendGemma4Speculator._share_embeddings is Gemma4Speculator._share_embeddings


def test_sync_kv_sharing_target_to_impl():
    speculator = AscendGemma4Speculator.__new__(AscendGemma4Speculator)
    synced_impl = SimpleNamespace(kv_sharing_target_layer_name=None)
    untouched_impl = SimpleNamespace(kv_sharing_target_layer_name=None)
    draft_model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        attn=SimpleNamespace(
                            impl=synced_impl,
                            kv_sharing_target_layer_name="target.attn",
                        )
                    )
                ),
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        attn=SimpleNamespace(
                            impl=untouched_impl,
                            kv_sharing_target_layer_name=None,
                        )
                    )
                ),
            ]
        )
    )

    speculator._sync_kv_sharing_target_to_impl(draft_model)

    assert synced_impl.kv_sharing_target_layer_name == "target.attn"
    assert untouched_impl.kv_sharing_target_layer_name is None


def test_sync_kv_sharing_target_to_impl_without_layers():
    speculator = AscendGemma4Speculator.__new__(AscendGemma4Speculator)
    speculator._sync_kv_sharing_target_to_impl(SimpleNamespace(model=SimpleNamespace(layers=[])))
