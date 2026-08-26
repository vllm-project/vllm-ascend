# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
"""Unit tests for the opt-in SP-with-PP patch (CPU only, mocked comms)."""

from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch
from vllm.sequence import IntermediateTensors

import vllm_ascend.patch.platform.patch_sp_with_pp as sp_pp


@dataclass
class FakeParallelConfig:
    pipeline_parallel_size: int = 2
    use_sequence_parallel_moe: bool = True


@dataclass
class FakeVllmConfig:
    parallel_config: FakeParallelConfig | None = None


class TestPredicate:
    def test_disabled_by_default(self):
        with patch.object(sp_pp, "VLLM_ASCEND_ENABLE_SP_WITH_PP", False):
            assert not sp_pp.sp_with_pp_enabled(FakeParallelConfig())

    def test_enabled_matrix(self):
        cases = [
            (2, True, True),
            # PP=1 is also covered: the patched forwards fix the upstream
            # SP decode NaN there
            (1, True, True),
            (2, False, False),
        ]
        for pp, sp_moe, expected in cases:
            with patch.object(sp_pp, "VLLM_ASCEND_ENABLE_SP_WITH_PP", True):
                cfg = FakeParallelConfig(pipeline_parallel_size=pp, use_sequence_parallel_moe=sp_moe)
                assert sp_pp.sp_with_pp_enabled(cfg) is expected, (pp, sp_moe)

    def test_none_config(self):
        with patch.object(sp_pp, "VLLM_ASCEND_ENABLE_SP_WITH_PP", True):
            assert not sp_pp.sp_with_pp_enabled(None)


def fake_all_gather(x, dim):
    # simulate TP=2: each rank holds one half; gather concatenates in rank order
    return torch.cat([x, x], dim=dim)


class TestGatherIntermediateTensors:
    def test_gathers_shards_and_trims_pad(self):
        full = 3
        # rank-0 view of a padded shard: [2, 4] (padded from ceil(3/2))
        shard = torch.ones(2, 4)
        shard[1] *= 2  # mark the pad row so we can check the trim
        tensors = IntermediateTensors({"hidden_states": shard, "residual": torch.ones(2, 4)})
        with patch.object(sp_pp, "tensor_model_parallel_all_gather", fake_all_gather):
            out = sp_pp._gather_intermediate_tensors(tensors, full)
        assert out.tensors["hidden_states"].shape == (3, 4)
        # after gather the first `full` rows come from the rank-0 shard
        assert torch.allclose(out.tensors["hidden_states"][0], torch.ones(4))
        assert torch.allclose(out.tensors["hidden_states"][1], torch.ones(4) * 2)

    def test_full_tensors_passthrough(self):
        t = torch.randn(5, 4)
        tensors = IntermediateTensors({"hidden_states": t, "residual": t})
        out = sp_pp._gather_intermediate_tensors(tensors, 5)
        assert out.tensors["hidden_states"] is t


class TestModelForwardWrapper:
    def _make_model(self):
        class Layer:
            use_attn_reduce_scatter_for_moe = True

        class Model:
            vllm_config = FakeVllmConfig(FakeParallelConfig())
            config = type("C", (), {"hidden_size": 4})()
            layers = [Layer()]

            def forward(self, input_ids, positions, intermediate_tensors=None):
                # non-last PP rank: return sequence-sharded tensors
                return IntermediateTensors({"hidden_states": torch.ones(2, 4), "residual": torch.ones(2, 4)})

        return Model

    def test_wrapper_gathers_on_pp_boundary(self):
        model = self._make_model()()
        sp_pp._wrap_model_forward(type(model))
        positions = torch.zeros(3)  # 1-D: one position per token
        with (
            patch.object(sp_pp, "VLLM_ASCEND_ENABLE_SP_WITH_PP", True),
            patch.object(sp_pp, "tensor_model_parallel_all_gather", fake_all_gather),
        ):
            out = model.forward(input_ids=None, positions=positions)
        assert isinstance(out, IntermediateTensors)
        assert out.tensors["hidden_states"].shape == (3, 4)

    def test_wrapper_noop_when_sp_off(self):
        model = self._make_model()()
        model.vllm_config = FakeVllmConfig(
            FakeParallelConfig(pipeline_parallel_size=1, use_sequence_parallel_moe=False)
        )
        sp_pp._wrap_model_forward(type(model))
        with patch.object(sp_pp, "tensor_model_parallel_all_gather", fake_all_gather):
            out = model.forward(input_ids=None, positions=torch.zeros(3))
        # not patched: returns the original sharded tensors unchanged
        assert out.tensors["hidden_states"].shape == (2, 4)


class TestApply:
    def test_apply_noop_when_env_off(self):
        # VLLM_ASCEND_ENABLE_SP_WITH_PP defaults to 0; apply() must not
        # import or touch any vllm model class.
        with patch.object(sp_pp, "VLLM_ASCEND_ENABLE_SP_WITH_PP", False):
            sp_pp.apply()  # must be a no-op
        from vllm.model_executor.models import qwen3_next

        assert "sp_pp_forward" not in qwen3_next.Qwen3NextModel.__dict__.get("forward", lambda self: None).__name__

    def test_layer_init_passthrough_when_disabled(self):
        # _wrap_layer_init must call the original __init__ unchanged when
        # the predicate is off (SP-MoE off)
        seen = {}

        class Layer:
            def __init__(self, vllm_config, prefix=""):
                seen["pp"] = vllm_config.parallel_config.pipeline_parallel_size
                self.use_attn_reduce_scatter_for_moe = False

        sp_pp._wrap_layer_init(Layer)
        cfg = FakeVllmConfig(FakeParallelConfig(pipeline_parallel_size=1, use_sequence_parallel_moe=False))
        layer = Layer(cfg)
        assert seen["pp"] == 1
        assert layer.use_attn_reduce_scatter_for_moe is False

    def test_layer_init_uses_pp1_view_when_enabled(self):
        # with the feature on, the original __init__ must see PP=1 so the
        # attention projection is built with SP semantics
        seen = {}

        class Layer:
            def __init__(self, vllm_config, prefix=""):
                seen["pp"] = vllm_config.parallel_config.pipeline_parallel_size
                seen["orig_is_same_object"] = vllm_config.parallel_config is cfg.parallel_config
                self.use_attn_reduce_scatter_for_moe = (
                    vllm_config.parallel_config.use_sequence_parallel_moe
                    and vllm_config.parallel_config.pipeline_parallel_size == 1
                )

        sp_pp._wrap_layer_init(Layer)
        cfg = FakeVllmConfig(FakeParallelConfig(pipeline_parallel_size=2))
        with patch.object(sp_pp, "VLLM_ASCEND_ENABLE_SP_WITH_PP", True):
            layer = Layer(cfg)
        assert seen["pp"] == 1
        # the caller's config object must not be mutated
        assert cfg.parallel_config is not None
        assert cfg.parallel_config.pipeline_parallel_size == 2
        assert layer.use_attn_reduce_scatter_for_moe is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
