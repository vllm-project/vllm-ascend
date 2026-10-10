# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def _dummy_runner(use_embeds, runner_cls):
    runner = runner_cls.__new__(runner_cls)
    runner.kvpp = SimpleNamespace(scheduler=object(), prepare_forward=lambda _: None, complete_forward=lambda: None)
    runner.uniform_decode_query_len = 1
    runner.scheduler_config = SimpleNamespace(max_num_batched_tokens=4, max_num_seqs=4)
    runner.dynamic_eplb = False
    runner.ascend_config = SimpleNamespace(enable_force_eplb=False)
    runner.dcp_size = 1
    runner._determine_batch_execution_and_padding = MagicMock(
        return_value=(CUDAGraphMode.FULL, SimpleNamespace(num_tokens=4, num_reqs=4), None, None, None)
    )
    runner.synchronize_input_prep = nullcontext
    runner._should_build_dummy_attn_metadata = MagicMock(return_value=False)
    runner.maybe_dummy_run_with_lora = MagicMock(return_value=nullcontext())
    runner.lora_config = None
    runner.max_num_tokens = 4
    runner.device = torch.device("cpu")
    runner.supports_mm_inputs = False
    runner.model_config = SimpleNamespace(is_encoder_decoder=False, get_vocab_size=lambda: 97)
    runner.enable_prompt_embeds = use_embeds
    runner.input_ids = SimpleNamespace(gpu=torch.zeros(4, dtype=torch.int64))
    runner.inputs_embeds = SimpleNamespace(gpu=torch.zeros(4, 3))
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.positions = torch.zeros(4, dtype=torch.int64)
    runner.drafter = None
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=None), parallel_config=SimpleNamespace(data_parallel_size=1)
    )
    runner.model = MagicMock()
    runner._has_sinks = False
    runner.use_aux_hidden_state_outputs = False
    runner.use_compress = False
    runner._finalize_dump_data = MagicMock()
    runner.device_metadata_executor = None
    return runner


@pytest.mark.parametrize("runner_cls", [NPUModelRunner, NPUModelRunner310])
@pytest.mark.parametrize("randomize_inputs", [False, True])
@pytest.mark.parametrize("use_embeds", [False, True])
def test_v1_dummy_randomization_is_applied_during_forward_and_restored(randomize_inputs, use_embeds, runner_cls):
    runner = _dummy_runner(use_embeds, runner_cls)
    inputs = runner.inputs_embeds.gpu if use_embeds else runner.input_ids.gpu
    expected_value = (2 if use_embeds else 7) if randomize_inputs else 0

    def forward(num_tokens, input_ids, positions, intermediate_tensors, inputs_embeds):
        observed = inputs_embeds if use_embeds else input_ids
        assert torch.equal(observed, torch.full_like(inputs, expected_value))
        return torch.zeros((num_tokens, 1))

    runner._model_forward = MagicMock(side_effect=forward)
    with (
        patch("vllm_ascend.worker.model_runner_v1.get_pp_group", return_value=SimpleNamespace(is_first_rank=True)),
        patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable", return_value=False),
        patch("vllm_ascend.worker.model_runner_v1.set_ascend_forward_context", return_value=nullcontext()),
        patch("vllm_ascend.worker.model_runner_v1.update_cos_sin"),
        patch("torch.randint_like", return_value=torch.full_like(runner.input_ids.gpu, 7)),
        patch("torch.randn_like", return_value=torch.full_like(runner.inputs_embeds.gpu, 2)),
    ):
        runner._dummy_run(4, cudagraph_runtime_mode=CUDAGraphMode.FULL, randomize_inputs=randomize_inputs)
    runner._model_forward.assert_called_once()
    assert torch.count_nonzero(inputs) == 0


@pytest.mark.parametrize("randomize_inputs", [False, True])
def test_v1_profile_passes_randomization_to_mc2_and_upstream(randomize_inputs):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.eplb_warmup = MagicMock()
    runner.sparse_kv_offload_enabled = False
    runner.max_num_tokens = 8
    runner.vllm_config = MagicMock()
    runner.get_model = MagicMock()
    runner._dummy_run = MagicMock()

    def upstream_profile(self, randomize_inputs=False):
        self._dummy_run(self.max_num_tokens, is_profile=True, randomize_inputs=randomize_inputs)

    with (
        patch("vllm_ascend.worker.model_runner_v1.get_mc2_tokens_capacity", return_value=4),
        patch("vllm_ascend.worker.model_runner_v1.select_moe_comm_method", return_value=MoECommType.MC2),
        patch("vllm_ascend.worker.model_runner_v1.disable_compilation", return_value=nullcontext()),
        patch.object(GPUModelRunner, "profile_run", upstream_profile),
    ):
        runner.profile_run(randomize_inputs=randomize_inputs)
    runner._dummy_run.assert_has_calls(
        [
            call(4, with_prefill=True, is_profile=True, randomize_inputs=randomize_inputs),
            call(8, is_profile=True, randomize_inputs=randomize_inputs),
        ]
    )
    assert runner._dummy_run.call_count == 2


@pytest.mark.parametrize("runner_cls", [NPUModelRunner, NPUModelRunner310])
def test_upstream_v1_profile_dispatch_accepts_new_randomize_keyword(runner_cls):
    import inspect

    runner = runner_cls.__new__(runner_cls)
    runner.supports_mm_inputs = False
    runner.max_num_tokens = 4
    observed = []

    class ProfileReachedDummy(Exception):
        pass

    def dummy_dispatch(*args, **kwargs):
        # Bind against the actual Ascend override before stopping ahead of
        # device execution. The new upstream profile always supplies this kwarg.
        inspect.signature(runner_cls._dummy_run).bind(runner, *args, **kwargs)
        observed.append((args, kwargs))
        raise ProfileReachedDummy

    runner._dummy_run = dummy_dispatch
    with pytest.raises(ProfileReachedDummy):
        GPUModelRunner.profile_run(runner)
    assert observed == [((4,), {"is_profile": True, "randomize_inputs": False})]
