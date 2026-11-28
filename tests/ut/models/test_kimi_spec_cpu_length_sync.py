# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState

from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPMetadataBuilder
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
from vllm_ascend.models.kimi_k3.model_state import AscendKimiModelState
from vllm_ascend.models.kimi_k3.recoverssm_metadata import AscendKimiKDARecoverSSMMetadataBuilder
from vllm_ascend.ops.kimi_kda import KimiKDAHostMetadataBuilder
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


@pytest.fixture
def runner():
    state = object.__new__(AscendKimiModelState)
    state.recoverssm = object.__new__(RecoverSSMState)
    state.device_metadata = None
    draft = SimpleNamespace(
        use_a3_mla_sink=True,
        attn_architecture="MLA",
        draft_attn_layer_names={"draft"},
        use_flashmla=False,
        attn_backends={"draft": object},
    )
    target = SimpleNamespace(
        layer_names=["target"], metadata_builders=[object.__new__(AscendKimiKDARecoverSSMMetadataBuilder)]
    )
    draft_group = SimpleNamespace(layer_names=["draft"], metadata_builders=[object()])
    runner = SimpleNamespace(
        model_state=state,
        speculator=draft,
        attn_groups=[[target, draft_group]],
        sync_spec_pp_cpu_counts=False,
        num_computed_tokens_stream=Mock(),
        model_config=SimpleNamespace(
            architecture="KimiK31ForConditionalGeneration", hf_config=SimpleNamespace(model_type="kimi_k31")
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=8
        ),
        ascend_config=SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False)),
        speculative_config=SimpleNamespace(
            method="dspark",
            draft_model_config=SimpleNamespace(architectures=["K3DSparkModel"]),
            enable_adaptive_verification=False,
        ),
    )
    runner._can_skip_spec_cpu_length_sync = lambda: NPUModelRunner._can_skip_spec_cpu_length_sync(runner)
    return runner


@pytest.mark.parametrize("dcp", [1, 8])
@pytest.mark.parametrize("native", [True, False], ids=["a3_mla", "fia"])
@pytest.mark.parametrize("recover", [False, True])
def test_copy_and_wait_after_rejection(runner, monkeypatch, native, dcp, recover):
    if not recover:
        runner.model_state.recoverssm = None
        runner.attn_groups[0][0].metadata_builders = [object.__new__(KimiKDAHostMetadataBuilder)]
    runner.parallel_config.decode_context_parallel_size = dcp
    sfa = AscendSFAMetadataBuilder if dcp == 1 else AscendSFADCPMetadataBuilder
    runner.attn_groups[0][0].metadata_builders.append(object.__new__(sfa))
    runner.speculator.use_a3_mla_sink = native
    NPUModelRunner.initialize_spec_cpu_length_sync(runner)
    assert runner._can_skip_spec_cpu_length_sync() is native
    assert runner.num_computed_tokens_stream.synchronize.call_count == int(native)
    runner.num_computed_tokens_event = Mock()
    runner.req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([98, 195])),
        num_computed_tokens_cpu=torch.tensor([100, 200]),
        req_id_to_index={"a": 0, "b": 1},
    )
    runner.num_computed_tokens_cpu = torch.full((2,), -1)
    runner.input_buffers = SimpleNamespace(seq_lens_cpu=torch.zeros(2, dtype=torch.int32))
    scheduler = SimpleNamespace(
        num_scheduled_tokens={"a": 4, "b": 4}, scheduled_cached_reqs=SimpleNamespace(req_ids=["b", "a"])
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(stream=lambda _: nullcontext()), raising=False)
    NPUModelRunner._copy_num_computed_tokens_to_cpu(runner)
    NPUModelRunner._update_seq_lens_cpu(runner, scheduler, ["b", "a"])
    assert runner.num_computed_tokens_event.record.call_count == int(not native)
    assert runner.num_computed_tokens_event.synchronize.call_count == int(not native)
    assert runner.num_computed_tokens_cpu.tolist() == ([-1, -1] if native else [98, 195])
    assert runner.input_buffers.seq_lens_cpu.tolist() == ([204, 104] if native else [199, 102])
    assert runner.req_states.num_computed_tokens.gpu.tolist() == [98, 195]


@pytest.mark.parametrize(
    "change", ["unknown_target", "draft_only", "custom_recover", "offload", "adaptive", "pcp", "pp", "a5"]
)
def test_reinitialize_length_policy(runner, change):
    NPUModelRunner.initialize_spec_cpu_length_sync(runner)
    assert runner._can_skip_spec_cpu_length_sync()
    if change == "unknown_target":
        runner.attn_groups[0][0].metadata_builders.append(object())
    elif change == "draft_only":
        runner.attn_groups[0].pop(0)
    elif change == "custom_recover":
        runner.model_state.recoverssm = object()
    elif change == "offload":
        runner.ascend_config.sparse_kv_offload_config.enabled = True
    elif change == "adaptive":
        runner.speculative_config.enable_adaptive_verification = True
    elif change == "pcp":
        runner.parallel_config.prefill_context_parallel_size = 2
    elif change == "pp":
        runner.parallel_config.pipeline_parallel_size = 2
    else:
        runner.speculator.use_a3_mla_sink = False
        runner.speculator.use_flashmla = True
        runner.model_state.device_metadata = object()
    NPUModelRunner.initialize_spec_cpu_length_sync(runner)
    assert not runner.model_state.can_skip_spec_cpu_length_sync()
    assert runner._can_skip_spec_cpu_length_sync() is (change == "a5")
