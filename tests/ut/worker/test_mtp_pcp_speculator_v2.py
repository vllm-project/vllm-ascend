# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
from contextlib import nullcontext
from copy import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import (
    CUDAGraphMode,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    replace,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.gpu import dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode import utils as spec_utils
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

from vllm_ascend.worker.v2 import pcp_manager as pcp_manager_module
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.spec_decode.autoregressive import (
    speculator as speculator_module,
)
from vllm_ascend.worker.v2.spec_decode.eagle import (
    speculator as eagle_speculator_module,
)
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import (
    AscendMTPSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.pcp_utils import (
    disable_profiling_chunk_for_draft,
    disable_target_pcp_for_replicated_draft,
)


@pytest.fixture(autouse=True)
def _stub_ascend_config(monkeypatch):
    # Speculators built through the real __init__ read the global ascend
    # config during the lmhead TP construction-time validation; the stub
    # reads as lmhead-off so validation no-ops.
    from vllm_ascend import ascend_config as _ascend_config_module

    monkeypatch.setattr(
        _ascend_config_module,
        "_ASCEND_CONFIG",
        SimpleNamespace(
            finegrained_tp_config=SimpleNamespace(lmhead_tensor_parallel_size=0),
            ascend_compilation_config=object(),
            eplb_config=object(),
        ),
    )


def _fake_config_replace(config, **changes):
    values = vars(config).copy()
    values.update(changes)
    return SimpleNamespace(**values)


def _config(additional_config, pp_size=2):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        additional_config=additional_config,
    )


def _make_padded_input_batch() -> MagicMock:
    input_batch = MagicMock(spec=AscendInputBatch)
    input_batch.num_reqs = 2
    input_batch.num_reqs_after_padding = 4
    input_batch.num_tokens = 6
    input_batch.num_tokens_after_padding = 8
    input_batch.idx_mapping = torch.tensor([3, 7], dtype=torch.int32)
    input_batch.query_start_loc = torch.tensor([0, 3, 6, 6, 6], dtype=torch.int32)
    input_batch.query_start_loc_np = np.array([0, 3, 6, 6, 6], dtype=np.int32)
    input_batch.seq_lens = torch.arange(4, dtype=torch.int32)
    input_batch.seq_lens_cpu_upper_bound = torch.arange(4, dtype=torch.int32)
    input_batch.input_ids = torch.arange(8, dtype=torch.int32)
    input_batch.positions = torch.arange(8, dtype=torch.int64)
    input_batch.is_padding = torch.zeros(8, dtype=torch.bool)
    input_batch.seq_lens_np = np.arange(4, dtype=np.int32)
    input_batch.is_prefilling_np = np.zeros(4, dtype=np.bool_)
    return input_batch


@pytest.mark.parametrize(
    ("target_pcp_size", "expected_execution_pcp_size", "dcp_size"),
    [(2, 1, 4), (2, 1, 8), (1, 1, 4)],
)
def test_draft_runtime_config_preserves_target_worker_topology(
    target_pcp_size: int,
    expected_execution_pcp_size: int,
    dcp_size: int,
) -> None:
    draft_parallel_config = SimpleNamespace(
        prefill_context_parallel_size=2,
        cp_kv_cache_interleave_size=64,
        decode_context_parallel_size=4,
        enable_expert_parallel=False,
        enable_eplb=False,
        rank=0,
    )
    target_parallel_config = SimpleNamespace(
        prefill_context_parallel_size=target_pcp_size,
        cp_kv_cache_interleave_size=128,
        decode_context_parallel_size=dcp_size,
        enable_expert_parallel=True,
        enable_eplb=True,
        rank=7,
        data_parallel_size=2,
        data_parallel_rank=1,
        pipeline_parallel_size=2,
    )
    target_cache_config = SimpleNamespace(
        block_size=128,
    )
    target_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=False),
        parallel_config=target_parallel_config,
        speculative_config=SimpleNamespace(
            draft_parallel_config=draft_parallel_config,
            moe_backend=None,
            kv_cache_dtype=None,
            attention_backend=None,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(decode_mode=lambda: None),
        ),
        cache_config=target_cache_config,
        additional_config={"scheduler_config": {"profiling_chunk_config": {"enabled": True}}},
        load_config=SimpleNamespace(load_format="auto"),
    )
    draft_model_config = SimpleNamespace(is_moe=False)
    captured: dict[str, SimpleNamespace] = {}

    def fake_replace(config, **changes):
        if "pipeline_parallel_size" in changes:
            assert config.prefill_context_parallel_size == target_pcp_size
            assert config.decode_context_parallel_size == dcp_size
        if "model_config" in changes:
            assert changes["parallel_config"].prefill_context_parallel_size == target_pcp_size
            assert changes["parallel_config"].decode_context_parallel_size == dcp_size
        values = vars(config).copy()
        values.update(changes)
        return SimpleNamespace(**values)

    def fake_parent_init(speculator, execution_config, device):
        captured["execution_config"] = execution_config
        speculator.device = device
        speculator.vllm_config = execution_config
        speculator.speculative_config = execution_config.speculative_config
        speculator.draft_model_config = draft_model_config
        speculator.input_buffers = object()
        speculator.max_num_reqs = 4
        speculator.max_num_tokens = 8
        speculator.num_speculative_steps = 3

    with (
        patch.object(spec_utils, "get_pp_group", return_value=SimpleNamespace(world_size=1)),
        patch.object(speculator_module, "get_dcp_group", return_value=SimpleNamespace(rank_in_group=0)),
        patch.object(speculator_module, "DCPManager") as dcp_manager,
        patch.object(
            speculator_module,
            "replace",
            side_effect=fake_replace,
        ),
        patch.object(
            speculator_module.AutoRegressiveSpeculator,
            "__init__",
            new=fake_parent_init,
        ),
        patch.object(
            speculator_module,
            "AscendInputBuffers",
            return_value=object(),
        ),
    ):
        speculator = AscendMTPSpeculator(target_config, torch.device("cpu"))

    assert dcp_manager.call_args.kwargs["dcp_world_size"] == dcp_size
    assert dcp_manager.call_args.kwargs["dcp_rank"] == 0
    assert captured["execution_config"] is target_config
    execution_config = speculator.vllm_config
    execution_parallel_config = execution_config.parallel_config
    assert execution_parallel_config.prefill_context_parallel_size == expected_execution_pcp_size
    assert execution_parallel_config.cp_kv_cache_interleave_size == 128
    assert execution_parallel_config.decode_context_parallel_size == dcp_size
    assert target_parallel_config.decode_context_parallel_size == dcp_size
    assert execution_parallel_config.enable_expert_parallel
    assert execution_parallel_config.enable_eplb
    assert execution_parallel_config.rank == target_parallel_config.rank
    assert execution_parallel_config.data_parallel_size == 2
    assert execution_parallel_config.data_parallel_rank == 1
    assert target_parallel_config.prefill_context_parallel_size == target_pcp_size
    assert target_parallel_config.cp_kv_cache_interleave_size == 128
    assert target_parallel_config.enable_expert_parallel
    assert target_parallel_config.enable_eplb

    draft_config = speculator.draft_vllm_config
    assert speculator.attn_vllm_config is draft_config
    assert draft_parallel_config.prefill_context_parallel_size == 2
    assert draft_parallel_config.cp_kv_cache_interleave_size == 64
    assert not draft_parallel_config.enable_expert_parallel
    assert not draft_parallel_config.enable_eplb
    assert draft_config.model_config is draft_model_config
    assert draft_config.parallel_config.prefill_context_parallel_size == expected_execution_pcp_size
    assert draft_config.parallel_config.cp_kv_cache_interleave_size == 128
    assert draft_config.parallel_config.pipeline_parallel_size == 1
    assert draft_config.parallel_config.decode_context_parallel_size == dcp_size
    assert draft_config.additional_config["scheduler_config"]["profiling_chunk_config"]["enabled"] is False
    assert target_config.additional_config["scheduler_config"]["profiling_chunk_config"]["enabled"] is True


@pytest.mark.parametrize("dcp_size", [1, 2])
def test_eagle_draft_config_disables_profiling_chunk(dcp_size) -> None:
    additional_config = {"scheduler_config": {"profiling_chunk_config": {"enabled": "yes"}}}
    target_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2,
            prefill_context_parallel_size=2,
            enable_expert_parallel=True,
            enable_eplb=True,
        ),
        additional_config=additional_config,
    )
    speculator = object.__new__(AscendEagleSpeculator)
    # Delegation to the base's _create_draft_vllm_config reads these too.
    speculator.replicated_pcp = False
    speculator.vllm_config = target_config
    target_config.cache_config = SimpleNamespace()
    target_config.parallel_config.decode_context_parallel_size = dcp_size
    speculator.draft_model_config = SimpleNamespace(is_moe=False)

    with (
        patch.object(eagle_speculator_module, "replace", side_effect=_fake_config_replace),
        patch.object(speculator_module, "replace", side_effect=_fake_config_replace),
    ):
        draft_config = speculator._create_draft_vllm_config()

    assert draft_config.additional_config["scheduler_config"]["profiling_chunk_config"]["enabled"] is False
    assert target_config.additional_config is additional_config
    assert additional_config["scheduler_config"]["profiling_chunk_config"]["enabled"] == "yes"
    assert draft_config.parallel_config.pipeline_parallel_size == 1
    assert draft_config.parallel_config.is_moe_model is False
    assert draft_config.parallel_config.prefill_context_parallel_size == 1
    assert draft_config.parallel_config.decode_context_parallel_size == dcp_size
    assert not draft_config.parallel_config.enable_expert_parallel
    assert not draft_config.parallel_config.enable_eplb
    assert target_config.parallel_config.prefill_context_parallel_size == 2
    assert target_config.parallel_config.enable_expert_parallel
    assert target_config.parallel_config.enable_eplb


@pytest.fixture
def eplb_worker_config(tmp_path, monkeypatch):
    # Runner V2 checks Triton availability; this test stops before execution.
    monkeypatch.setattr("vllm.config.vllm.HAS_TRITON", True)
    # Real validation with local metadata; no model weights or network needed.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV3ForCausalLM"],
                "model_type": "deepseek_v3",
                "hidden_size": 16,
                "intermediate_size": 32,
                "moe_intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_nextn_predict_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "vocab_size": 32,
                "max_position_embeddings": 128,
                "n_routed_experts": 8,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "n_group": 1,
                "topk_group": 1,
                "kv_lora_rank": 4,
                "qk_nope_head_dim": 4,
                "qk_rope_head_dim": 4,
                "v_head_dim": 4,
            }
        ),
        encoding="utf-8",
    )
    with patch("vllm.config.parallel.current_platform.is_cuda_alike", return_value=True):
        model_config = ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            dtype="float32",
            enforce_eager=True,
            max_model_len=128,
        )
        parallel_config = ParallelConfig(
            tensor_parallel_size=1,
            prefill_context_parallel_size=8,
            enable_expert_parallel=True,
            enable_eplb=True,
            distributed_executor_backend="mp",
        )
        return VllmConfig(
            model_config=model_config,
            parallel_config=parallel_config,
            speculative_config=SpeculativeConfig(
                method="mtp",
                num_speculative_tokens=2,
                target_model_config=model_config,
                target_parallel_config=parallel_config,
            ),
            scheduler_config=SchedulerConfig(
                max_model_len=128, max_num_batched_tokens=128, max_num_seqs=4, is_encoder_decoder=False
            ),
            device_config=DeviceConfig(device="cpu"),
        )


@pytest.mark.parametrize(
    ("pcp_size", "dcp_size", "branch", "pp_size"),
    [
        (1, 1, "default", 1),
        (1, 1, "all", 1),
        (8, 1, "default", 1),
        (8, 1, "moe", 1),
        (8, 1, "kv", 1),
        (8, 1, "attention", 1),
        (8, 1, "all", 1),
        (8, 8, "all", 1),
        (8, 8, "all", 2),
        (8, 1, "default", 2),
    ],
)
def test_eplb_draft_overrides_reach_inherited_loader(eplb_worker_config, pcp_size, dcp_size, branch, pp_size):
    def fake_parent_init(speculator, config, device):
        speculator.vllm_config = config
        speculator.speculative_config = config.speculative_config
        speculator.draft_model_config = config.speculative_config.draft_model_config
        speculator.input_buffers = object()
        speculator.max_num_reqs = 4
        speculator.max_num_tokens = 8
        speculator.num_speculative_steps = 2

    with patch("vllm.config.parallel.current_platform.is_cuda_alike", return_value=True):
        speculative_config = copy(eplb_worker_config.speculative_config)
        speculative_config.moe_backend = "auto" if branch in ("moe", "all") else None
        speculative_config.kv_cache_dtype = "bfloat16" if branch in ("kv", "all") else None
        speculative_config.attention_backend = AttentionBackendEnum.CUSTOM if branch in ("attention", "all") else None
        worker_config = replace(
            eplb_worker_config,
            parallel_config=replace(
                eplb_worker_config.parallel_config,
                prefill_context_parallel_size=pcp_size,
                decode_context_parallel_size=dcp_size,
                enable_eplb=pcp_size > 1,
            ),
            speculative_config=speculative_config,
            kernel_config=replace(eplb_worker_config.kernel_config, moe_backend="triton"),
            load_config=replace(eplb_worker_config.load_config, load_format="fastsafetensors"),
        )
        with (
            patch.object(spec_utils, "get_pp_group", return_value=SimpleNamespace(world_size=pp_size)),
            patch.object(speculator_module.AutoRegressiveSpeculator, "__init__", new=fake_parent_init),
            patch.object(speculator_module.AscendAutoRegressiveSpeculator, "_init_dcp"),
            patch.object(speculator_module, "AscendInputBuffers", return_value=object()),
        ):
            speculator = AscendMTPSpeculator(worker_config, torch.device("cpu"))
        if pcp_size > 1:
            # Reconstructing the execution view reproduces the original EPLB error.
            with pytest.raises(ValueError, match="EPLB requires"):
                replace(speculator.vllm_config, cache_config=replace(speculator.vllm_config.cache_config))

        execution_config = speculator.vllm_config
        with (
            patch.object(spec_utils, "get_pp_group", return_value=SimpleNamespace(world_size=pp_size)),
            patch.object(eagle_utils, "get_model", side_effect=RuntimeError("reached model construction")) as get_model,
            pytest.raises(RuntimeError, match="reached model construction"),
        ):
            speculator.load_draft_model(torch.nn.Module(), set())

    loading_config = get_model.call_args.kwargs["vllm_config"]
    assert loading_config is execution_config
    assert get_model.call_args.kwargs["model_config"] is speculator.draft_model_config
    assert loading_config.model_config is worker_config.model_config
    assert loading_config.parallel_config.prefill_context_parallel_size == 1
    assert loading_config.parallel_config.decode_context_parallel_size == dcp_size
    assert loading_config.parallel_config.enable_eplb == (pcp_size > 1)
    assert loading_config.parallel_config.world_size == worker_config.parallel_config.world_size
    assert loading_config.kernel_config.moe_backend == (
        speculative_config.moe_backend or worker_config.kernel_config.moe_backend
    )
    assert loading_config.cache_config.cache_dtype == (
        speculative_config.kv_cache_dtype or worker_config.cache_config.cache_dtype
    )
    assert loading_config.attention_config.backend == (
        speculative_config.attention_backend or worker_config.attention_config.backend
    )
    assert loading_config.load_config.load_format == ("auto" if pp_size > 1 else "fastsafetensors")
    assert speculator.vllm_config is execution_config
    assert worker_config.speculative_config is speculative_config
    assert worker_config.parallel_config.prefill_context_parallel_size == pcp_size
    assert worker_config.parallel_config.decode_context_parallel_size == dcp_size
    assert worker_config.cache_config.cache_dtype == "auto"
    assert worker_config.kernel_config.moe_backend == "triton"
    assert worker_config.load_config.load_format == "fastsafetensors"


@pytest.mark.parametrize("replicated_pcp", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
def test_run_model_broadcasts_only_actual_replicated_hidden_states(replicated_pcp, shared, contiguous):
    speculator = object.__new__(speculator_module.AscendAutoRegressiveSpeculator)
    speculator.replicated_pcp = replicated_pcp
    last_hidden = torch.arange(48, dtype=torch.float32).view(6, 8)[:, ::2]
    other_hidden = (torch.arange(48, dtype=torch.float32) + 50).view(6, 8)[:, ::2]
    if contiguous:
        last_hidden = last_hidden.contiguous()
        other_hidden = other_hidden.contiguous()
    hidden = last_hidden if shared else other_hidden
    originals = [last_hidden[:4].clone(), hidden[:4].clone()]
    sent = []

    def broadcast(tensor, src):
        assert src == 0
        assert tensor.is_contiguous()
        assert tensor.shape == (4, 4)
        sent.append(tensor.clone())
        return tensor.add_(100)

    group = SimpleNamespace(broadcast=MagicMock(side_effect=broadcast))
    with (
        patch.object(speculator_module.AutoRegressiveSpeculator, "_run_model", return_value=(last_hidden, hidden)),
        patch.object(pcp_manager_module, "get_pcp_group", return_value=group) as get_group,
    ):
        result_last, result_hidden = speculator._run_model(4, None, None, None)

    if not replicated_pcp:
        get_group.assert_not_called()
        group.broadcast.assert_not_called()
        assert result_last is last_hidden
        assert result_hidden is hidden
        return
    get_group.assert_called_once_with()
    assert group.broadcast.call_count == (1 if shared else 2)
    torch.testing.assert_close(sent[0], originals[0])
    if shared:
        assert result_hidden is result_last
    else:
        torch.testing.assert_close(sent[1], originals[1])
    torch.testing.assert_close(result_last, originals[0] + 100)
    torch.testing.assert_close(result_hidden, originals[1] + 100)


@pytest.mark.parametrize(("replicated_pcp", "manager_is_disabled"), [(True, True), (False, False)])
def test_draft_pcp_context_restores_manager_after_error(
    replicated_pcp: bool,
    manager_is_disabled: bool,
) -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.pcp_manager = MagicMock()
    speculator.replicated_pcp = replicated_pcp
    speculator.model_state = SimpleNamespace(pcp_manager=speculator.pcp_manager)

    with (
        pytest.raises(RuntimeError, match="proposal failed"),
        disable_target_pcp_for_replicated_draft(speculator),
    ):
        assert (speculator.model_state.pcp_manager is None) is manager_is_disabled
        raise RuntimeError("proposal failed")

    assert speculator.model_state.pcp_manager is speculator.pcp_manager


def test_draft_pcp_context_rejects_mismatched_manager() -> None:
    current_pcp_manager = MagicMock()
    speculator = SimpleNamespace(
        replicated_pcp=True,
        model_state=SimpleNamespace(pcp_manager=current_pcp_manager),
        pcp_manager=MagicMock(),
    )

    with (
        pytest.raises(
            RuntimeError,
            match="requires model_state to use the target PCP manager",
        ),
        disable_target_pcp_for_replicated_draft(speculator),
    ):
        pytest.fail("mismatched manager must fail before draft execution")

    assert speculator.model_state.pcp_manager is current_pcp_manager


@pytest.mark.parametrize(
    ("replicated_pcp", "expected_source"),
    [(True, "draft"), (False, "target")],
)
def test_draft_prefill_attn_groups_follow_draft_topology(
    replicated_pcp: bool,
    expected_source: str,
) -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.replicated_pcp = replicated_pcp
    speculator.attn_groups = [["draft"]]
    speculator.target_attn_groups = [["target"]]

    expected = speculator.attn_groups if expected_source == "draft" else speculator.target_attn_groups
    assert speculator.draft_prefill_attn_groups is expected


@pytest.mark.parametrize("attn_architecture", ["GQA", "MLA", "DSA", "SFA"])
@pytest.mark.parametrize("cudagraph_runtime_mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
def test_prepare_replicated_prefill_attn_uses_global_batch(attn_architecture, cudagraph_runtime_mode) -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.attn_architecture = attn_architecture
    speculator.block_tables = MagicMock()
    speculator.kv_cache_config = object()
    speculator._build_attn_metadata = MagicMock(return_value={"draft.layer": object()})
    input_batch = _make_padded_input_batch()
    input_batch.is_dummy = False
    speculator.input_batch = input_batch
    speculator.replicated_pcp = True
    global_slot_mapping = torch.arange(input_batch.num_tokens_after_padding).unsqueeze(0)
    slot_mappings = {"draft.layer": object()}
    speculator.block_tables.compute_slot_mappings.return_value = global_slot_mapping
    original_attn_metadata = {"local.layer": object()}
    original_slot_mappings = MagicMock()

    with patch.object(
        speculator_module,
        "build_slot_mappings_by_layer",
        return_value=slot_mappings,
    ) as build_slot_mappings:
        attn_metadata, actual_slot_mappings = speculator._prepare_replicated_prefill_attn(
            original_attn_metadata,
            original_slot_mappings,
            input_batch.num_reqs_after_padding,
            input_batch.num_tokens_after_padding,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
        )

    assert attn_metadata == speculator._build_attn_metadata.return_value
    assert actual_slot_mappings is slot_mappings
    speculator.block_tables.gather_block_tables.assert_called_once_with(
        input_batch.idx_mapping,
        num_reqs_padded=input_batch.num_reqs_after_padding,
    )
    speculator.block_tables.compute_slot_mappings.assert_called_once_with(
        input_batch.idx_mapping,
        input_batch.query_start_loc,
        input_batch.positions,
        num_tokens_padded=input_batch.num_tokens_after_padding,
    )
    build_slot_mappings.assert_called_once_with(
        global_slot_mapping,
        speculator.kv_cache_config,
    )
    speculator._build_attn_metadata.assert_called_once_with(
        num_reqs=input_batch.num_reqs,
        batch_desc=BatchExecutionDescriptor(
            cg_mode=cudagraph_runtime_mode,
            num_tokens=input_batch.num_tokens_after_padding,
            num_reqs=input_batch.num_reqs_after_padding,
        ),
        query_start_loc_np=input_batch.query_start_loc_np,
        seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
        step=0,
    )


@pytest.mark.parametrize("cudagraph_runtime_mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
def test_prefill_rebuilds_replicated_pcp_metadata_before_filtering(cudagraph_runtime_mode) -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.replicated_pcp = True
    speculator.input_batch = _make_padded_input_batch()
    speculator.input_batch.is_dummy = False
    speculator.draft_attn_layer_names = {"draft.layer"}

    draft_metadata = object()
    global_slot_mappings = MagicMock()
    speculator._prepare_replicated_prefill_attn = MagicMock(
        return_value=(
            {
                "draft.layer": draft_metadata,
                "target.layer": object(),
            },
            global_slot_mappings,
        )
    )
    local_attn_metadata = {"local.layer": object()}
    local_slot_mappings = MagicMock()

    with patch.object(
        speculator_module.AutoRegressiveSpeculator,
        "_prefill",
    ) as parent_prefill:
        speculator._prefill(
            num_reqs=2,
            num_tokens=8,
            attn_metadata=local_attn_metadata,
            slot_mappings=local_slot_mappings,
            num_tokens_across_dp=None,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
        )

    speculator._prepare_replicated_prefill_attn.assert_called_once_with(
        local_attn_metadata,
        local_slot_mappings,
        2,
        8,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
    )
    parent_prefill.assert_called_once()
    parent_args = parent_prefill.call_args.args
    assert parent_args[:2] == (2, 8)
    assert parent_args[2] == {"draft.layer": draft_metadata}
    assert parent_args[3] is global_slot_mappings


@pytest.mark.parametrize("attn_architecture", ["GQA", "MLA", "DSA", "SFA"])
@pytest.mark.parametrize("replicated_pcp", [False, True])
def test_graph_prefill_builds_draft_metadata(attn_architecture: str, replicated_pcp: bool) -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.replicated_pcp = replicated_pcp
    speculator.attn_architecture = attn_architecture
    speculator.input_batch = _make_padded_input_batch()
    speculator.input_batch.is_dummy = False
    speculator.block_tables = MagicMock()
    speculator.kv_cache_config = object()
    speculator.draft_attn_layer_names = {"draft.layer"}
    local_draft_metadata = SimpleNamespace(decode=SimpleNamespace(actual_seq_lengths_q=[4, 8]))
    global_draft_metadata = object()
    speculator.model_state = SimpleNamespace(
        attn_metadata={"draft.layer": local_draft_metadata, "target.layer": object()},
    )
    speculator._build_attn_metadata = MagicMock(
        return_value={"draft.layer": global_draft_metadata},
    )

    with patch.object(speculator_module, "build_slot_mappings_by_layer", return_value={}) as build_slots:
        actual = speculator.build_draft_attn_metadatas(
            num_reqs_padded=2,
            num_tokens_padded=8,
            is_draft_model_prefill=True,
        )

    rebuild_metadata = replicated_pcp and attn_architecture in ("DSA", "SFA")
    expected_metadata = global_draft_metadata if rebuild_metadata else local_draft_metadata
    assert actual == [{"draft.layer": expected_metadata}]
    assert actual[0]["draft.layer"] is expected_metadata
    assert speculator._build_attn_metadata.call_count == int(rebuild_metadata)
    assert build_slots.call_count == int(rebuild_metadata)
    assert speculator.block_tables.gather_block_tables.call_count == int(replicated_pcp)
    assert speculator.block_tables.compute_slot_mappings.call_count == int(replicated_pcp)
    assert local_draft_metadata.decode.actual_seq_lengths_q[-1] == 8


@pytest.mark.parametrize("attn_architecture", ["GQA", "MLA"])
def test_graph_prefill_refreshes_captured_cache_buffers(attn_architecture: str) -> None:
    speculator = object.__new__(AscendEagleSpeculator)
    speculator.replicated_pcp = True
    speculator.attn_architecture = attn_architecture
    speculator.draft_attn_layer_names = {"draft.layer"}
    metadata = SimpleNamespace(actual_seq_lengths_q=[4, 8])
    speculator.model_state = SimpleNamespace(attn_metadata={"draft.layer": metadata})
    speculator.kv_cache_config = object()
    speculator._build_attn_metadata = MagicMock()

    # These views stand in for the persistent buffers bound during capture.
    captured_blocks = torch.zeros((2, 3), dtype=torch.int32)
    captured_slots = torch.full((1, 8), -1, dtype=torch.int32)
    block_ptr, slot_ptr = captured_blocks.data_ptr(), captured_slots.data_ptr()
    request_blocks = {3: [7, 13, 0], 7: [17, 19, 23]}

    def gather_blocks(idx_mapping, num_reqs_padded):
        captured_blocks.zero_()
        for row, req_idx in enumerate(idx_mapping.tolist()):
            captured_blocks[row] = torch.tensor(request_blocks[req_idx])
        return (captured_blocks[:num_reqs_padded],)

    def compute_slots(idx_mapping, query_start_loc, positions, num_tokens_padded):
        captured_slots.fill_(-1)
        for row, req_idx in enumerate(idx_mapping.tolist()):
            start, end = query_start_loc[row : row + 2].tolist()
            for token in range(start, end):
                position = int(positions[token])
                block = request_blocks[req_idx][position // 128]
                captured_slots[0, token] = block * 128 + position % 128
        return captured_slots[:, :num_tokens_padded]

    speculator.block_tables = SimpleNamespace(
        gather_block_tables=gather_blocks,
        compute_slot_mappings=compute_slots,
    )
    for req_idx, positions, expected_slots in [
        (3, [126, 127, 128, 129], [1022, 1023, 1664, 1665]),
        (7, [254, 255, 256, 257], [2558, 2559, 2944, 2945]),
    ]:
        batch = _make_padded_input_batch()
        batch.is_dummy = False
        batch.num_reqs, batch.num_tokens = 1, 4
        batch.query_start_loc_np = np.array([0, 4, 8], dtype=np.int32)
        batch.idx_mapping = torch.tensor([req_idx], dtype=torch.int32)
        batch.query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
        batch.positions = torch.tensor(positions, dtype=torch.int64)
        speculator.input_batch = batch
        # The preceding single-token decode has left a different slot layout.
        captured_slots.fill_(-1)
        captured_slots[0, 0] = 123
        captured_blocks.fill_(-1)

        with patch.object(speculator_module, "build_slot_mappings_by_layer") as build_slots:
            result = speculator.build_draft_attn_metadatas(2, 8, is_draft_model_prefill=True)

        assert result[0]["draft.layer"] is metadata
        speculator._build_attn_metadata.assert_not_called()
        build_slots.assert_not_called()
        assert metadata.actual_seq_lengths_q == [4, 8]
        assert captured_slots.tolist() == [expected_slots + [-1] * 4]
        assert captured_blocks.tolist() == [request_blocks[req_idx], [0, 0, 0]]
        assert (captured_blocks.data_ptr(), captured_slots.data_ptr()) == (block_ptr, slot_ptr)


@pytest.mark.parametrize("guard", ["non_pcp", "no_batch", "no_metadata"])
def test_prepare_replicated_prefill_preserves_bypass(guard: str) -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.replicated_pcp = guard != "non_pcp"
    speculator.input_batch = _make_padded_input_batch() if guard != "no_batch" else None
    if speculator.input_batch is not None:
        speculator.input_batch.is_dummy = False
    speculator.block_tables = MagicMock()
    speculator._build_attn_metadata = MagicMock()
    metadata = None if guard == "no_metadata" else {"draft.layer": object()}
    slots = {"draft.layer": object()}

    actual_metadata, actual_slots = speculator._prepare_replicated_prefill_attn(
        metadata, slots, 2, 8, cudagraph_runtime_mode=CUDAGraphMode.FULL
    )

    assert actual_metadata is metadata
    assert actual_slots is slots
    speculator.block_tables.gather_block_tables.assert_not_called()
    speculator.block_tables.compute_slot_mappings.assert_not_called()
    speculator._build_attn_metadata.assert_not_called()


@pytest.mark.parametrize("attn_architecture", ["GQA", "SFA"])
def test_graph_prefill_uses_local_dummy_metadata(attn_architecture: str) -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.replicated_pcp = True
    speculator.attn_architecture = attn_architecture
    speculator.input_batch = _make_padded_input_batch()
    speculator.input_batch.is_dummy = True
    speculator.block_tables = MagicMock()
    speculator.kv_cache_config = object()
    target_metadata, draft_metadata = object(), object()
    speculator._build_attn_metadata = MagicMock(return_value={"draft.layer": draft_metadata})
    speculator.draft_attn_layer_names = {"draft.layer"}
    speculator.model_state = SimpleNamespace(attn_metadata={"draft.layer": target_metadata})

    with patch.object(speculator_module, "build_slot_mappings_by_layer") as build_slots:
        [actual] = speculator.build_draft_attn_metadatas(4, 8, is_draft_model_prefill=True)

    assert actual["draft.layer"] is draft_metadata
    speculator.block_tables.gather_block_tables.assert_not_called()
    speculator.block_tables.compute_slot_mappings.assert_not_called()
    speculator.block_tables.get_dummy_block_tables.assert_called_once_with(4)
    speculator.block_tables.get_dummy_slot_mappings.assert_called_once_with(8)
    build_slots.assert_called_once_with(
        speculator.block_tables.get_dummy_slot_mappings.return_value, speculator.kv_cache_config
    )
    kwargs = speculator._build_attn_metadata.call_args.kwargs
    assert kwargs["num_reqs"] == 2
    assert kwargs["step"] == 0
    assert kwargs["query_start_loc_np"] is speculator.input_batch.query_start_loc_np
    assert kwargs["seq_lens_cpu_upper_bound"] is speculator.input_batch.seq_lens_cpu_upper_bound


@pytest.mark.parametrize("replicated", [False, True])
def test_capture_preserves_prepared_prefill_inputs(replicated: bool) -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.replicated_pcp = replicated
    speculator.pcp_manager = object()
    speculator.model_state = SimpleNamespace(pcp_manager=speculator.pcp_manager)
    speculator.last_token_indices = torch.ones(2, dtype=torch.int32)
    speculator.num_speculative_steps = 1
    speculator.prefill_cudagraph_manager = MagicMock(use_breakable_cg=False)
    speculator.target_input_buffers = object()
    speculator.block_tables = object()
    speculator.attn_groups = []
    speculator.target_attn_groups = []
    speculator.kv_cache_config = object()
    speculator._prefill = MagicMock()
    metadata, slots = object(), object()

    def capture(forward, *args, **kwargs):
        metadata_context.__enter__.assert_called_once()
        metadata_context.__exit__.assert_not_called()
        assert (speculator.model_state.pcp_manager is None) is replicated
        forward(metadata, slots)

    speculator.prefill_cudagraph_manager.capture.side_effect = capture
    with (
        patch.object(speculator_module, "build_attn_metadata_wrapper") as metadata_wrapper,
        patch.object(speculator_module.AutoRegressiveSpeculator, "_prefill") as upstream_prefill,
    ):
        metadata_context = metadata_wrapper.return_value
        speculator.capture()
        chosen = upstream_prefill if replicated else speculator._prefill
        chosen.assert_called_once_with(metadata, slots)
        unused = speculator._prefill if replicated else upstream_prefill
        unused.assert_not_called()
    metadata_context.__exit__.assert_called_once()
    assert speculator.model_state.pcp_manager is speculator.pcp_manager


@pytest.mark.parametrize(
    ("speculator_cls", "parent_cls", "replicated_pcp", "batch_kind"),
    [
        (AscendMTPSpeculator, MTPSpeculator, True, "prefill"),
        (AscendMTPSpeculator, MTPSpeculator, True, "decode"),
        (AscendMTPSpeculator, MTPSpeculator, True, "idle"),
        (AscendMTPSpeculator, MTPSpeculator, False, "prefill"),
        (AscendEagleSpeculator, EagleSpeculator, True, "prefill"),
    ],
)
def test_propose_sync_follows_draft_token_layout(speculator_cls, parent_cls, replicated_pcp, batch_kind) -> None:
    speculator = object.__new__(speculator_cls)
    speculator.replicated_pcp = replicated_pcp
    speculator.input_batch = None
    speculator.pcp_manager = MagicMock()
    speculator.model_state = SimpleNamespace(
        pcp_manager=speculator.pcp_manager,
    )
    input_batch = _make_padded_input_batch()
    input_batch.has_prefill = batch_kind == "prefill"
    input_batch.is_dummy = batch_kind == "idle"
    if batch_kind != "prefill":
        input_batch.num_tokens_after_padding = 2
    num_tokens = input_batch.num_tokens_after_padding
    target_num_tokens = num_tokens // 2 if replicated_pcp and input_batch.has_prefill else num_tokens
    target_sync = SimpleNamespace(
        eager=True,
        uniform_token_count=None,
        num_tokens_across_dp=torch.tensor([target_num_tokens, target_num_tokens]),
    )
    expected = object()

    def parent_propose(*args, **kwargs):
        assert args[0] is input_batch
        assert (speculator.model_state.pcp_manager is None) is replicated_pcp
        # Exercise upstream's real reuse checks; only the collective is mocked.
        with patch.object(dp_utils, "sync_cudagraph_and_dp_padding") as sync:
            sync.return_value = (SimpleNamespace(cg_mode=CUDAGraphMode.NONE), object())
            dp_utils.dispatch_cg_and_sync_dp(
                None,
                input_batch.num_reqs,
                num_tokens,
                None,
                dp_size=2,
                dp_rank=0,
                need_eager=True,
                dp_sync=args[11],
            )
        assert sync.call_count == int(replicated_pcp)
        return expected

    with (
        patch.object(
            parent_cls,
            "propose",
            side_effect=parent_propose,
        ),
        patch.object(
            speculator_module,
            "build_attn_metadata_wrapper",
            return_value=nullcontext(),
        ),
        patch.object(
            speculator_module,
            "torch_gather_wrapper",
            return_value=nullcontext(),
        ),
    ):
        actual = speculator.propose(
            input_batch,
            *[MagicMock() for _ in range(10)],
            dp_sync=target_sync,
        )

    assert actual is expected
    assert speculator.input_batch is input_batch
    assert speculator.model_state.pcp_manager is speculator.pcp_manager


def test_propose_preserves_dp_sync_state() -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.replicated_pcp = False
    input_batch = object()
    dp_sync = object()
    with (
        patch.object(speculator_module, "disable_target_pcp_for_replicated_draft", return_value=nullcontext()),
        patch.object(speculator_module, "build_attn_metadata_wrapper", return_value=nullcontext()),
        patch.object(speculator_module, "torch_gather_wrapper", return_value=nullcontext()),
        patch.object(MTPSpeculator, "propose") as parent,
    ):
        speculator.propose(input_batch, *[MagicMock() for _ in range(10)], dp_sync)
    assert parent.call_args.args[11] is dp_sync


@pytest.mark.parametrize(("enabled", "legacy"), [(True, False), ("yes", True)])
def test_disable_profiling_chunk_for_draft_accepts_pydantic_true_values(enabled, legacy):
    profiling_chunk = {"enabled": enabled, "min_chunk": 128}
    if legacy:
        additional_config = {"profiling_chunk_config": profiling_chunk, "enable_cpu_binding": True}
    else:
        additional_config = {
            "scheduler_config": {"profiling_chunk_config": profiling_chunk},
            "enable_cpu_binding": True,
        }
    config = _config(additional_config)

    with disable_profiling_chunk_for_draft(config):
        draft_additional_config = config.additional_config
        draft_profiling_chunk = (
            draft_additional_config["profiling_chunk_config"]
            if legacy
            else draft_additional_config["scheduler_config"]["profiling_chunk_config"]
        )
        assert draft_profiling_chunk == {"enabled": False, "min_chunk": 128}
        assert draft_additional_config is not additional_config
        assert draft_profiling_chunk is not profiling_chunk

    assert config.additional_config is additional_config
    assert profiling_chunk["enabled"] == enabled


@pytest.mark.parametrize(("pp_size", "enabled"), [(1, True), (2, "off")])
def test_disable_profiling_chunk_for_draft_noop(pp_size, enabled):
    additional_config = {"profiling_chunk_config": {"enabled": enabled}}
    config = _config(additional_config, pp_size=pp_size)

    with disable_profiling_chunk_for_draft(config):
        assert config.additional_config is additional_config


def test_disable_profiling_chunk_for_draft_uses_nested_precedence():
    additional_config = {
        "scheduler_config": {"profiling_chunk_config": {"enabled": False}},
        "profiling_chunk_config": {"enabled": True},
    }
    config = _config(additional_config)

    with disable_profiling_chunk_for_draft(config):
        assert config.additional_config is additional_config


def test_disable_profiling_chunk_for_draft_restores_after_failure():
    additional_config = {"scheduler_config": {"profiling_chunk_config": {"enabled": True}}}
    config = _config(additional_config)
    expected_context = pytest.raises(RuntimeError, match="draft failed")

    with expected_context, disable_profiling_chunk_for_draft(config):
        assert config.additional_config is not additional_config
        raise RuntimeError("draft failed")

    assert config.additional_config is additional_config


def test_disable_profiling_chunk_for_draft_rejects_invalid_boolean():
    config = _config({"profiling_chunk_config": {"enabled": "sometimes"}})

    with (
        pytest.raises(ValueError, match="additional_config.profiling_chunk_config.enabled must be a boolean"),
        disable_profiling_chunk_for_draft(config),
    ):
        pass
