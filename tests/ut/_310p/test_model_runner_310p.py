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

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

import vllm_ascend._310p.model_runner_310p as model_runner_310p
from tests.ut.base import TestBase
from vllm_ascend._310p.model_runner_310p import (
    NPUModelRunner310,
    _resize_dflash_draft_kv_cache_specs,
    _snapshot_num_computed_tokens_to_device,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.spec_decode.utils import (
    update_num_computed_tokens_for_batch_change,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def _prepare_inputs_source() -> str:
    source_path = Path(__file__).resolve().parents[3] / "vllm_ascend" / "_310p" / "model_runner_310p.py"
    source = source_path.read_text(encoding="utf-8")
    start = source.index("    def _prepare_inputs(")
    end = source.index("    @torch.inference_mode()", start)
    return source[start:end]


def test_prepare_inputs_uses_device_metadata_after_async_correction() -> None:
    source = _prepare_inputs_source()

    assert "block_table.compute_slot_mapping(" in source
    assert "req_indices," in source
    assert "positions_np[:total_num_scheduled_tokens]" in source

    assert "self.input_batch.block_table.compute_slot_mapping(" not in source
    assert "query_start_loc.gpu[: num_reqs + 1]" not in source
    assert "use_async_device_metadata" in source
    assert "and not self.use_cp" in source
    assert "self.num_accepted_tokens.gpu.fill_(1)" in source
    assert "valid_sampled_token_count_gpu" in source
    assert "block_table.compute_slot_mapping_device(" in source
    assert "self.num_computed_tokens[req_indices_gpu]" in source

    assert "self.positions[:total_num_scheduled_tokens].copy_(" in source
    assert "self._positions_cpu_buf[:total_num_scheduled_tokens]" in source
    assert "self.seq_lens[:num_reqs] = self.num_computed_tokens" in source
    assert "self.optimistic_seq_lens_cpu[:num_reqs]" in source

    fallback_start = source.index("        if not use_async_device_metadata:")
    fallback_end = source.index(
        "        # Make sure when you update the positions and slot mapping",
        fallback_start,
    )
    fallback_source = source[fallback_start:fallback_end]
    assert "self.num_accepted_tokens_event.synchronize()" in fallback_source
    assert "update_num_computed_tokens_for_batch_change(" in fallback_source

    device_start = source.index("        if use_async_device_metadata:", fallback_end)
    device_end = source.index("        self.req_indices.np", device_start)
    device_source = source[device_start:device_end]
    assert device_source.count("update_num_computed_tokens_for_batch_change(") == 1
    assert "_snapshot_num_computed_tokens_to_device(" in device_source


def test_num_computed_tokens_snapshot_isolated_from_deferred_correction() -> None:
    num_computed_tokens_cpu = torch.tensor([119, 88], dtype=torch.int32)

    snapshot = _snapshot_num_computed_tokens_to_device(
        num_computed_tokens_cpu,
        torch.device("cpu"),
    )
    num_computed_tokens_cpu.sub_(torch.tensor([14, 7], dtype=torch.int32))

    torch.testing.assert_close(
        snapshot,
        torch.tensor([119, 88], dtype=torch.int32),
    )


def test_dflash_draft_kv_block_reuses_target_mamba_page() -> None:
    target_spec = AttentionSpec(
        block_size=1280,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.float16,
    )
    draft_spec = AttentionSpec(
        block_size=1280,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.float16,
    )
    mamba_spec = MambaSpec(
        block_size=1280,
        shapes=((18, 4096), (16, 128, 128)),
        dtypes=(torch.float16, torch.float32),
        mamba_cache_mode="align",
    )
    specs = {
        "target.attn": target_spec,
        "draft.attn": draft_spec,
        "target.linear_attn": mamba_spec,
    }

    resized = _resize_dflash_draft_kv_cache_specs(
        specs,
        {"draft.attn"},
        target_spec.page_size_bytes,
    )

    assert resized["target.attn"] is target_spec
    assert resized["draft.attn"].block_size == 640
    assert resized["draft.attn"].page_size_bytes == target_spec.page_size_bytes
    assert specs["draft.attn"].block_size == 1280


def test_model_forward_updates_mtp_full_graph_params_before_replay() -> None:
    runner = object.__new__(NPUModelRunner310)
    runner.uses_mrope = False
    runner.enable_enpu = False
    runner.speculative_config = SimpleNamespace(method="mtp")
    runner.update_stream = MagicMock()
    runner._all_gather_hidden_states_and_aux = MagicMock()

    calls = []

    def fake_update(*args):
        calls.append("update")

    def fake_model(**kwargs):
        calls.append("model")
        return torch.ones(1)

    runner.model = fake_model
    runner._update_full_graph_params_if_needed = fake_update
    forward_context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        capturing=False,
        flash_comm_v1_enabled=False,
    )
    current_stream = MagicMock()

    with (
        patch(
            "vllm_ascend._310p.model_runner_310p.get_forward_context",
            return_value=forward_context,
        ),
        patch(
            "vllm_ascend._310p.model_runner_310p.torch.npu.current_stream",
            return_value=current_stream,
        ),
    ):
        hidden_states = runner._model_forward(
            8,
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0]),
        )

    assert calls == ["update", "model"]
    current_stream.synchronize.assert_called_once_with()
    current_stream.wait_stream.assert_called_once_with(runner.update_stream)
    torch.testing.assert_close(hidden_states, torch.ones(1))


def _load_dflash_piecewise_scope_module():
    try:
        return importlib.import_module("vllm_ascend._310p.dflash_piecewise")
    except ModuleNotFoundError:
        pytest.fail("310P DFlash Piecewise scope module is missing")


@pytest.mark.parametrize(
    ("is_310p_platform", "method", "mode", "expected"),
    [
        (True, "dflash", CUDAGraphMode.PIECEWISE, True),
        (False, "dflash", CUDAGraphMode.PIECEWISE, False),
        (True, "mtp", CUDAGraphMode.PIECEWISE, False),
        (True, None, CUDAGraphMode.PIECEWISE, False),
        (True, "dflash", CUDAGraphMode.NONE, False),
        (True, "dflash", CUDAGraphMode.FULL, False),
        (True, "dflash", CUDAGraphMode.FULL_DECODE_ONLY, False),
        (True, "dflash", CUDAGraphMode.FULL_AND_PIECEWISE, False),
    ],
)
def test_dflash_piecewise_scope_requires_every_condition(
    is_310p_platform: bool,
    method: str | None,
    mode: CUDAGraphMode,
    expected: bool,
) -> None:
    scope_module = _load_dflash_piecewise_scope_module()
    config = SimpleNamespace(
        speculative_config=(SimpleNamespace(method=method) if method is not None else None),
        compilation_config=SimpleNamespace(
            cudagraph_mode=mode,
            cudagraph_capture_sizes=[64, 32],
        ),
    )

    with patch.object(scope_module, "is_310p", return_value=is_310p_platform):
        active = scope_module.is_310p_dflash_piecewise(config)

    assert active is expected
    assert config.compilation_config.cudagraph_mode is mode
    assert config.compilation_config.cudagraph_capture_sizes == [64, 32]


def test_dflash_piecewise_scope_has_no_sticky_process_state() -> None:
    scope_module = _load_dflash_piecewise_scope_module()
    piecewise = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash"),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE),
    )
    eager = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash"),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
    )

    with patch.object(scope_module, "is_310p", return_value=True):
        assert scope_module.is_310p_dflash_piecewise(piecewise) is True
        assert scope_module.is_310p_dflash_piecewise(eager) is False
        assert scope_module.is_310p_dflash_piecewise(piecewise) is True


@pytest.mark.parametrize(
    ("is_310p_platform", "method", "mode", "expected_force_eager"),
    [
        (True, "dflash", CUDAGraphMode.PIECEWISE, False),
        (False, "dflash", CUDAGraphMode.PIECEWISE, True),
        (True, "mtp", CUDAGraphMode.PIECEWISE, True),
        (True, "dflash", CUDAGraphMode.NONE, True),
        (True, "dflash", CUDAGraphMode.FULL, True),
        (True, "dflash", CUDAGraphMode.FULL_DECODE_ONLY, True),
        (True, "dflash", CUDAGraphMode.FULL_AND_PIECEWISE, False),
    ],
)
def test_dflash_piecewise_capable_modes_bypass_uniform_spec_guard(
    is_310p_platform: bool,
    method: str,
    mode: CUDAGraphMode,
    expected_force_eager: bool,
) -> None:
    scope_module = _load_dflash_piecewise_scope_module()
    runner = object.__new__(NPUModelRunner310)
    runner.attn_state = AscendAttentionState.DecodeOnly
    runner._spec_dummy_capture = False
    runner.speculative_config = SimpleNamespace(method=method)
    runner.uniform_decode_query_len = 16
    runner.input_batch = SimpleNamespace(num_computed_tokens_cpu=torch.zeros(16, dtype=torch.int32).numpy())
    runner.vllm_config = SimpleNamespace(
        speculative_config=runner.speculative_config,
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        additional_config={
            "ascend_compilation_config": {
                "dflash_full_and_piecewise_capture_config": {
                    "piecewise_capture_size": 64,
                    "full_capture_size": 160,
                },
            },
        },
    )
    observed_force_eager = []

    def parent_determine(self, **kwargs):
        observed_force_eager.append(kwargs["force_eager"])
        return "dispatch"

    with (
        patch.object(scope_module, "is_310p", return_value=is_310p_platform),
        patch(
            "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
            return_value=is_310p_platform,
        ),
        patch.object(
            model_runner_310p,
            "is_310p_dflash_full_decode_only",
            return_value=False,
        ),
        patch.object(
            NPUModelRunner,
            "_determine_batch_execution_and_padding",
            new=parent_determine,
        ),
    ):
        result = runner._determine_batch_execution_and_padding(
            num_tokens=64,
            num_reqs=16,
            num_scheduled_tokens_np=torch.full((16,), 4, dtype=torch.int32).numpy(),
            max_num_scheduled_tokens=64,
            use_cascade_attn=False,
        )

    assert result == "dispatch"
    assert observed_force_eager == [expected_force_eager]


@pytest.mark.parametrize(
    ("attn_state", "computed", "scheduled"),
    [
        (AscendAttentionState.PrefillNoCache, [0], [82]),
        (AscendAttentionState.ChunkedPrefill, [64], [64]),
        (AscendAttentionState.ChunkedPrefill, [24, 0], [16, 40]),
        (AscendAttentionState.SpecDecoding, [120, 80], [16, 16]),
    ],
)
def test_hybrid_target_phases_reach_existing_dispatcher_without_force_eager(
    attn_state,
    computed,
    scheduled,
):
    runner = object.__new__(NPUModelRunner310)
    runner.attn_state = attn_state
    runner._spec_dummy_capture = False
    runner.speculative_config = SimpleNamespace(
        method="dflash",
        num_speculative_tokens=15,
    )
    runner.uniform_decode_query_len = 16
    runner.max_num_reqs = 20
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=torch.tensor(computed, dtype=torch.int32).numpy(),
    )
    runner.vllm_config = SimpleNamespace(
        speculative_config=runner.speculative_config,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            max_cudagraph_capture_size=400,
        ),
        additional_config={
            "ascend_compilation_config": {
                "dflash_full_and_piecewise_capture_config": {
                    "piecewise_capture_size": 64,
                    "full_capture_size": 160,
                },
            },
        },
    )
    observed_force_eager = []
    selected = CUDAGraphMode.FULL if attn_state is AscendAttentionState.SpecDecoding else CUDAGraphMode.PIECEWISE

    def parent_determine(self, **kwargs):
        observed_force_eager.append(kwargs["force_eager"])
        descriptor = SimpleNamespace(
            num_tokens=max(sum(scheduled), 160),
            num_reqs=(len(scheduled) if selected is CUDAGraphMode.FULL else None),
        )
        return selected, descriptor

    with (
        patch(
            "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
            return_value=True,
        ),
        patch.object(
            model_runner_310p,
            "is_310p_dflash_full_decode_only",
            return_value=False,
        ),
        patch.object(
            NPUModelRunner,
            "_determine_batch_execution_and_padding",
            new=parent_determine,
        ),
    ):
        result = runner._determine_batch_execution_and_padding(
            num_tokens=sum(scheduled),
            num_reqs=len(scheduled),
            num_scheduled_tokens_np=torch.tensor(
                scheduled,
                dtype=torch.int32,
            ).numpy(),
            max_num_scheduled_tokens=max(scheduled),
            use_cascade_attn=False,
        )

    assert result[0] is selected
    assert observed_force_eager == [False]


def test_piecewise_dispatch_debug_handles_uninitialized_attention_state() -> None:
    scope_module = _load_dflash_piecewise_scope_module()
    runner = object.__new__(NPUModelRunner310)
    runner.attn_state = None
    runner.speculative_config = SimpleNamespace(method="dflash")
    runner.uniform_decode_query_len = 16
    runner.input_batch = SimpleNamespace(num_computed_tokens_cpu=torch.zeros(16, dtype=torch.int32).numpy())
    runner.vllm_config = SimpleNamespace(
        speculative_config=runner.speculative_config,
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE),
    )

    with (
        patch.object(scope_module, "is_310p", return_value=True),
        patch.object(
            NPUModelRunner,
            "_determine_batch_execution_and_padding",
            return_value="dispatch",
        ),
    ):
        result = runner._determine_batch_execution_and_padding(
            num_tokens=64,
            num_reqs=16,
            num_scheduled_tokens_np=torch.full((16,), 4, dtype=torch.int32).numpy(),
            max_num_scheduled_tokens=64,
            use_cascade_attn=False,
            force_uniform_decode=False,
        )

    assert result == "dispatch"


def test_async_correction_remaps_lengths_and_accepted_counts() -> None:
    num_computed_tokens = torch.tensor([10, 20, 0], dtype=torch.int32)
    num_accepted_tokens = torch.ones(3, dtype=torch.int32)
    prev_positions = torch.tensor([1, 0, -1], dtype=torch.int32)
    valid_sampled_token_count = torch.tensor([4, 2], dtype=torch.int32)
    prev_num_draft_tokens = torch.tensor([15, 15], dtype=torch.int32)
    cpu_num_computed_tokens = torch.tensor([21, 11, 5], dtype=torch.int32)

    update_num_computed_tokens_for_batch_change(
        num_computed_tokens,
        num_accepted_tokens,
        prev_positions,
        valid_sampled_token_count,
        prev_num_draft_tokens,
        cpu_num_computed_tokens,
    )

    torch.testing.assert_close(
        num_computed_tokens,
        torch.tensor([22, 14, 5], dtype=torch.int32),
    )
    torch.testing.assert_close(
        num_accepted_tokens,
        torch.tensor([2, 4, 1], dtype=torch.int32),
    )


class TestNPUModelRunner310(TestBase):
    def test_may_reinitialize_input_batch_expands_prefix_mamba_block_table(self):
        runner = object.__new__(NPUModelRunner310)
        runner.max_num_reqs = 8
        runner.max_model_len = 512
        runner.max_encoder_len = 0
        runner.max_num_tokens = 1024
        runner.device = torch.device("cpu")
        runner.pin_memory = False
        runner.is_pooling_model = False
        runner.model_config = SimpleNamespace(max_model_len=512, get_vocab_size=lambda: 32000)
        runner.cache_config = SimpleNamespace(block_size=128, enable_prefix_caching=True)
        runner.parallel_config = SimpleNamespace(cp_kv_cache_interleave_size=4)
        runner.vllm_config = SimpleNamespace(speculative_config=None)
        runner.offload_config = SimpleNamespace(uva=SimpleNamespace(cpu_offload_gb=0))
        runner.input_batch = SimpleNamespace(logitsprocs=MagicMock())
        attention_backend = SimpleNamespace(get_supported_kernel_block_sizes=lambda: [128, 64])
        runner.attn_groups = [[SimpleNamespace(backend=attention_backend)]]

        attention_spec = AttentionSpec(
            block_size=128,
            num_kv_heads=2,
            head_size=64,
            dtype=torch.float16,
        )
        mamba_spec = MambaSpec(
            block_size=128,
            shapes=((16,),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
            num_speculative_blocks=2,
        )
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=attention_spec),
                SimpleNamespace(kv_cache_spec=mamba_spec),
            ]
        )

        with (
            patch("vllm_ascend._310p.model_runner_310p.NPUInputBatch") as mock_input_batch,
            patch("vllm_ascend._310p.model_runner_310p.get_total_cp_world_size", return_value=1),
        ):
            runner.may_reinitialize_input_batch(kv_cache_config)

        kwargs = mock_input_batch.call_args.kwargs
        self.assertEqual(kwargs["block_sizes"], [128, 128])
        self.assertEqual(kwargs["kernel_block_sizes"], [[128, 64], [0]])
        self.assertEqual(kwargs["max_num_blocks_per_req"], [4, 6])
        self.assertIs(kwargs["kv_cache_groups"], kv_cache_config.kv_cache_groups)
        self.assertEqual(kwargs["cp_kv_cache_interleave_size"], 4)
