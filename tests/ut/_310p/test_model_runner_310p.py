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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

from tests.ut.base import TestBase
from vllm_ascend._310p.attention.attention_v1 import AscendAttentionBackend310
from vllm_ascend._310p.block_table import MultiGroupBlockTable as MultiGroupBlockTable310
from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
from vllm_ascend.attention.attention_v1 import AscendAttentionState


def _initialize_310p_batch_table(**kwargs):
    return SimpleNamespace(
        logitsprocs=kwargs["logitsprocs"],
        block_table=MultiGroupBlockTable310(
            max_num_reqs=kwargs["max_num_reqs"],
            max_model_len=kwargs["max_model_len"],
            max_num_batched_tokens=kwargs["max_num_batched_tokens"],
            pin_memory=kwargs["pin_memory"],
            device=kwargs["device"],
            block_sizes=kwargs["block_sizes"],
            max_num_blocks=kwargs["max_num_blocks_per_req"],
            kernel_sizes=kwargs["kernel_block_sizes"],
            kv_cache_groups=kwargs["kv_cache_groups"],
        ),
    )


def _prepare_inputs_source() -> str:
    source_path = Path(__file__).resolve().parents[3] / "vllm_ascend" / "_310p" / "model_runner_310p.py"
    source = source_path.read_text(encoding="utf-8")
    start = source.index("    def _prepare_inputs(")
    end = source.index("    @torch.inference_mode()", start)
    return source[start:end]


def test_prepare_inputs_keeps_aclgraph_metadata_on_cpu() -> None:
    source = _prepare_inputs_source()

    assert "block_table.compute_slot_mapping(" in source
    assert "req_indices," in source
    assert "positions_np[:total_num_scheduled_tokens]" in source

    assert "self.input_batch.block_table.compute_slot_mapping(" not in source
    assert "query_start_loc.gpu[: num_reqs + 1]" not in source
    assert "req_indices_gpu" not in source
    assert "self.num_computed_tokens[req_indices_gpu]" not in source

    assert "self.positions[:total_num_scheduled_tokens].copy_(" in source
    assert "self._positions_cpu_buf[:total_num_scheduled_tokens]" in source
    assert "self.seq_lens[:num_reqs].copy_(" in source
    assert "self.optimistic_seq_lens_cpu[:num_reqs]" in source
    assert "self._sync_num_accepted_tokens(" in source
    assert "self.input_batch.num_accepted_tokens_cpu[" not in source


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
        assert "is_dummy_run" not in kwargs
        calls.append("model")
        return torch.ones(1)

    runner.model = fake_model
    runner._update_full_graph_params_if_needed = fake_update
    forward_context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        capturing=False,
    )

    with patch(
        "vllm_ascend._310p.model_runner_310p.get_forward_context",
        return_value=forward_context,
    ):
        hidden_states = runner._model_forward(
            8,
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0]),
        )

    assert calls == ["update", "model"]
    torch.testing.assert_close(hidden_states, torch.ones(1))


def test_310p_runner_does_not_advertise_standardized_shared_kv_backing() -> None:
    assert NPUModelRunner310.supports_standardized_shared_kv_backing is False


def test_graph_dispatch_does_not_treat_later_prefill_chunk_as_decode() -> None:
    runner = object.__new__(NPUModelRunner310)
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([8], dtype=np.int32),
        num_prompt_tokens=np.array([16], dtype=np.int32),
    )
    runner.attn_state = AscendAttentionState.DecodeOnly
    runner.speculative_config = None

    with patch(
        "vllm_ascend.worker.model_runner_v1.NPUModelRunner._determine_batch_execution_and_padding",
        return_value="dispatch-result",
    ) as parent_dispatch:
        result = runner._determine_batch_execution_and_padding(
            num_tokens=1,
            num_reqs=1,
            num_scheduled_tokens_np=np.array([1], dtype=np.int32),
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
        )

    assert result == "dispatch-result"
    assert parent_dispatch.call_args.kwargs["force_uniform_decode"] is None


@pytest.mark.parametrize(
    ("physical_block_size", "head_size", "expected_candidates"),
    [(64, 128, [64]), (128, 128, [128, 64]), (256, 128, [128, 64]), (128, 256, [64]), (256, 256, [64])],
)
def test_reinitialized_310p_batch_uses_compatible_kernel_sizes(physical_block_size, head_size, expected_candidates):
    runner = object.__new__(NPUModelRunner310)
    runner.max_num_reqs = 2
    runner.max_model_len = 512
    runner.max_encoder_len = 0
    runner.max_num_tokens = 64
    runner.device = torch.device("cpu")
    runner.pin_memory = False
    runner.is_pooling_model = False
    runner.model_config = SimpleNamespace(get_vocab_size=lambda: 32)
    runner.cache_config = SimpleNamespace(block_size=32, enable_prefix_caching=False)
    runner.parallel_config = SimpleNamespace(cp_kv_cache_interleave_size=1)
    runner.vllm_config = SimpleNamespace(speculative_config=None)
    runner.offload_config = SimpleNamespace(uva=SimpleNamespace(cpu_offload_gb=0))
    runner.input_batch = SimpleNamespace(
        logitsprocs=MagicMock(),
        block_table=SimpleNamespace(block_tables=[SimpleNamespace(physical_block_size=32, kernel_sizes=[32])]),
    )
    backend = SimpleNamespace(get_supported_kernel_block_sizes=lambda: [128, 64])
    runner.attn_groups = [[SimpleNamespace(backend=backend)]]
    spec = AttentionSpec(block_size=physical_block_size, num_kv_heads=1, head_size=head_size, dtype=torch.float16)
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)])

    with (
        patch("vllm_ascend._310p.model_runner_310p.NPUInputBatch", side_effect=_initialize_310p_batch_table) as batch,
        patch("vllm_ascend._310p.model_runner_310p.get_decode_context_model_parallel_world_size", return_value=1),
        patch(
            "vllm_ascend.worker.block_table.get_dcp_group",
            return_value=SimpleNamespace(world_size=1, rank_in_group=0),
        ),
    ):
        runner.may_reinitialize_input_batch(config, [expected_candidates[0]])

    assert batch.call_args.kwargs["kernel_block_sizes"] == [expected_candidates]
    table = runner.input_batch.block_table[0]
    assert table.physical_block_size == physical_block_size
    assert table.block_size == expected_candidates[0]
    assert table.blocks_per_phys_block == physical_block_size // expected_candidates[0]


@pytest.mark.parametrize(
    ("initial_block_size", "initial_kernel_sizes", "block_size", "head_size", "expected_kernel_sizes", "reinitialize"),
    [
        (128, [128], 64, 128, [64], True),
        (64, [64], 64, 128, [64], False),
        (128, [128], 128, 256, [64], True),
        (128, [64], 128, 256, [64], False),
        (256, [128, 64], 256, 128, [128, 64], False),
    ],
)
def test_310p_batch_compares_existing_table_layout(
    initial_block_size, initial_kernel_sizes, block_size, head_size, expected_kernel_sizes, reinitialize
):
    runner = object.__new__(NPUModelRunner310)
    runner.max_num_reqs = 2
    runner.max_model_len = 512
    runner.max_encoder_len = 0
    runner.max_num_tokens = 64
    runner.device = torch.device("cpu")
    runner.pin_memory = False
    runner.is_pooling_model = False
    runner.model_config = SimpleNamespace(get_vocab_size=lambda: 32)
    runner.cache_config = SimpleNamespace(block_size=block_size, enable_prefix_caching=False)
    runner.parallel_config = SimpleNamespace(cp_kv_cache_interleave_size=1)
    runner.vllm_config = SimpleNamespace(speculative_config=None)
    runner.offload_config = SimpleNamespace(uva=SimpleNamespace(cpu_offload_gb=0))
    backend = SimpleNamespace(get_supported_kernel_block_sizes=lambda: [128, 64])
    runner.attn_groups = [[SimpleNamespace(backend=backend)]]
    spec = AttentionSpec(block_size=block_size, num_kv_heads=1, head_size=head_size, dtype=torch.float16)
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)])

    with (
        patch("vllm_ascend._310p.model_runner_310p.NPUInputBatch", side_effect=_initialize_310p_batch_table) as batch,
        patch("vllm_ascend._310p.model_runner_310p.get_decode_context_model_parallel_world_size", return_value=1),
        patch("vllm_ascend._310p.block_table.get_decode_context_model_parallel_world_size", return_value=1),
        patch(
            "vllm_ascend.worker.block_table.get_dcp_group",
            return_value=SimpleNamespace(world_size=1, rank_in_group=0),
        ),
    ):
        runner.input_batch = _initialize_310p_batch_table(
            max_num_reqs=runner.max_num_reqs,
            max_model_len=runner.max_model_len,
            max_num_batched_tokens=runner.max_num_tokens,
            pin_memory=runner.pin_memory,
            device=runner.device,
            block_sizes=[initial_block_size],
            kernel_block_sizes=[initial_kernel_sizes],
            max_num_blocks_per_req=None,
            kv_cache_groups=None,
            logitsprocs=MagicMock(),
        )
        original_batch = runner.input_batch
        runner.may_reinitialize_input_batch(config, [expected_kernel_sizes[0]])

    assert batch.call_count == int(reinitialize)
    assert (runner.input_batch is not original_batch) is reinitialize
    table = runner.input_batch.block_table[0]
    assert table.physical_block_size == block_size
    assert table.kernel_sizes == expected_kernel_sizes
    assert table.block_size == expected_kernel_sizes[0]
    assert table.blocks_per_phys_block == block_size // expected_kernel_sizes[0]
    block_ids = np.array([3, 7], dtype=np.int32)
    positions = np.array([0, table.block_size - 1, table.block_size, block_size - 1, block_size, block_size + 1])
    table.append_row(block_ids.tolist(), 0)
    table.compute_slot_mapping(np.zeros(len(positions), dtype=np.int64), positions)
    expected_slots = block_ids[positions // block_size] * block_size + positions % block_size
    np.testing.assert_array_equal(table.slot_mapping.np[: len(positions)], expected_slots)


@pytest.mark.parametrize(
    ("physical_block_size", "head_size", "expected_kernel_size"),
    [(64, 128, 64), (128, 128, 128), (256, 128, 128), (128, 256, 64), (256, 256, 64)],
)
def test_allocate_310p_kv_cache_uses_compatible_kernel_sizes(physical_block_size, head_size, expected_kernel_size):
    runner = object.__new__(NPUModelRunner310)
    runner.cache_config = SimpleNamespace()
    runner.runner_only_attn_layers = set()
    runner.device = torch.device("cpu")
    runner._acl_format = 29
    runner.attn_backend = AscendAttentionBackend310
    layer_names = ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"]
    num_blocks = 3
    num_kv_heads = 8
    spec = AttentionSpec(
        block_size=physical_block_size, num_kv_heads=num_kv_heads, head_size=head_size, dtype=torch.float16
    )
    config = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[SimpleNamespace(layer_names=layer_names, kv_cache_spec=spec)],
        kv_cache_tensors=[
            SimpleNamespace(size=num_blocks * spec.page_size_bytes * len(layer_names), layers=layer_names)
        ],
    )
    with patch(
        "vllm_ascend._310p.model_runner_310p.torch_npu.empty_with_format",
        side_effect=lambda size, dtype, device, acl_format: torch.empty(size, dtype=dtype, device=device),
        create=True,
    ) as allocate:
        caches = runner._allocate_kv_cache_tensors(config)

    expected_shape = (
        num_blocks * (physical_block_size // expected_kernel_size),
        num_kv_heads * head_size // 16,
        expected_kernel_size,
        16,
    )
    assert set(caches) == set(layer_names)
    assert allocate.call_count == 2 * len(layer_names)
    for call in allocate.call_args_list:
        assert call.kwargs["acl_format"] == runner._acl_format
    for key_cache, value_cache in caches.values():
        for cache in (key_cache, value_cache):
            assert tuple(cache.shape) == expected_shape
            assert cache.numel() == num_blocks * physical_block_size * num_kv_heads * head_size
            assert cache.dtype == spec.dtype
        assert key_cache.data_ptr() != value_cache.data_ptr()
    assert caches[layer_names[0]][0].data_ptr() != caches[layer_names[1]][0].data_ptr()


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
        runner.input_batch = SimpleNamespace(
            logitsprocs=MagicMock(),
            block_table=SimpleNamespace(block_tables=[SimpleNamespace(physical_block_size=128, kernel_sizes=[128])]),
        )
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
            patch(
                "vllm_ascend._310p.model_runner_310p.get_decode_context_model_parallel_world_size",
                return_value=1,
            ),
        ):
            runner.may_reinitialize_input_batch(kv_cache_config)

        kwargs = mock_input_batch.call_args.kwargs
        self.assertEqual(kwargs["block_sizes"], [128, 128])
        self.assertEqual(kwargs["kernel_block_sizes"], [[128, 64], [0]])
        self.assertEqual(kwargs["max_num_blocks_per_req"], [4, 6])
        self.assertIs(kwargs["kv_cache_groups"], kv_cache_config.kv_cache_groups)
        self.assertEqual(kwargs["cp_kv_cache_interleave_size"], 4)
