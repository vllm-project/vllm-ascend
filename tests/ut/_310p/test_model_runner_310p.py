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

import torch
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

from tests.ut.base import TestBase
from vllm_ascend._310p.model_runner_310p import NPUModelRunner310


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
