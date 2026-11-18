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

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.worker.dcp_utils import DCPManager
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.speculator import AscendTargetDependentARSpeculator


@pytest.mark.parametrize("step", [0, 1, 4])
@pytest.mark.parametrize("num_reqs,num_reqs_padded", [(0, 0), (0, 4), (2, 2), (2, 4)])
def test_draft_cpu_inputs_preserve_target_and_clear_padding(step, num_reqs, num_reqs_padded):
    config = SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1))
    manager = DCPManager(2, 1, config)
    target = torch.tensor([7, 127, 99, 99], dtype=torch.int32)
    prefilling = torch.tensor([True, False, True, True])
    original_target, original_flags = target.clone(), prefilling.clone()

    lengths, flags = manager.prepare_draft_dcp_metadata_inputs(
        target, prefilling, num_reqs, num_reqs_padded, step, max_model_len=128
    )

    expected = [min(length + step, 128) for length in target[:num_reqs].tolist()]
    assert lengths.tolist() == expected + [0] * (num_reqs_padded - num_reqs)
    expected_flags = prefilling[:num_reqs].tolist() if step == 0 else [False] * num_reqs
    assert flags.tolist() == expected_flags + [False] * (num_reqs_padded - num_reqs)
    assert lengths.dtype == torch.int32
    assert flags.dtype == torch.bool
    torch.testing.assert_close(target, original_target)
    torch.testing.assert_close(prefilling, original_flags)


@pytest.mark.parametrize(
    "cls,config_name,module",
    [
        (
            AscendTargetDependentARSpeculator,
            "draft_vllm_config",
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.speculator",
        ),
        (
            AscendDSparkSpeculator,
            "attn_vllm_config",
            "vllm_ascend.worker.v2.spec_decode.dspark.speculator",
        ),
    ],
)
@pytest.mark.parametrize("size", [1, 2, 8])
def test_speculator_initializes_only_dcp_host_state(monkeypatch, cls, config_name, module, size):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=size, cp_kv_cache_interleave_size=4)
    )
    speculator = object.__new__(cls)
    if cls is AscendDSparkSpeculator:
        monkeypatch.setattr(cls, config_name, property(lambda self: config))
    else:
        setattr(speculator, config_name, config)
    monkeypatch.setattr(f"{module}.get_dcp_group", lambda: SimpleNamespace(rank_in_group=1))

    speculator._init_dcp()

    assert speculator.use_dcp == (size > 1)
    if size == 1:
        assert speculator.dcp_manager is None
    else:
        manager = speculator.dcp_manager
        assert manager.vllm_config is config
        assert manager.dcp_world_rank == 1
        assert manager.dcp_world_size == size
        assert manager.prepare_dcp_local_seq_lens_cpu(torch.tensor([4, 8, 9])).tolist() == [0, 4, 4]
