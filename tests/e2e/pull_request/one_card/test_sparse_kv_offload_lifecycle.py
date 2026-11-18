# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise request ownership through the real eager TP1 runner and attention.

The tiny dummy-weight model needs no downloaded weights or tokenizer. The only
attention instrumentation selects every visible position, poisons retired
resident KV, and replays the same attention inputs with a cold resident row.
Request admission, completion/abort, metadata publication, host KV reload and
the attention kernel all use the serving implementation.
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch
from zlib import adler32

import pytest
import torch
import torch_npu
from vllm import SamplingParams, TokensPrompt

from tests.e2e.conftest import VllmRunner
from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    get_sparse_kv_offload_manager,
)

COLLIDING_IDS = ("234054828099429543", "471754017250656876")
PROMPT_TOKENS = 512
NUM_LAYERS = 2
MAX_STEPS = 32
LIVE_REQUEST_ID = "retained-live-control"
POISON_VALUE = 8.0
MODULE = "tests.e2e.pull_request.one_card.test_sparse_kv_offload_lifecycle"


@dataclass(frozen=True)
class DecodeObservation:
    request_id: str
    owner: int
    layer: int
    physical_row: int
    visible_tokens: int
    historical_misses: int


class AttentionProbe:
    def __init__(self, registry):
        self.registry = registry
        self.observations: list[DecodeObservation] = []
        self.target_id: str | None = None
        self.retired_owner: int | None = None
        self.retired_rows: dict[int, int] = {}
        self.compared_layers: set[int] = set()

    def observations_for(self, request_id):
        return [entry for entry in self.observations if entry.request_id == request_id]

    def arm(self, target_id, retired_owner, retired_rows):
        self.target_id = target_id
        self.retired_owner = retired_owner
        self.retired_rows = retired_rows
        self.compared_layers.clear()

    @staticmethod
    def _observe(manager, layer, request_ids, visible_lens):
        observations = []
        for row, (request_id, visible) in enumerate(zip(request_ids, visible_lens)):
            count = int(manager.lru_miss_count_cpu_list[layer][row])
            misses = manager.lru_miss_tokens_cpu_list[layer][row, :count]
            observations.append(
                DecodeObservation(
                    request_id=request_id,
                    owner=int(manager.lru_req_ids_cpu[row]),
                    layer=layer,
                    physical_row=int(manager.lru_physical_row_workspace[row]),
                    visible_tokens=visible,
                    historical_misses=int(((misses >= 0) & (misses < visible - 1)).sum()),
                )
            )
        return observations

    def execute(self, original, impl, q_nope, q_rope, topk, metadata, query_lens, key_lens, layer_name):
        manager = get_sparse_kv_offload_manager()
        layer = manager._get_offload_layer_id(layer_name)
        num_tokens = q_nope.shape[0]
        token_to_req = metadata.token_to_req[:num_tokens].cpu().long()
        owners = metadata.req_ids_tensor.cpu()[token_to_req].tolist()
        # Read the registry; never create/retire identities in the test harness.
        names_by_owner = {owner: name for name, owner in self.registry._owner_ids.items()}
        request_ids = [names_by_owner.get(owner, f"stale-owner-{owner}") for owner in owners]
        visible_lens = key_lens.cpu()[token_to_req].tolist()
        assert query_lens.cpu().tolist() == list(range(1, len(query_lens) + 1)), "Fixture requires one-token decode"

        # A dense selection makes stale historical slots observable even when
        # random dummy weights would otherwise choose a different index subset.
        selected = torch.full_like(topk.reshape(num_tokens, -1), -1)
        for row, visible in enumerate(visible_lens):
            assert visible <= selected.shape[1]
            selected[row, :visible] = torch.arange(visible, dtype=selected.dtype, device=selected.device)
        selected = selected.reshape_as(topk)

        compare = self.target_id in request_ids and layer not in self.compared_layers
        if compare:
            retired_row = self.retired_rows[layer]
            assert int(manager.lru_last_req_ids_cpu_list[layer][retired_row]) == self.retired_owner
            slot_tokens = manager.lru_slot_to_token_cpu_list[layer][retired_row]
            historical_slots = ((slot_tokens >= 0) & (slot_tokens < PROMPT_TOKENS)).nonzero().flatten()
            assert historical_slots.numel() == PROMPT_TOKENS
            manager.topk_buffers_k[layer][retired_row].index_fill_(
                0, historical_slots.to(device=q_nope.device, dtype=torch.long), POISON_VALUE
            )

        args = (q_nope, q_rope, selected, metadata, query_lens, key_lens, layer_name)
        output = original(impl, *args)
        observations = self._observe(manager, layer, request_ids, visible_lens)
        assert [entry.owner for entry in observations] == owners
        self.observations.extend(observations)

        if compare:
            row = request_ids.index(self.target_id)
            actual = output[row].detach().clone()
            observation = observations[row]
            assert observation.visible_tokens == PROMPT_TOKENS + 1
            # The live control occupies the other physical row, forcing the
            # resubmitted request onto the retired row without faking owners.
            assert observation.physical_row == self.retired_rows[layer]
            assert observation.historical_misses == PROMPT_TOKENS

            # Cold resident-cache reference with identical Q, TopK and host KV.
            # Only the target row is invalidated; the live control is retained.
            manager.lru_last_req_ids_cpu_list[layer][observation.physical_row] = -1
            # Different initial HBM contents prevent a missing reload in both
            # calls from accidentally producing the same poisoned output.
            manager.topk_buffers_k[layer][observation.physical_row].fill_(-POISON_VALUE)
            manager.topk_buffers_v[layer][observation.physical_row].zero_()
            reference = original(impl, *args)
            cold = self._observe(manager, layer, request_ids, visible_lens)[row]
            assert cold.historical_misses == PROMPT_TOKENS
            assert bool(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, reference[row], rtol=0, atol=0)
            self.compared_layers.add(layer)
        return output


def _step_until(engine, predicate):
    for _ in range(MAX_STEPS):
        outputs = engine.step()
        if predicate(outputs):
            return
    raise AssertionError(f"Runner did not satisfy the lifecycle condition in {MAX_STEPS} steps")


def _add_request(engine, request_id, prompt_token, max_tokens):
    internal_id = engine.add_request(
        request_id,
        TokensPrompt(prompt_token_ids=[prompt_token] * PROMPT_TOKENS),
        SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True, detokenize=False),
    )
    assert internal_id == request_id, "The fixture must exercise the colliding final worker IDs"


def _run_cases(engine, probe):
    assert {adler32(request_id.encode()) for request_id in COLLIDING_IDS} == {585106354}
    registry = probe.registry
    _add_request(engine, LIVE_REQUEST_ID, prompt_token=41, max_tokens=256)
    _step_until(engine, lambda _: len(probe.observations_for(LIVE_REQUEST_ID)) >= NUM_LAYERS * 2)
    live_owner = registry._owner_ids[LIVE_REQUEST_ID]
    first_id = COLLIDING_IDS[0]

    for finish_kind in ("completion", "abort"):
        for next_id in COLLIDING_IDS:
            probe.target_id = None
            probe.observations.clear()
            max_tokens = 4 if finish_kind == "completion" else 64
            _add_request(engine, first_id, prompt_token=42, max_tokens=max_tokens)
            _step_until(engine, lambda _: len(probe.observations_for(first_id)) >= NUM_LAYERS * 2)
            old_owner = registry._owner_ids[first_id]
            old_rows = {entry.layer: entry.physical_row for entry in probe.observations_for(first_id)}
            assert len(old_rows) == NUM_LAYERS

            if finish_kind == "completion":
                _step_until(engine, lambda outputs: any(out.request_id == first_id and out.finished for out in outputs))
            else:
                engine.abort_request([first_id])
            # Keep scheduling the live control so finished IDs reach the worker.
            _step_until(engine, lambda _: first_id not in registry._owner_ids)
            assert registry._owner_ids[LIVE_REQUEST_ID] == live_owner

            probe.arm(next_id, old_owner, old_rows)
            probe.observations.clear()
            _add_request(engine, next_id, prompt_token=43, max_tokens=4)
            _step_until(engine, lambda _: len(probe.compared_layers) == NUM_LAYERS)
            new_owner = registry._owner_ids[next_id]
            assert new_owner != old_owner
            _step_until(engine, lambda _, request_id=next_id: len(probe.observations_for(request_id)) >= NUM_LAYERS * 2)

            target_warm = probe.observations_for(next_id)[-NUM_LAYERS:]
            assert all(entry.owner == new_owner and entry.historical_misses == 0 for entry in target_warm)
            live = probe.observations_for(LIVE_REQUEST_ID)
            assert live and all(entry.owner == live_owner and entry.historical_misses == 0 for entry in live)
            assert registry._owner_ids[LIVE_REQUEST_ID] == live_owner
            print(
                json.dumps(
                    {
                        "finish": finish_kind,
                        "resubmitted_id": next_id,
                        "old_owner": old_owner,
                        "new_owner": new_owner,
                        "attention_layers_compared": len(probe.compared_layers),
                    }
                )
            )

            probe.target_id = None
            engine.abort_request([next_id])
            _step_until(engine, lambda _, request_id=next_id: request_id not in registry._owner_ids)
    engine.abort_request([LIVE_REQUEST_ID])


def _run_in_worker_process(model_path):
    with VllmRunner(
        model_path,
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=2,
        max_num_batched_tokens=2048,
        block_size=128,
        num_gpu_blocks_override=32,
        gpu_memory_utilization=0.1,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        async_scheduling=False,
        distributed_executor_backend="uni",
        seed=0,
        additional_config={
            "sparse_kv_offload_config": {
                "enabled": True,
                "use_fused_overlap": True,
                "keep_device_kv_cache": True,
                "topk_buffer_size": 4096,
                "dram_size_per_dp_GB": 1,
            }
        },
    ) as runner:
        engine = runner.model.llm_engine
        assert engine.vllm_config.use_v2_model_runner
        model_runner = engine.model_executor.driver_worker.model_runner
        state = model_runner.model_state
        probe = AttentionProbe(state._offload_request_identity)
        assert get_sparse_kv_offload_manager().max_num_topk_rows == 2
        original = AscendSFAKVOffloadImpl._execute_fused_overlap_offload_decode

        def execute(impl, *args, **kwargs):
            return probe.execute(original, impl, *args, **kwargs)

        with patch.object(AscendSFAKVOffloadImpl, "_execute_fused_overlap_offload_decode", execute):
            _run_cases(engine, probe)


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A3 or not torch_npu.npu.is_available(),
    reason="The real sparse-offload runner regression requires an Ascend A3 NPU",
)
def test_sparse_kv_offload_runner_request_lifecycle(tmp_path):
    model_path = tmp_path / "tiny-dsa"
    model_path.mkdir()
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "vocab_size": 256,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "moe_intermediate_size": 128,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "first_k_dense_replace": 1,
        "kv_lora_rank": 512,
        "q_lora_rank": 2048,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "index_topk": 2048,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_topk_pattern": "FF",
        "index_topk_freq": 1,
        "max_position_embeddings": 2048,
        "torch_dtype": "bfloat16",
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    }
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    env = {
        **os.environ,
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
        "OMP_NUM_THREADS": "8",
    }
    # Fresh interpreter: no inherited NPU context, manager singleton, or patches.
    subprocess.run(
        [sys.executable, "-m", MODULE, str(model_path)],
        cwd=Path(__file__).resolve().parents[4],
        env=env,
        timeout=600,
        check=True,
    )


if __name__ == "__main__":
    _run_in_worker_process(sys.argv[1])
