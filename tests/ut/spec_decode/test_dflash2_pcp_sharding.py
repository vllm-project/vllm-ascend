# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tensor coverage for DFlash2 PCP query packing and token exchange.

The packing module has no worker dependencies. Loading it directly also lets
these tests run on hosts without vLLM, torch_npu or the repository UT stubs:
pytest --noconftest --import-mode=importlib <this file>
"""

import importlib.util
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def buffers_cls():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/dflash2/pcp.py"
    module_spec = importlib.util.spec_from_file_location("dflash2_pcp_buffers", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.PCPQueryBuffers


def make_spec(num_reqs, steps=2, capacity=40):
    capacity = max(capacity, num_reqs)
    width = steps + 1
    tokens = capacity * width
    # Deliberately non-contiguous persistent request slots.
    slots = torch.arange(capacity, dtype=torch.int64).flip(0) * 3 + 7
    inputs = SimpleNamespace(
        max_num_reqs=capacity,
        max_num_tokens=tokens,
        input_ids=torch.arange(tokens, dtype=torch.int32),
        positions=torch.arange(tokens, dtype=torch.int64) + 100,
        is_padding=torch.zeros(tokens, dtype=torch.bool),
        seq_lens=torch.arange(capacity, dtype=torch.int32) * 4 + width,
        dcp_local_seq_lens=torch.zeros(capacity, dtype=torch.int32),
        query_start_loc=torch.arange(capacity + 1, dtype=torch.int32) * width,
    )
    return SimpleNamespace(
        max_num_reqs=capacity,
        num_query_per_req=width,
        num_speculative_steps=steps,
        input_buffers=inputs,
        sample_indices=torch.zeros(capacity * steps, dtype=torch.int64),
        sample_pos=torch.arange(capacity * steps, dtype=torch.int64) + 200,
        sample_idx_mapping=slots.repeat_interleave(steps),
        draft_tokens=torch.full((capacity, steps), -99, dtype=torch.int64),
        draft_kv_cache_group_ids=[1, 2],
        block_tables=SimpleNamespace(
            input_block_tables=[torch.arange(capacity * 5).view(capacity, 5) + gid * 1000 for gid in range(3)],
            slot_mappings=torch.arange(3 * tokens).view(3, tokens),
        ),
        input_batch=SimpleNamespace(num_reqs=num_reqs, idx_mapping=slots[:num_reqs]),
    )


@pytest.mark.parametrize("num_reqs", [0, 1, 7, 8, 9, 32, 64])
@pytest.mark.parametrize("steps", [1, 2, 4, 7])
def test_query_blocks_preserve_request_state_and_shared_inputs(buffers_cls, num_reqs, steps):
    spec = make_spec(num_reqs, steps)
    width = spec.num_query_per_req
    original_slots = spec.block_tables.slot_mappings.clone()
    original_ids = spec.input_buffers.input_ids.clone()
    original_tables = [table.clone() for table in spec.block_tables.input_block_tables]
    owners = []
    for rank in range(8):
        buffers = buffers_cls(spec, rank, 8)
        n = buffers.pack(spec, num_reqs)
        selected = list(range(rank, num_reqs, 8))
        owners.extend(selected)
        assert n == len(selected)
        assert buffers.draft_tokens[: (num_reqs + 7) // 8].eq(-1).all()
        for name in ("input_ids", "positions", "is_padding"):
            expected = getattr(spec.input_buffers, name)[: num_reqs * width].view(num_reqs, width)[selected]
            torch.testing.assert_close(getattr(buffers.input_buffers, name)[: n * width].view(n, width), expected)
        torch.testing.assert_close(buffers.input_buffers.seq_lens[:n], spec.input_buffers.seq_lens[selected])
        torch.testing.assert_close(
            buffers.input_buffers.query_start_loc[: n + 1], torch.arange(n + 1, dtype=torch.int32) * width
        )
        for name in ("sample_pos", "sample_idx_mapping"):
            expected = getattr(spec, name)[: num_reqs * steps].view(num_reqs, steps)[selected]
            torch.testing.assert_close(getattr(buffers, name)[: n * steps].view(n, steps), expected)
        indices = buffers.sample_indices[: n * steps].view(n, steps)
        torch.testing.assert_close(indices, torch.arange(n)[:, None] * width + torch.arange(1, steps + 1))
        for gid in spec.draft_kv_cache_group_ids:
            torch.testing.assert_close(buffers.block_tables.input_block_tables[gid][:n], original_tables[gid][selected])
            expected = original_slots[gid, : num_reqs * width].view(num_reqs, width)[selected]
            torch.testing.assert_close(buffers.block_tables.slot_mappings[gid, : n * width].view(n, width), expected)
    assert sorted(owners) == list(range(num_reqs))
    torch.testing.assert_close(spec.input_buffers.input_ids, original_ids)
    torch.testing.assert_close(spec.block_tables.slot_mappings, original_slots)
    for table, original in zip(spec.block_tables.input_block_tables, original_tables):
        torch.testing.assert_close(table, original)


@pytest.mark.parametrize("num_reqs", [0, 1, 7, 8, 9, 32, 64])
def test_all_ranks_collect_disjoint_proposals_in_global_order(buffers_cls, num_reqs):
    world_size = 8
    barrier = Barrier(world_size, timeout=30)
    transmitted = [None] * world_size
    counts = [0] * world_size

    def rank_forward(rank):
        spec = make_spec(num_reqs)
        buffers = buffers_cls(spec, rank, world_size)
        count = buffers.pack(spec, num_reqs)
        counts[rank] = count
        if count:
            # A deterministic candidate function that depends on request state
            # and positions, not rank or local batch-row numbering.
            buffers.draft_tokens[:count].copy_(
                buffers.sample_idx_mapping[: count * 2].view(count, 2) + buffers.sample_pos[: count * 2].view(count, 2)
            )

        def gather(tensor, dim):
            assert dim == 0
            transmitted[rank] = tensor.clone()
            barrier.wait()
            return torch.cat(transmitted, dim=0)

        buffers.gather(SimpleNamespace(all_gather=gather), spec.draft_tokens, num_reqs)
        expected = (spec.sample_idx_mapping + spec.sample_pos)[: num_reqs * 2].view(num_reqs, 2)
        torch.testing.assert_close(spec.draft_tokens[:num_reqs], expected)
        assert spec.draft_tokens[num_reqs:].eq(-99).all()

    with ThreadPoolExecutor(max_workers=world_size) as executor:
        list(executor.map(rank_forward, range(world_size)))
    assert sum(counts) == num_reqs
    assert max(counts) - min(counts) <= 1
    if num_reqs:
        assert all(tensor.shape == ((num_reqs + 7) // 8, 2) for tensor in transmitted)
    else:
        assert transmitted == [None] * world_size


def test_reused_buffers_follow_new_owners_and_clear_old_proposals(buffers_cls):
    spec = make_spec(32)
    buffers = buffers_cls(spec, 1, 8)
    pointers = (buffers.input_buffers.input_ids.data_ptr(), buffers.draft_tokens.data_ptr())
    buffers.pack(spec, 32)
    buffers.draft_tokens.fill_(1234)
    # Reordering/slot reuse changes ownership and physical KV blocks.
    spec.sample_idx_mapping.copy_(spec.sample_idx_mapping.flip(0))
    spec.input_buffers.positions.add_(50)
    spec.block_tables.input_block_tables[1].add_(123)
    count = buffers.pack(spec, 9)
    assert count == 1
    assert buffers.draft_tokens[:2].eq(-1).all()
    torch.testing.assert_close(buffers.sample_idx_mapping[:2], spec.sample_idx_mapping[2:4])
    torch.testing.assert_close(buffers.input_buffers.positions[:3], spec.input_buffers.positions[3:6])
    torch.testing.assert_close(
        buffers.block_tables.input_block_tables[1][0], spec.block_tables.input_block_tables[1][1]
    )
    assert pointers == (buffers.input_buffers.input_ids.data_ptr(), buffers.draft_tokens.data_ptr())


@pytest.mark.parametrize("fail", [False, True])
def test_query_scope_restores_global_references_even_on_failure(buffers_cls, fail):
    spec = make_spec(9)
    originals = vars(spec).copy()
    buffers = buffers_cls(spec, 0, 8)
    buffers.pack(spec, 9)
    try:
        with buffers.activate(spec):
            assert spec.input_buffers is buffers.input_buffers
            assert spec.block_tables is buffers.block_tables
            assert spec.input_batch is originals["input_batch"]
            if fail:
                raise RuntimeError("candidate forward failed")
    except RuntimeError:
        assert fail
    for name, value in originals.items():
        assert getattr(spec, name) is value


@pytest.fixture
def worker_module(monkeypatch, buffers_cls):
    """Load the real extension with small upstream interfaces, no NPU runtime.

    These doubles exercise the extension's dispatch/metadata/synchronization
    contract; they do not claim coverage of upstream attention kernels or KV.
    """

    class Base:
        def load_model(self, target):
            pass

    class Draft(Base):
        def _generate_draft(self, n, tokens, metadata, slots, dp, mode):
            self.generated.append(n)
            assert tokens == n * self.num_query_per_req
            self.draft_tokens[:n].copy_(
                (self.sample_pos + self.sample_idx_mapping)[: n * self.num_speculative_steps].view(n, -1)
            )

    class Ascend(Base):
        def propose(self, batch, *args, **kwargs):
            self.parent_batch_sizes.append(batch.num_reqs)
            if self.fail_parent:
                raise RuntimeError("context update failed")
            self.input_batch = batch
            if kwargs.get("num_speculative_tokens") == 0:
                return self.draft_tokens[: batch.num_reqs]
            self._generate_draft(batch.num_reqs, batch.num_reqs * self.num_query_per_req, {}, {}, None)
            return self.draft_tokens[: batch.num_reqs]

    class Model:
        pass

    vocab = SimpleNamespace(STANDARD="standard", REPLICATED="replicated", PCP_X_TP="pcp", FINE_GRAINED="fine")
    cg = SimpleNamespace(NONE="none", FULL="full")

    # Both v0.33.0 and the verified upstream main helper take two arguments.
    def slots_by_layer(slots, config):
        return {"draft": slots[1]}

    stubs = {
        "vllm.config.compilation": {"CUDAGraphMode": cg},
        "vllm.distributed": {"get_pcp_group": lambda: None},
        "vllm.logger": {"logger": logging.getLogger("dflash2_pcp_test")},
        "vllm.triton_utils": {"tl": SimpleNamespace(constexpr=int), "triton": SimpleNamespace(jit=lambda f: f)},
        "vllm.v1.worker.gpu.attn_utils": {"build_slot_mappings_by_layer": slots_by_layer},
        "vllm.v1.worker.gpu.cudagraph_utils": {"BatchExecutionDescriptor": SimpleNamespace},
        "vllm.v1.worker.gpu.sample.gumbel": {"_DRAFT_NOISE_SALT": 0, "_log1p_neg_stable": None, "tl_rand32": None},
        "vllm.v1.worker.gpu.spec_decode.dflash2.speculator": {"DFlash2Speculator": Draft},
        "vllm_ascend.ascend_config": {"validate_additional_config_bool": lambda value, path: value},
        "vllm_ascend.models.qwen3_dflash2": {"DFlash2Qwen3ForCausalLM": Model},
        "vllm_ascend.ops.vocab_parallel_embedding": {"VocabParallelMode": vocab},
        "vllm_ascend.worker.v2.spec_decode.dflash.speculator": {"AscendDFlashSpeculator": Ascend},
        "vllm_ascend.worker.v2.spec_decode.dflash2.pcp": {"PCPQueryBuffers": buffers_cls},
    }
    for name, attrs in stubs.items():
        parts = name.split(".")
        for end in range(1, len(parts)):
            parent = ".".join(parts[:end])
            if parent not in sys.modules:
                package = ModuleType(parent)
                package.__path__ = []
                monkeypatch.setitem(sys.modules, parent, package)
        module = ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/dflash2/speculator.py"
    module_spec = importlib.util.spec_from_file_location("dflash2_pcp_worker", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def make_worker(module, buffers_cls, n, rank=0):
    spec = object.__new__(module.AscendDFlash2Speculator)
    vars(spec).update(vars(make_spec(n)))
    spec._pcp_sharding_enabled = True
    spec._pcp_real_proposal = False
    spec._pcp_sharding_requested = True
    spec._pcp_query_buffers = buffers_cls(spec, rank, 8)
    spec._group_causal = {1: False, 2: True}
    spec.kv_cache_config = None
    spec.attn_groups = None
    spec.generated = []
    spec.parent_batch_sizes = []
    spec.fail_parent = False
    spec.input_batch.seq_lens_cpu_upper_bound = torch.arange(40, dtype=torch.int32) + 500
    spec._pcp_target_config = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(enforce_eager=True),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            pipeline_parallel_size=2,
            prefill_context_parallel_size=8,
        ),
    )
    spec.speculative_config = SimpleNamespace(enforce_eager=True, draft_sample_method="greedy")
    spec.draft_logits = None
    spec.model = module.DFlash2Qwen3ForCausalLM()
    spec.model.model = SimpleNamespace(embed_tokens=SimpleNamespace(tp_size=1, parallel_mode="standard"))
    spec.model.lm_head = SimpleNamespace(tp_size=1, parallel_mode="replicated")
    return spec


@pytest.mark.parametrize("n,rank", [(32, 0), (32, 7), (9, 0), (9, 7), (1, 7), (0, 0)])
def test_worker_only_runs_owned_queries_after_global_parent(worker_module, buffers_cls, n, rank):
    spec = make_worker(worker_module, buffers_cls, n, rank)
    originals = spec.input_buffers, spec.block_tables, spec.draft_tokens
    expected = (spec.sample_pos + spec.sample_idx_mapping)[: n * 2].view(n, 2)
    count = len(range(rank, n, 8))
    padded = (n + 7) // 8
    gathered = torch.full((8, padded, 2), -1, dtype=torch.int64)
    for r in range(8):
        part = expected[r::8]
        gathered[r, : len(part)].copy_(part)
    calls = []

    def gather(local, dim):
        calls.append(local.clone())
        torch.testing.assert_close(local[:count], expected[rank::8])
        return gathered.flatten(0, 1)

    def metadata(**kwargs):
        assert spec.input_buffers is not originals[0]
        assert spec.block_tables is not originals[1]
        assert kwargs["num_reqs"] == count
        assert kwargs["batch_desc"].num_tokens == count * 3
        assert kwargs["causal"] == {1: False, 2: True}
        torch.testing.assert_close(
            kwargs["seq_lens_cpu_upper_bound"], spec.input_batch.seq_lens_cpu_upper_bound[rank:n:8]
        )
        return {"draft": SimpleNamespace(actual_seq_lengths_q=None)}

    spec._build_uniform_attn_metadata = metadata
    spec._update_draft_attn_metadata = lambda md, n: setattr(
        md["draft"], "actual_seq_lengths_q", list(range(3, n * 3 + 1, 3))
    )
    spec._pcp_query_group = SimpleNamespace(all_gather=gather)
    result = spec.propose(spec.input_batch, *[None] * 10)
    torch.testing.assert_close(result, expected)
    assert spec.generated == ([count] if count else [])
    assert spec.parent_batch_sizes == ([n] if n else [])
    assert len(calls) == int(n > 0)
    assert not spec._pcp_real_proposal
    assert (spec.input_buffers, spec.block_tables, spec.draft_tokens) == originals


@pytest.mark.parametrize("mode", ["disabled", "dummy", "profile"])
def test_worker_preserves_replicated_path(worker_module, buffers_cls, mode):
    spec = make_worker(worker_module, buffers_cls, 9)
    spec._pcp_sharding_enabled = mode != "disabled"
    spec.propose(spec.input_batch, *[None] * 10, dummy_run=mode == "dummy", is_profile=mode == "profile")
    assert spec.generated == [9]
    assert spec.parent_batch_sizes == [9]
    assert not spec._pcp_real_proposal


def test_parent_exception_resets_proposal_scope(worker_module, buffers_cls):
    spec = make_worker(worker_module, buffers_cls, 9)
    spec.fail_parent = True
    with pytest.raises(RuntimeError, match="context update failed"):
        spec.propose(spec.input_batch, *[None] * 10)
    assert not spec._pcp_real_proposal


def test_no_drafts_step_still_reaches_global_context_path(worker_module, buffers_cls):
    spec = make_worker(worker_module, buffers_cls, 9)
    spec.propose(spec.input_batch, *[None] * 10, num_speculative_tokens=0)
    assert spec.parent_batch_sizes == [9]
    assert spec.generated == []


def test_query_failure_restores_references_without_runtime_fallback(worker_module, buffers_cls):
    spec = make_worker(worker_module, buffers_cls, 9)
    originals = spec.input_buffers, spec.block_tables, spec.draft_tokens

    def metadata(**kwargs):
        raise RuntimeError("metadata failed")

    spec._build_uniform_attn_metadata = metadata
    with pytest.raises(RuntimeError, match="metadata failed"):
        spec.propose(spec.input_batch, *[None] * 10)
    assert (spec.input_buffers, spec.block_tables, spec.draft_tokens) == originals
    assert spec._pcp_sharding_enabled
    assert spec.generated == []
    assert not spec._pcp_real_proposal


def test_disabled_feature_needs_no_startup_collective(worker_module, buffers_cls, monkeypatch):
    spec = make_worker(worker_module, buffers_cls, 9)
    spec._pcp_sharding_requested = False
    spec._pcp_sharding_enabled = False

    def unexpected():
        pytest.fail("disabled feature must not access the PCP group")

    monkeypatch.setattr(worker_module, "get_pcp_group", unexpected)
    spec.load_model(None)
    assert not spec._pcp_sharding_enabled


@pytest.mark.parametrize(
    "condition",
    ["pcp", "tp", "dp", "dcp", "pp", "graph", "probabilistic", "logits", "adaptive", "vocab", "fine", "unknown"],
)
def test_unsupported_configuration_has_a_startup_reason(worker_module, buffers_cls, condition):
    spec = make_worker(worker_module, buffers_cls, 9)
    assert spec._pcp_sharding_ineligibility() is None
    parallel = spec._pcp_target_config.parallel_config
    attrs = {
        "pcp": "prefill_context_parallel_size",
        "tp": "tensor_parallel_size",
        "dp": "data_parallel_size",
        "dcp": "decode_context_parallel_size",
        "pp": "pipeline_parallel_size",
    }
    if condition in attrs:
        setattr(parallel, attrs[condition], 1 if condition in ("pcp", "pp") else 2)
    elif condition == "graph":
        spec.speculative_config.enforce_eager = False
    elif condition == "probabilistic":
        spec.speculative_config.draft_sample_method = "probabilistic"
    elif condition == "logits":
        spec.draft_logits = object()
    elif condition == "adaptive":
        spec.enable_adaptive_verification = True
    elif condition in ("vocab", "fine"):
        spec.model.lm_head.parallel_mode = "pcp" if condition == "vocab" else "fine"
    elif condition == "unknown":
        spec.model = object()
    assert spec._pcp_sharding_ineligibility()


@pytest.mark.parametrize("peer_rejects", [False, True])
def test_startup_eligibility_is_agreed_by_all_pcp_ranks(worker_module, buffers_cls, monkeypatch, peer_rejects):
    spec = make_worker(worker_module, buffers_cls, 9)
    spec._pcp_sharding_enabled = False
    group = SimpleNamespace(world_size=8, cpu_group=object())
    monkeypatch.setattr(worker_module, "get_pcp_group", lambda: group)
    calls = []

    def consensus(results, reason, **kwargs):
        assert kwargs["group"] is group.cpu_group
        calls.append(reason)
        results[:] = [None] * 8
        if peer_rejects:
            results[-1] = "peer has a PCP-sharded head"

    monkeypatch.setattr(worker_module.dist, "all_gather_object", consensus)
    spec.load_model(None)
    assert calls == [None]
    assert spec._pcp_sharding_enabled is (not peer_rejects)
