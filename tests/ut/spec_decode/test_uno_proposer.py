# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Index arithmetic and wiring for UNO speculative decoding.

The proposer's device work is a handful of index computations whose failure
modes are all silent: a wrong frontier drafts from the wrong position, a wrong
slot mapping corrupts KV, and wrong LoRA routing degrades acceptance while the
output stays correct. All of it is plain torch, so it is tested here on CPU.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.lora.request import LoRARequest

import vllm_ascend.spec_decode.uno_proposer as uno_proposer_module
from vllm_ascend.spec_decode.uno_proposer import (
    UNO_LORA_INT_ID,
    UNO_METHOD,
    AscendUnoProposer,
    compute_uno_frontier,
    resolve_uno_lora_path,
    uno_owns_lora_slot,
)

FORWARD_WIDTH = 4
VOCAB_SIZE = 97
BLOCK_SIZE = 8
MAX_MODEL_LEN = 4096


def test_uno_adapter_subdirectory_is_resolved_before_loading(tmp_path):
    snapshot = tmp_path / "snapshot"
    with (
        patch.object(uno_proposer_module.envs, "VLLM_USE_MODELSCOPE", False),
        patch.object(uno_proposer_module, "hf_api") as api,
    ):
        api.return_value.snapshot_download.return_value = str(snapshot)
        resolved = resolve_uno_lora_path("s-sahoo/uno-qwen3-8B/adapter", revision="pinned-revision")
    assert resolved == str(snapshot / "adapter")
    api.return_value.snapshot_download.assert_called_once_with(
        repo_id="s-sahoo/uno-qwen3-8B",
        revision="pinned-revision",
        allow_patterns=["adapter/*"],
    )


@pytest.mark.parametrize("local_path", ["adapter", "org/repo/adapter"])
def test_uno_adapter_local_directory_never_downloads(tmp_path, monkeypatch, local_path):
    monkeypatch.chdir(tmp_path)
    adapter = tmp_path / local_path
    adapter.mkdir(parents=True)
    with patch.object(uno_proposer_module, "hf_api") as api:
        assert resolve_uno_lora_path(local_path) == str(adapter)
        assert resolve_uno_lora_path(str(adapter)) == str(adapter)
    api.assert_not_called()


def test_uno_adapter_explicit_missing_local_path_never_downloads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch.object(uno_proposer_module, "hf_api") as api:
        assert resolve_uno_lora_path("./missing/adapter") == str(tmp_path / "missing" / "adapter")
        assert resolve_uno_lora_path(str(tmp_path / "absent")) == str(tmp_path / "absent")
    api.assert_not_called()


def test_uno_adapter_standalone_repo_uses_the_existing_loader():
    with patch.object(uno_proposer_module, "hf_api") as api:
        assert resolve_uno_lora_path("org/standalone-adapter") == "org/standalone-adapter"
    api.assert_not_called()


@pytest.mark.parametrize("path", ["org/repo/../adapter", "org/repo//adapter", "org/repo/./adapter"])
def test_uno_adapter_invalid_subdirectory_is_rejected(path):
    with (
        patch.object(uno_proposer_module, "hf_api") as api,
        pytest.raises(ValueError, match="Invalid UNO adapter"),
    ):
        resolve_uno_lora_path(path)
    api.assert_not_called()


def test_uno_adapter_modelscope_subdirectory_does_not_use_huggingface():
    with (
        patch.object(uno_proposer_module.envs, "VLLM_USE_MODELSCOPE", True),
        patch.object(uno_proposer_module, "hf_api") as api,
        pytest.raises(ValueError, match="local adapter directory"),
    ):
        resolve_uno_lora_path("org/repo/adapter")
    api.assert_not_called()


def test_uno_adapter_download_failure_is_not_hidden():
    with (
        patch.object(uno_proposer_module.envs, "VLLM_USE_MODELSCOPE", False),
        patch.object(uno_proposer_module, "hf_api") as api,
        pytest.raises(OSError, match="offline cache miss"),
    ):
        api.return_value.snapshot_download.side_effect = OSError("offline cache miss")
        resolve_uno_lora_path("org/repo/adapter")


def _make_proposer(forward_width: int = FORWARD_WIDTH, max_num_reqs: int = 8) -> AscendUnoProposer:
    """Build a proposer without a VllmConfig, a model, or a device."""
    proposer = object.__new__(AscendUnoProposer)
    proposer.device = torch.device("cpu")
    proposer.forward_width = forward_width
    proposer.vocab_size = VOCAB_SIZE
    proposer.max_num_reqs = max_num_reqs
    proposer.max_num_tokens = max_num_reqs * forward_width
    proposer.input_ids = torch.zeros(proposer.max_num_tokens, dtype=torch.int32)
    proposer.positions = torch.zeros(proposer.max_num_tokens, dtype=torch.int64)
    proposer.slot_mapping = torch.zeros(proposer.max_num_tokens, dtype=torch.int32)
    proposer.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32)
    proposer.block_offsets = torch.arange(forward_width, dtype=torch.int64)
    proposer._block_size = BLOCK_SIZE
    proposer.max_model_len = MAX_MODEL_LEN
    proposer._lora_loaded = True
    proposer.lora_request = LoRARequest(lora_name="uno-test", lora_int_id=UNO_LORA_INT_ID, lora_path="/uno/adapter")
    proposer._gated_mapping_cache = {}
    proposer._base_mapping_cache = {}
    proposer._draft_graph = None
    proposer._draft_graph_batch_sizes = set()
    proposer.speculative_config = SimpleNamespace(enforce_eager=None)
    return proposer


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_DECODE_ONLY])
def test_draft_graph_buckets_use_exact_request_counts(mode):
    proposer = _make_proposer(max_num_reqs=4)
    proposer.vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=mode))
    sizes = proposer.graph_capture_sizes([0, 1, 4, 5, 10, 15, 20, 25])
    assert sizes == ([4, 8, 12, 16] if mode == CUDAGraphMode.FULL_DECODE_ONLY else [])


def test_explicit_eager_draft_disables_its_graph_buckets():
    proposer = _make_proposer()
    proposer.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    )
    proposer.speculative_config.enforce_eager = True
    assert proposer.graph_capture_sizes([5, 10]) == []


def test_batch_invariant_full_decode_reports_the_supported_alternative():
    proposer = _make_proposer()
    proposer.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    )
    with (
        patch.object(uno_proposer_module.envs, "VLLM_BATCH_INVARIANT", True),
        pytest.raises(ValueError, match="PIECEWISE"),
    ):
        proposer.graph_capture_sizes([5, 10])


@pytest.mark.parametrize("enable_enpu", [False, True])
@pytest.mark.parametrize("captured", [False, True])
def test_draft_graph_replay_updates_attention_and_preserves_compiled_bypass(enable_enpu, captured):
    proposer = _make_proposer()
    events = []
    model = MagicMock(side_effect=lambda **kwargs: events.append("eager") or torch.zeros(FORWARD_WIDTH, 3))
    proposer._draft_graph = MagicMock(
        side_effect=lambda **kwargs: events.append("replay") or torch.zeros(FORWARD_WIDTH, 3)
    )
    proposer._draft_graph_batch_sizes = {1} if captured else {2}
    proposer.vllm_config = SimpleNamespace()
    proposer.runner = SimpleNamespace(get_model=lambda: model, enable_enpu=enable_enpu)
    proposer._update_graph_params = MagicMock(side_effect=lambda *args: events.append("update"))
    context = SimpleNamespace()
    with (
        patch.object(uno_proposer_module, "set_ascend_forward_context") as ctx,
        patch.object(uno_proposer_module, "get_forward_context", return_value=context),
        patch.object(torch.npu, "current_stream") as current_stream,
    ):
        proposer._forward(proposer.input_ids[:FORWARD_WIDTH], proposer.positions[:FORWARD_WIDTH], {}, 1)
    assert ctx.call_args.kwargs["skip_compiled"] is True
    assert ctx.call_args.kwargs["is_draft_model"] is True
    if captured:
        assert ctx.call_args.kwargs["aclgraph_runtime_mode"] == CUDAGraphMode.FULL
        descriptor = ctx.call_args.kwargs["batch_descriptor"]
        assert descriptor.num_tokens == FORWARD_WIDTH and descriptor.num_reqs == 1
        assert descriptor.has_lora and descriptor.num_active_loras == 1
        assert events == (["update", "replay"] if enable_enpu else ["replay", "update"])
        assert current_stream.return_value.synchronize.call_count == int(enable_enpu)
        model.assert_not_called()
    else:
        assert ctx.call_args.kwargs["aclgraph_runtime_mode"] == CUDAGraphMode.NONE
        assert events == ["eager"]
        proposer._draft_graph.assert_not_called()


def test_capture_uses_real_adapter_after_warmup_and_restores_base_mapping_on_error():
    proposer = _make_proposer(max_num_reqs=2)
    compilation = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY, cudagraph_capture_sizes=[5, 10])
    proposer.vllm_config = SimpleNamespace(compilation_config=compilation)
    compilation.cudagraph_num_of_warmups = 1
    proposer.query_start_loc = torch.arange(3, dtype=torch.int32) * FORWARD_WIDTH
    proposer.query_start_loc_cpu = proposer.query_start_loc.clone()
    proposer.runner = SimpleNamespace(
        compilation_config=compilation,
        enable_enpu=False,
        get_model=MagicMock(),
        input_batch=SimpleNamespace(block_table=[SimpleNamespace(get_device_tensor=lambda: torch.zeros(2, 8))]),
    )
    proposer.load_lora_adapter = MagicMock()
    proposer._set_gated_lora_routing = MagicMock()
    proposer._clear_lora_routing = MagicMock()
    proposer._build_draft_attn_metadata = MagicMock(return_value={})
    proposer._forward = MagicMock(side_effect=[torch.zeros(8, 3), RuntimeError("capture failed")])
    with (
        patch.object(uno_proposer_module.envs, "VLLM_BATCH_INVARIANT", False),
        patch.object(uno_proposer_module, "ACLGraphWrapper"),
        patch.object(torch.npu, "synchronize"),
        pytest.raises(RuntimeError, match="capture failed"),
    ):
        proposer.capture_model()
    proposer.load_lora_adapter.assert_called_once()
    proposer._set_gated_lora_routing.assert_called_once_with(2)
    proposer._clear_lora_routing.assert_called_once_with(8)
    assert proposer._forward.call_args.kwargs["capture"] is True
    assert torch.all(proposer.slot_mapping[:8] == -1)
    assert not proposer._draft_graph_batch_sizes


@pytest.mark.parametrize("model_raises", [False, True])
def test_propose_bypasses_compiled_verify_and_restores_routing(model_raises):
    proposer = _make_proposer()
    model = MagicMock(return_value=torch.zeros(FORWARD_WIDTH, 3))
    model.compute_logits.return_value = torch.zeros(FORWARD_WIDTH, VOCAB_SIZE)
    if model_raises:
        model.side_effect = RuntimeError("draft forward failed")
    proposer.vllm_config = SimpleNamespace()
    proposer.runner = SimpleNamespace(
        attn_groups=[[]],
        requests={},
        input_batch=SimpleNamespace(),
        discard_request_indices=SimpleNamespace(gpu=torch.empty(0, dtype=torch.int32)),
        num_discarded_requests=0,
        _copy_valid_sampled_token_count=MagicMock(),
        _sync_metadata_across_dp=lambda *args, **kwargs: (FORWARD_WIDTH, None, None),
        get_model=lambda: model,
    )
    proposer.prepare_next_token_ids_padded = MagicMock(return_value=(torch.tensor([17]), torch.tensor([1])))
    proposer._build_draft_attn_metadata = MagicMock(return_value={})
    proposer._set_gated_lora_routing = MagicMock()
    proposer._clear_lora_routing = MagicMock()
    proposer._sample_draft_tokens = MagicMock(return_value=(torch.arange(FORWARD_WIDTH), None))
    metadata = SimpleNamespace(num_reqs=1, seq_lens=torch.tensor([3]), block_table_tensor=torch.tensor([[0]]))
    with (
        patch.object(uno_proposer_module, "set_ascend_forward_context", return_value=nullcontext()) as context,
        patch.object(uno_proposer_module, "get_forward_context", return_value=SimpleNamespace()),
        pytest.raises(RuntimeError, match="draft forward failed") if model_raises else nullcontext(),
    ):
        proposer.propose(FORWARD_WIDTH, torch.tensor([[17]]), metadata, None, None)
    assert context.call_args.kwargs["skip_compiled"] is True
    assert context.call_args.kwargs["aclgraph_runtime_mode"] == uno_proposer_module.CUDAGraphMode.NONE
    proposer._set_gated_lora_routing.assert_called_once_with(1)
    proposer._clear_lora_routing.assert_called_once_with(FORWARD_WIDTH)


def test_draft_input_is_seed_then_noise():
    proposer = _make_proposer()
    torch.manual_seed(0)
    seeds = torch.tensor([11, 22, 33], dtype=torch.int64)

    rows = proposer._build_draft_input_ids(seeds, num_reqs=3).view(3, FORWARD_WIDTH)

    assert rows[:, 0].tolist() == [11, 22, 33]
    noise = rows[:, 1:]
    assert noise.min().item() >= 0
    # Noise must be a valid token id: the rejection sampler drops anything
    # >= vocab_size as well as the -1 placeholder.
    assert noise.max().item() < VOCAB_SIZE


def test_draft_positions_start_at_the_frontier():
    proposer = _make_proposer()
    frontier = torch.tensor([5, 130], dtype=torch.int32)

    positions = proposer._build_draft_positions(frontier, num_reqs=2).view(2, FORWARD_WIDTH)

    assert positions[0].tolist() == [5, 6, 7, 8]
    assert positions[1].tolist() == [130, 131, 132, 133]


def test_draft_positions_are_clamped_at_the_end_of_the_model_length():
    """A request within F tokens of max_model_len must not index past its own
    block table; the clamp target is at or after the frontier, so it can only
    land on scratch KV."""
    proposer = _make_proposer()
    frontier = torch.tensor([MAX_MODEL_LEN - 2], dtype=torch.int32)

    positions = proposer._build_draft_positions(frontier, num_reqs=1).view(1, FORWARD_WIDTH)

    assert positions[0].tolist() == [
        MAX_MODEL_LEN - 2,
        MAX_MODEL_LEN - 1,
        MAX_MODEL_LEN - 1,
        MAX_MODEL_LEN - 1,
    ]
    assert positions.max().item() < MAX_MODEL_LEN


def test_draft_slot_mapping_follows_the_block_table_across_a_boundary():
    proposer = _make_proposer()
    # Request 0 starts mid-block and crosses into the next block, which is why
    # the slot has to be resolved per row rather than as base + offset.
    frontier = torch.tensor([6, 0], dtype=torch.int32)
    block_table = torch.tensor([[3, 9, 0, 0], [4, 5, 0, 0]], dtype=torch.int32)
    positions = proposer._build_draft_positions(frontier, num_reqs=2)

    slots = proposer._build_draft_slot_mapping(positions, block_table, num_reqs=2).view(2, FORWARD_WIDTH)

    # positions 6,7 land in block 3; positions 8,9 land in block 9.
    assert slots[0].tolist() == [3 * BLOCK_SIZE + 6, 3 * BLOCK_SIZE + 7, 9 * BLOCK_SIZE + 0, 9 * BLOCK_SIZE + 1]
    assert slots[1].tolist() == [4 * BLOCK_SIZE + i for i in range(FORWARD_WIDTH)]


def test_gated_routing_puts_only_the_seed_row_on_base_weights():
    proposer = _make_proposer()
    captured = {}

    def _capture(prompt_mapping, token_mapping, lora_requests):
        captured["prompt"] = prompt_mapping
        captured["token"] = token_mapping
        captured["requests"] = lora_requests

    proposer.runner = SimpleNamespace(_set_active_loras=_capture, _ensure_lora_enabled=lambda: None)
    proposer._set_gated_lora_routing(num_reqs=2)

    assert captured["token"] == (
        0,
        UNO_LORA_INT_ID,
        UNO_LORA_INT_ID,
        UNO_LORA_INT_ID,
        0,
        UNO_LORA_INT_ID,
        UNO_LORA_INT_ID,
        UNO_LORA_INT_ID,
    )
    # The prompt (sampler) mapping must be per row too: UNO samples every draft
    # row, and a per-request vector would be shorter than the rows the sampler
    # index tensor is narrowed to.
    assert captured["prompt"] == captured["token"]
    assert len(captured["requests"]) == 1


def test_clearing_routing_returns_every_row_to_base_weights():
    proposer = _make_proposer()
    captured = {}
    proposer.runner = SimpleNamespace(_set_active_loras=lambda p, t, r: captured.update(prompt=p, token=t, requests=r))

    proposer._clear_lora_routing(num_tokens=12)

    assert set(captured["token"]) == {0}
    assert len(captured["token"]) == 12
    assert captured["requests"] == set()


@pytest.mark.parametrize("num_emitted", [1, 2, 3, 5])
def test_frontier_advances_by_the_number_of_emitted_tokens(num_emitted):
    # One request scheduled with F draft tokens: the verify window ends at
    # C + F + 1 and the new frontier must be C + num_emitted.
    committed = 100
    num_draft = FORWARD_WIDTH
    verify_window_end = torch.tensor([committed + num_draft + 1], dtype=torch.int32)
    spec_decode_metadata = SimpleNamespace(cu_num_draft_tokens=torch.tensor([num_draft], dtype=torch.int32))
    valid_count = torch.tensor([num_emitted], dtype=torch.int32)

    frontier = compute_uno_frontier(verify_window_end, spec_decode_metadata, valid_count, num_reqs=1)

    assert frontier.tolist() == [committed + num_emitted]


def test_frontier_uses_the_window_end_when_there_were_no_drafts():
    verify_window_end = torch.tensor([17, 42], dtype=torch.int32)
    frontier = compute_uno_frontier(verify_window_end, None, torch.tensor([1, 1]), num_reqs=2)
    assert frontier.tolist() == [17, 42]


def test_frontier_handles_a_mixed_batch_of_drafted_and_undrafted_requests():
    # Request 1 is still prefilling, so it was scheduled with zero drafts and
    # must not be charged a rejection.
    verify_window_end = torch.tensor([50, 33, 90], dtype=torch.int32)
    cu_num_draft_tokens = torch.tensor([4, 4, 8], dtype=torch.int32)  # per-request 4, 0, 4
    valid_count = torch.tensor([3, 1, 5], dtype=torch.int32)

    frontier = compute_uno_frontier(
        verify_window_end,
        SimpleNamespace(cu_num_draft_tokens=cu_num_draft_tokens),
        valid_count,
        num_reqs=3,
    )

    # req0: 50 - (4 + 1 - 3) = 48 ; req1: untouched ; req2: 90 - (4 + 1 - 5) = 90
    assert frontier.tolist() == [48, 33, 90]


def test_uno_acceptance_matches_sglangs_arithmetic():
    """vLLM emits ``n_accepted + 1``; SGLang reports ``n + 2``. Same number.

    SGLang verifies only ``candidates[1:]`` and prepends the clean token, so its
    ``n`` counts matching *proposals*. Here ``candidates[0]`` is registered as
    the first draft token, so vLLM's accepted count is ``n + 1``. Both describe
    a row of ``n + 2`` emitted tokens.

    The identity is asserted *through* ``compute_uno_frontier`` rather than on
    paper, so that a change to the frontier arithmetic breaks this test.
    """
    forward_width = 5
    committed = 100
    candidates = [10, 20, 30, 40, 50]
    # The target agrees on the clean token and on the first two proposals.
    target = [10, 20, 30, 41, 51, 60]

    # SGLang: candidates[0] is free, then match candidates[1:] against its own
    # target_top1. Its verify window starts one position later, so its
    # target_top1[i] is the token for C+i+2, which is target[i+1] here.
    sglang_target_top1 = target[1:]
    n = 0
    while n < forward_width - 1 and candidates[n + 1] == sglang_target_top1[n]:
        n += 1
    sglang_accept_len = n + 2

    # vLLM: walk the drafts, stop at the first mismatch, then append the target.
    accepted = 0
    while accepted < forward_width and candidates[accepted] == target[accepted]:
        accepted += 1
    vllm_emitted = accepted + 1
    assert vllm_emitted == sglang_accept_len == 4

    # And that is exactly how far the production frontier advances: the request
    # was scheduled with `forward_width` drafts, so its verify window ends at
    # C + F + 1, and `valid_sampled_tokens_count` is the emitted count.
    frontier = compute_uno_frontier(
        torch.tensor([committed + forward_width + 1], dtype=torch.int32),
        SimpleNamespace(cu_num_draft_tokens=torch.tensor([forward_width], dtype=torch.int32)),
        torch.tensor([vllm_emitted], dtype=torch.int32),
        num_reqs=1,
    )
    assert frontier.tolist() == [committed + sglang_accept_len]


def test_forward_width_must_leave_at_least_one_noise_row():
    speculative_config = SimpleNamespace(num_speculative_tokens=1, model="x", rejection_sample_method="standard")
    vllm_config = SimpleNamespace(speculative_config=speculative_config)
    with pytest.raises(ValueError, match="num_speculative_tokens >= 2"):
        AscendUnoProposer(vllm_config, torch.device("cpu"), runner=None)


def test_forward_width_is_capped_by_the_fia_query_row_limit():
    speculative_config = SimpleNamespace(num_speculative_tokens=16, model="x", rejection_sample_method="standard")
    vllm_config = SimpleNamespace(speculative_config=speculative_config)
    with pytest.raises(ValueError, match=r"at most\s+16"):
        AscendUnoProposer(vllm_config, torch.device("cpu"), runner=None)


def test_get_spec_decode_method_dispatches_uno():
    from vllm_ascend import spec_decode

    with patch.object(spec_decode, "AscendUnoProposer") as uno_cls:
        vllm_config = MagicMock()
        device = MagicMock()
        runner = MagicMock()
        result = spec_decode.get_spec_decode_method("uno", vllm_config, device, runner)

    uno_cls.assert_called_once_with(vllm_config, device, runner)
    assert result is uno_cls.return_value


def test_draft_seq_lens_are_capped_at_the_model_length():
    """FIA is handed the whole block-table row and bounded only by seq_lens.

    A row is ``cdiv(max_model_len, block_size)`` wide, so an unclamped
    ``frontier + F`` on a request near the length limit makes the kernel read
    into the next request's blocks.
    """
    proposer = _make_proposer()
    frontier = torch.tensor([10, MAX_MODEL_LEN - 2, MAX_MODEL_LEN - 1], dtype=torch.int32)
    proposer.seq_lens = torch.zeros(8, dtype=torch.int32)

    seq_lens = proposer._build_draft_seq_lens(frontier, num_reqs=3)

    assert seq_lens.tolist() == [10 + FORWARD_WIDTH, MAX_MODEL_LEN, MAX_MODEL_LEN]
    # The cap has to match the number of positions the block table can address,
    # which is what `_build_draft_positions` clamps to minus one.
    clamped_positions = proposer._build_draft_positions(frontier, num_reqs=3)
    assert clamped_positions.max().item() + 1 <= seq_lens.max().item()


def test_routing_vectors_are_cached_per_batch_size():
    proposer = _make_proposer()
    seen = []
    proposer.runner = SimpleNamespace(
        _set_active_loras=lambda p, t, r: seen.append(t),
        _ensure_lora_enabled=lambda: None,
    )

    proposer._set_gated_lora_routing(num_reqs=2)
    proposer._set_gated_lora_routing(num_reqs=2)
    proposer._set_gated_lora_routing(num_reqs=3)

    # Same object reused for a repeated batch size: rebuilding an
    # O(num_reqs * F) tuple on every decode step is pure host overhead.
    assert seen[0] is seen[1]
    assert seen[2] is not seen[0]
    assert len(seen[2]) == 3 * FORWARD_WIDTH


def test_draft_sampling_never_modifies_the_published_q():
    """`probs` is handed to the rejection sampler as q after this returns."""
    torch.manual_seed(0)
    probs = torch.randn(9, 64).softmax(dim=-1, dtype=torch.float32)
    snapshot = probs.clone()

    ids = uno_proposer_module._sample_chunked(probs)

    assert torch.equal(probs, snapshot)
    assert ids.shape == (9,)
    assert int(ids.min()) >= 0 and int(ids.max()) < 64


def test_draft_sampling_is_an_exponential_race():
    # A one-hot row can only ever sample its own token, whatever the noise is.
    onehot = torch.zeros(5, 32)
    targets = torch.arange(5) * 6
    onehot[torch.arange(5), targets] = 1.0
    for _ in range(10):
        assert torch.equal(uno_proposer_module._sample_chunked(onehot), targets)

    # And the frequencies follow the distribution.
    torch.manual_seed(0)
    p = torch.tensor([[0.6, 0.3, 0.1]]).repeat(4000, 1)
    freq = torch.bincount(uno_proposer_module._sample_chunked(p), minlength=3).float() / 4000
    assert (freq - torch.tensor([0.6, 0.3, 0.1])).abs().max().item() < 0.03


def test_draft_sampling_chunking_does_not_change_the_result(monkeypatch):
    """The chunked path exists to bound the noise buffer, not to change sampling."""
    onehot = torch.zeros(9, 128)
    targets = torch.arange(9) * 3
    onehot[torch.arange(9), targets] = 1.0
    # Two rows per chunk over nine rows: five chunks, last one short.
    monkeypatch.setattr(uno_proposer_module, "DRAFT_SAMPLE_CHUNK_BYTES", 128 * 4 * 2)

    assert torch.equal(uno_proposer_module._sample_chunked(onehot), targets)


def test_uno_owns_lora_slot_only_for_uno():
    assert uno_owns_lora_slot(SimpleNamespace(method=UNO_METHOD))
    assert not uno_owns_lora_slot(SimpleNamespace(method="eagle"))
    assert not uno_owns_lora_slot(None)
