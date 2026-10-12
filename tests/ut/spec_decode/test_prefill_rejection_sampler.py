# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for the upstream sampler's gathered-draft boundary."""

import ast
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as upstream

from vllm_ascend.patch.worker.patch_v2 import patch_rejection_sampler as sampler_patch
from vllm_ascend.patch.worker.patch_v2.prefill_rejection_sampler import (
    install_prefill_placeholder_mask,
    mask_prefill_placeholder_drafts,
)
from vllm_ascend.patch.worker.patch_v2.recompute_rejection_sampler import (
    RecomputeAwareRejectionSampler,
)


@dataclass
class Batch:
    num_draft_tokens: int
    is_prefilling_np: np.ndarray
    num_computed_prefill_tokens_np: np.ndarray
    prefill_len_np: np.ndarray
    has_prefill: bool
    idx_mapping_np: np.ndarray
    expanded_idx_mapping: torch.Tensor
    expanded_local_pos: torch.Tensor
    input_ids: torch.Tensor
    logits_indices: torch.Tensor
    positions: torch.Tensor
    cu_num_logits_np: np.ndarray
    cu_num_logits: torch.Tensor
    idx_mapping: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu_upper_bound: torch.Tensor
    num_reqs: int = 3


def _batch(k=3, variable=False, phase="prefill"):
    # Request-state indices are deliberately neither dense nor sorted. The
    # final decode request contributes a legitimate zero-valued draft.
    counts = [k + 1, 2 if variable else k + 1, k if variable else k + 1]
    mapping = np.array([9, 2, 7])
    expanded = torch.repeat_interleave(torch.tensor(mapping), torch.tensor(counts))
    local_pos = torch.cat([torch.arange(count) for count in counts])
    size = sum(counts)
    flags = np.array([phase == "prefill", phase == "prefill", False])
    computed = np.array([99, 99, 100]) if phase != "decode" else np.full(3, 100)
    cu_logits = np.concatenate(([0], np.cumsum(counts)))
    return Batch(
        num_draft_tokens=0 if phase == "no_draft" else size - 3,
        is_prefilling_np=flags,
        num_computed_prefill_tokens_np=computed,
        prefill_len_np=np.full(3, 100),
        has_prefill=bool(flags.any()),
        idx_mapping_np=mapping,
        expanded_idx_mapping=expanded,
        expanded_local_pos=local_pos,
        input_ids=torch.tensor(([0, 17, 4] * size)[:size], dtype=torch.int32),
        logits_indices=torch.arange(size - 1, -1, -1),
        positions=torch.arange(size),
        cu_num_logits_np=cu_logits,
        cu_num_logits=torch.tensor(cu_logits, dtype=torch.int32),
        idx_mapping=torch.tensor(mapping),
        seq_lens=torch.full((3,), 104, dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.full((3,), 104, dtype=torch.int32),
    )


def _reference_drafts(batch, enabled):
    """Independent per-row oracle for the original two patches' conditions."""
    flags = batch.is_prefilling_np.copy()
    has_prefill = batch.has_prefill
    if enabled and batch.num_draft_tokens:
        restored = flags | (batch.num_computed_prefill_tokens_np < batch.prefill_len_np)
        if not np.array_equal(restored, flags):
            flags = restored
            has_prefill = bool(flags.any())
    prefill_states = set(batch.idx_mapping_np[flags])
    expected = batch.input_ids[batch.logits_indices].clone()
    if has_prefill and batch.num_draft_tokens:
        for row, (state, local_pos) in enumerate(
            zip(batch.expanded_idx_mapping.tolist(), batch.expanded_local_pos.tolist())
        ):
            if state in prefill_states and local_pos > 0:
                expected[row] = -1
    return expected, flags, has_prefill


class RecordingSampler(upstream.RejectionSampler):
    """Run real upstream orchestration; record only at the kernel boundary."""

    def __init__(self, adaptive=False):
        self.enable_adaptive_verification = adaptive
        self.sampler = SimpleNamespace(
            compute_nans=False,
            logprobs_mode="raw_logprobs",
            sampling_states=SimpleNamespace(max_num_logprobs=lambda mapping: 0),
            req_states=SimpleNamespace(prefill_len=SimpleNamespace(gpu=torch.full((10,), 100))),
        )
        self.verified = []
        self.draft_logits_seen = []

    def _verify(self, logits, draft_logits, draft_sampled, pos, cu_num_logits, idx_mapping, *expanded):
        self.verified.append(draft_sampled.clone())
        self.draft_logits_seen.append(draft_logits)
        num_reqs = len(idx_mapping)
        return logits, torch.zeros((num_reqs, 1), dtype=torch.int32), torch.ones(num_reqs, dtype=torch.int32)

    def _get_logprobs_tensors(self, *args):
        return None


@pytest.mark.parametrize("k", [3, 5, 7])
@pytest.mark.parametrize("phase", ["prefill", "recompute", "decode", "no_draft"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("layout", ["uniform", "chunked", "adaptive"])
def test_upstream_pipeline_matches_two_patch_reference(monkeypatch, k, phase, enabled, layout):
    """Exercise actual __call__ and chunk dispatch on CPU, with kernel stubs."""
    batch = _batch(k, variable=layout == "adaptive", phase=phase)
    original_ids = batch.input_ids.clone()
    original_flags = batch.is_prefilling_np.copy()
    original_has_prefill = batch.has_prefill
    expected, _, _ = _reference_drafts(batch, enabled)
    sampler = RecordingSampler(adaptive=layout == "adaptive")
    install_prefill_placeholder_mask(sampler)
    actual_sampler: Callable[..., Any]
    if enabled:
        actual_sampler = RecomputeAwareRejectionSampler(sampler)
    else:
        actual_sampler = sampler
    # A small cap forces real upstream request-boundary chunking. Adaptive
    # layouts must stay in one chunk, exactly as required by upstream.
    monkeypatch.setattr(upstream, "get_max_chunk_logits", lambda vocab_size: k + 1 if layout == "chunked" else 1000)
    monkeypatch.setattr(
        upstream, "get_num_sampled_and_rejected", lambda counts, *args: (counts, torch.zeros_like(counts))
    )
    logits = torch.randn(len(batch.logits_indices), 32)
    draft_logits = torch.randn(10, k, 32)

    result = actual_sampler(logits, batch, draft_logits)

    torch.testing.assert_close(torch.cat(sampler.verified), expected)
    assert len(sampler.verified) == (3 if layout == "chunked" else 1)
    assert all(value is draft_logits for value in sampler.draft_logits_seen)
    torch.testing.assert_close(result.num_sampled, torch.ones(3, dtype=torch.int32))
    torch.testing.assert_close(batch.input_ids, original_ids)
    np.testing.assert_array_equal(batch.is_prefilling_np, original_flags)
    assert batch.has_prefill == original_has_prefill


def test_mask_preserves_decode_zero_and_prefill_position_zero():
    batch = _batch()
    drafts = torch.zeros(len(batch.logits_indices), dtype=torch.int32)
    actual = mask_prefill_placeholder_drafts(batch, drafts)
    assert actual.tolist() == [0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0]
    assert actual.dtype == drafts.dtype
    assert actual.device == drafts.device
    assert torch.count_nonzero(drafts) == 0
    torch.testing.assert_close(mask_prefill_placeholder_drafts(batch, actual), actual)


@pytest.mark.parametrize("case", ["decode", "no_draft", "no_flagged_request", "has_prefill_false"])
def test_mask_noop_preserves_gathered_tensor_identity(case):
    batch = _batch(phase=case if case in ("decode", "no_draft") else "prefill")
    if case == "no_flagged_request":
        batch = replace(batch, is_prefilling_np=np.zeros(3, dtype=bool), has_prefill=True)
    if case == "has_prefill_false":
        batch = replace(batch, has_prefill=False)
    drafts = batch.input_ids[batch.logits_indices]
    assert mask_prefill_placeholder_drafts(batch, drafts) is drafts


def test_install_is_idempotent_and_instance_local():
    sampler, other = RecordingSampler(), RecordingSampler()
    original_class_method = upstream.RejectionSampler._verify_in_chunks
    install_prefill_placeholder_mask(sampler)
    installed = sampler._verify_in_chunks
    install_prefill_placeholder_mask(sampler)
    assert sampler._verify_in_chunks is installed
    assert upstream.RejectionSampler._verify_in_chunks is original_class_method
    assert other._verify_in_chunks.__func__ is RecordingSampler._verify_in_chunks


def test_boundary_forwards_other_arguments_and_result_without_copying():
    batch = _batch()
    drafts = batch.input_ids[batch.logits_indices]
    args = (torch.randn(len(drafts), 32), batch, torch.randn(10, 3, 32), drafts, batch.positions, 4, 2)
    result = object()
    original = Mock(return_value=result)
    sampler = SimpleNamespace(_verify_in_chunks=original)
    install_prefill_placeholder_mask(sampler)
    assert sampler._verify_in_chunks(*args) is result
    original.assert_called_once()
    forwarded = original.call_args.args
    for index in (0, 1, 2, 4):
        assert forwarded[index] is args[index]
    assert forwarded[5:] == args[5:]
    expected, _, _ = _reference_drafts(batch, enabled=False)
    torch.testing.assert_close(forwarded[3], expected)


def test_boundary_propagates_verification_failure():
    sampler = SimpleNamespace(_verify_in_chunks=Mock(side_effect=RuntimeError("verification failed")))
    install_prefill_placeholder_mask(sampler)
    batch = _batch()
    with pytest.raises(RuntimeError, match="verification failed"):
        sampler._verify_in_chunks(None, batch, None, batch.input_ids[batch.logits_indices], None, 4, 0)


def test_custom_call_that_bypasses_upstream_stays_unchanged():
    class CustomSampler(RecordingSampler):
        def __call__(self, logits, input_batch, draft_logits=None):
            return input_batch

    sampler = CustomSampler()
    batch = _batch()
    install_prefill_placeholder_mask(sampler)
    assert sampler(None, batch) is batch
    assert sampler.verified == []


def _load_runner_boundary(config, monkeypatch):
    """Execute production load_model and the real gate without NPU setup."""
    root = Path(__file__).resolve().parents[3]
    runner_path = root / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(runner_path.read_text())
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "load_model")
    # Compiling the class preserves the __class__ cell used by zero-arg super.
    runner.body = [method]
    utils = ast.parse((root / "vllm_ascend/utils.py").read_text())
    gate = next(
        node
        for node in utils.body
        if isinstance(node, ast.FunctionDef) and node.name == "is_pd_decode_recompute_scheduler_enabled"
    )
    namespace: dict[str, Any] = {
        "GPUModelRunner": type("GPUModelRunner", (), {"load_model": lambda self, *args, **kwargs: None}),
        "RejectionSampler": upstream.RejectionSampler,
        "RecomputeAwareRejectionSampler": RecomputeAwareRejectionSampler,
        "install_prefill_placeholder_mask": install_prefill_placeholder_mask,
        "get_ascend_config": lambda: config,
        "get_pd_dspark_aux_layer_ids": lambda vllm_config: [1, 3],
        "logger": SimpleNamespace(info=lambda *args: None),
        "__file__": str(runner_path),
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, gate, runner], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(runner_path), "exec"), namespace)
    monkeypatch.setattr(
        sampler_patch, "is_pd_decode_recompute_scheduler_enabled", namespace["is_pd_decode_recompute_scheduler_enabled"]
    )
    runner_cls = namespace["NPUModelRunner"]
    sampler_patch.install_rejection_sampler_patch(runner_cls)
    return runner_cls()


@pytest.mark.parametrize("role", [None, "producer", "consumer", "both"])
@pytest.mark.parametrize("enabled", [False, True])
def test_load_model_preserves_mask_and_recompute_scopes(role, enabled, monkeypatch):
    config = SimpleNamespace(scheduler_config=SimpleNamespace(recompute_scheduler_enable=enabled))
    runner = _load_runner_boundary(config, monkeypatch)
    kv_config = (
        None
        if role is None
        else SimpleNamespace(is_kv_consumer=role in ("consumer", "both"), is_kv_producer=role in ("producer", "both"))
    )
    runner.vllm_config = SimpleNamespace(kv_transfer_config=kv_config)
    sampler = RecordingSampler()
    original = sampler._verify_in_chunks
    runner.rejection_sampler = sampler
    runner.load_model()
    assert runner.pd_dspark_aux_layer_ids == [1, 3]

    # Ordinary prefill is protected under every KV role and recompute setting.
    assert sampler._verify_in_chunks != original
    assert isinstance(runner.rejection_sampler, RecomputeAwareRejectionSampler) == (enabled and role == "consumer")
    installed = sampler._verify_in_chunks
    wrapped = runner.rejection_sampler
    runner.load_model()
    assert sampler._verify_in_chunks is installed
    assert runner.rejection_sampler is wrapped


@pytest.mark.parametrize("enabled", [False, True])
def test_load_model_keeps_missing_and_unrelated_custom_sampler(enabled, monkeypatch):
    config = SimpleNamespace(scheduler_config=SimpleNamespace(recompute_scheduler_enable=enabled))
    runner = _load_runner_boundary(config, monkeypatch)
    runner.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_consumer=True, is_kv_producer=False))
    runner.rejection_sampler = None
    runner.load_model()
    assert runner.rejection_sampler is None
    assert runner.pd_dspark_aux_layer_ids == [1, 3]

    custom = lambda logits, batch, draft=None: batch
    runner.rejection_sampler = custom
    runner.load_model()
    actual = runner.rejection_sampler
    if enabled:
        assert isinstance(actual, RecomputeAwareRejectionSampler)
        assert actual._sampler is custom
    else:
        assert actual is custom
    assert not hasattr(custom, "_verify_in_chunks")


def test_worker_patch_targets_only_ascend_v2_and_is_idempotent(monkeypatch):
    events: list[Any] = []

    class GPUModelRunner:
        def load_model(self, *args, **kwargs):
            events.append((args, kwargs))

    class NPUModelRunner(GPUModelRunner):
        vllm_config = None

        def load_model(self, *args, **kwargs):
            super().load_model(*args, **kwargs)
            self.rejection_sampler = RecordingSampler()
            events.append("initialized")

    class V1ModelRunner:
        load_model = GPUModelRunner.load_model

    class InheritedNPUModelRunner(NPUModelRunner):
        pass

    gpu_load = GPUModelRunner.load_model
    v1_load = V1ModelRunner.load_model
    original = NPUModelRunner.load_model
    upstream_verify = upstream.RejectionSampler._verify_in_chunks

    def enabled(config):
        assert events[-1] == "initialized"
        return False

    monkeypatch.setattr(sampler_patch, "is_pd_decode_recompute_scheduler_enabled", enabled)
    sampler_patch.install_rejection_sampler_patch(NPUModelRunner)
    installed = NPUModelRunner.load_model
    sampler_patch.install_rejection_sampler_patch(NPUModelRunner)
    assert NPUModelRunner.load_model is installed
    assert getattr(installed, "__wrapped__", None) is original
    assert GPUModelRunner.load_model is gpu_load
    assert V1ModelRunner.load_model is v1_load
    assert upstream.RejectionSampler._verify_in_chunks is upstream_verify

    runner = InheritedNPUModelRunner()
    runner.vllm_config = None
    runner.load_model(True, "sentinel", extra="forwarded")
    assert events == [((True, "sentinel"), {"extra": "forwarded"}), "initialized"]
    batch = _batch()
    sampler = runner.rejection_sampler
    sampler._verify_in_chunks(
        torch.zeros(len(batch.input_ids), 32), batch, None, batch.input_ids, batch.positions, 1000, 0
    )
    expected = mask_prefill_placeholder_drafts(batch, batch.input_ids)
    torch.testing.assert_close(torch.cat(sampler.verified), expected)


def test_loading_failure_does_not_install_sampler_adapter(monkeypatch):
    class Runner:
        rejection_sampler: RecordingSampler

        def load_model(self, *args, **kwargs):
            raise RuntimeError("model loading failed")

    gate = Mock()
    monkeypatch.setattr(sampler_patch, "is_pd_decode_recompute_scheduler_enabled", gate)
    sampler_patch.install_rejection_sampler_patch(Runner)
    runner = Runner()
    runner.rejection_sampler = RecordingSampler()
    original = runner.rejection_sampler._verify_in_chunks
    with pytest.raises(RuntimeError, match="model loading failed"):
        runner.load_model()
    assert runner.rejection_sampler._verify_in_chunks == original
    gate.assert_not_called()


def test_worker_registry_imports_patch_and_registers_ascend_runner():
    root = Path(__file__).resolve().parents[3]
    registry = ast.parse((root / "vllm_ascend/patch/worker/__init__.py").read_text())
    assert any(
        isinstance(node, ast.Import) and any(alias.name == sampler_patch.__name__ for alias in node.names)
        for node in registry.body
    )
    assert getattr(sampler_patch.NPUModelRunner.load_model, "_ascend_prefill_boundary", False)
