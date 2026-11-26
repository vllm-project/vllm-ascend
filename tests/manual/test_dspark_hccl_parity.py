# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun on 2/4/8 Ascend cards; no model weights required.

This exercises the production sharded sampling method, HCCL collectives and
categorical kernels. It does not replace real-model graph/rejection E2E tests.
"""

import os
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu  # noqa: F401
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode.dspark import hccl_sampling
from vllm_ascend.worker.v2.spec_decode.dspark.hccl_sampling import DSparkHCCLSamplingMixin

_SENTINEL = -12345.0
_VOCAB_SIZE = 1031
_STEPS = 3


class _Model:
    def __init__(self, rank, tp_size, padded_vocab):
        self.rank = rank
        self.width = padded_vocab // tp_size
        self.logits_processor = SimpleNamespace(org_vocab_size=_VOCAB_SIZE)

    def compute_draft_logits(self, hidden):
        return hidden[:, :_VOCAB_SIZE]

    def compute_draft_logits_local(self, hidden):
        start = self.rank * self.width
        return hidden[:, start : start + self.width].contiguous()

    def markov_embed(self, previous):
        return previous

    def markov_bias(self, embedded):
        preferred = (embedded * 17 + 3) % _VOCAB_SIZE
        return F.one_hot(preferred, _VOCAB_SIZE).float() * 0.75

    def map_draft_to_target(self, tokens):
        return tokens


class _Spec(DSparkHCCLSamplingMixin, DSparkSpeculator):
    pass


def _make_spec(n, rank, tp_size, device, slots, probabilistic, sharded_cache, threshold):
    spec = object.__new__(_Spec)
    spec.device = device
    spec.max_num_reqs = n + tp_size
    spec.num_speculative_steps = _STEPS
    spec._step_cols = torch.arange(_STEPS, dtype=torch.int32, device=device)
    spec._hccl_draft_enabled = True
    spec._hccl_legacy_033 = vllm_version_is("0.33.0")
    spec._hccl_min_reqs = threshold
    spec._draft_topk = None
    spec._d2t_scatter_index = None
    spec._fuse_markov_bias = False
    spec.use_confidence_head = False
    spec.use_fp64_gumbel = False
    spec.draft_tokens = torch.full((spec.max_num_reqs, _STEPS), -1, dtype=torch.int64, device=device)
    spec.sample_indices = torch.arange(n * _STEPS, device=device).flip(0)
    spec.sample_idx_mapping = slots[:, None].expand(-1, _STEPS).reshape(-1).contiguous()
    spec.sample_pos = (
        torch.arange(n, dtype=torch.int64, device=device)[:, None] * 13 + spec._step_cols + 19
    ).reshape(-1)
    spec._anchor_idx = torch.arange(n, device=device) * _STEPS
    spec.input_buffers = SimpleNamespace(input_ids=torch.arange(n * _STEPS, device=device) % _VOCAB_SIZE)
    state_count = 2 * n + 8
    spec.temperature = 0.5 + torch.arange(state_count, device=device).float() % 5 * 0.25
    spec.seeds = torch.arange(state_count, dtype=torch.int64, device=device) * 104729 + 17
    padded_vocab = ((_VOCAB_SIZE + tp_size - 1) // tp_size) * tp_size
    spec.model = _Model(rank, tp_size, padded_vocab)
    spec.draft_logits = None
    if probabilistic:
        row_mapping = torch.full((state_count,), -1, dtype=torch.int32, device=device)
        owners = torch.arange(rank, n, tp_size, device=device) if sharded_cache else torch.arange(n, device=device)
        row_mapping[slots[owners]] = torch.arange(owners.numel(), dtype=torch.int32, device=device)
        # One guard row catches accidental writes from padding/unmapped slots.
        values = torch.full((owners.numel() + 1, _STEPS, _VOCAB_SIZE), _SENTINEL, device=device)
        spec.draft_logits = SimpleNamespace(values=values, row_mapping=row_mapping)
    return spec


def _run_case(n, rank, tp_size, device, probabilistic, threshold):
    # Request slots deliberately differ from batch positions and local indices.
    slots = 2 * torch.arange(n, dtype=torch.int32, device=device).flip(0) + 3
    reference = _make_spec(n, rank, tp_size, device, slots, probabilistic, False, threshold)
    actual = _make_spec(n, rank, tp_size, device, slots, probabilistic, True, threshold)
    generator = torch.Generator().manual_seed(20261124 + n)
    padded_vocab = ((_VOCAB_SIZE + tp_size - 1) // tp_size) * tp_size
    hidden = torch.randn(n * _STEPS, padded_vocab, generator=generator).to(device)
    hidden[:, _VOCAB_SIZE:] = float("-inf")

    # Direct Core call is the replicated reference; the mixin runs the HCCL
    # path or its real super() fallback, depending on the threshold.
    DSparkSpeculator._sample_sequential(reference, n, hidden)
    actual._sample_sequential(n, hidden)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.draft_tokens[:n], reference.draft_tokens[:n], rtol=0, atol=0)

    if probabilistic:
        owners = torch.arange(rank, n, tp_size, device=device)
        owned_slots = slots[owners]
        actual_rows = actual.draft_logits.row_mapping[owned_slots].long()
        reference_rows = reference.draft_logits.row_mapping[owned_slots].long()
        torch.testing.assert_close(
            actual.draft_logits.values[actual_rows], reference.draft_logits.values[reference_rows], rtol=0, atol=0
        )
        assert torch.all(actual.draft_logits.values[-1] == _SENTINEL).item()
    assert torch.all(actual.draft_tokens[n:] == -1).item()


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    rank, tp_size = dist.get_rank(), dist.get_world_size()
    if tp_size <= 1:
        raise ValueError("Use torchrun --nproc-per-node=2 (or 4/8)")
    init_device_properties_triton()
    group = SimpleNamespace(world_size=tp_size, rank_in_group=rank, device_group=dist.group.WORLD)
    try:
        with patch.object(hccl_sampling, "get_tp_group", return_value=group):
            methods = [False] if vllm_version_is("0.33.0") else [False, True]
            for probabilistic in methods:
                for n in (tp_size - 1, tp_size, tp_size + 1):
                    _run_case(n, rank, tp_size, device, probabilistic, threshold=1)
                _run_case(tp_size - 1, rank, tp_size, device, probabilistic, threshold=tp_size)
            dist.barrier()
            if rank == 0:
                print("PASS: greedy/probabilistic paths available in this version; uneven batches and cache guards")
                if vllm_version_is("0.33.0"):
                    print("0.33.0: greedy only; probabilistic coverage requires the newer Core API")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
