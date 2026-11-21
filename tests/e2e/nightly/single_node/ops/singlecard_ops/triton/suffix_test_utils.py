# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared setup and references for MRV2 suffix NPU tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.platforms import current_platform

from vllm_ascend.utils import vllm_version_is

if current_platform.device_type != "npu":
    pytest.skip("requires Ascend NPU", allow_module_level=True)
if vllm_version_is("0.33.0"):
    pytest.skip("requires upstream MRV2 suffix support", allow_module_level=True)

# These tests instantiate the speculator without an NPUWorker.
from vllm.v1.worker.gpu.spec_decode.suffix.speculator import SuffixSpeculator  # noqa: E402

from vllm_ascend.patch.worker import patch_triton  # noqa: E402, F401

DEVICE = torch.device("npu")


def _local_draft(hist, k, depth):
    """Continuation of the longest, rightmost earlier occurrence of a suffix
    of ``hist`` within ``hist`` (empty if no token matches)."""
    best_len, best_pos = 0, 0
    for i in range(1, len(hist)):
        m = 0
        while m < depth and i - 1 - m >= 0 and hist[i - 1 - m] == hist[-1 - m]:
            m += 1
        if m > 0 and (m, i) >= (best_len, best_pos):
            best_len, best_pos = m, i
    return [int(t) for t in hist[best_pos : best_pos + k]] if best_len else []


def _follows_in_corpus(docs, hist, draft, n=2):
    """Draft continues some occurrence of hist's last n tokens in a doc."""
    tail = [int(t) for t in hist[-n:]]
    for doc in docs:
        for i in range(n, len(doc)):
            if doc[i - n : i] == tail and doc[i : i + len(draft)] == draft:
                return True
    return False


def _make(k, depth, max_reqs, max_len, corpus_tokens, adaptive=False):
    spec = SimpleNamespace(
        num_speculative_tokens=k,
        suffix_decoding_max_tree_depth=depth,
        suffix_decoding_corpus_tokens=corpus_tokens,
        enable_adaptive_verification=adaptive,
        suffix_decoding_max_cached_requests=10000,
    )
    cfg = SimpleNamespace(
        speculative_config=spec,
        scheduler_config=SimpleNamespace(max_num_seqs=max_reqs),
        model_config=SimpleNamespace(max_model_len=max_len),
    )

    def buf(*shape):
        return SimpleNamespace(gpu=torch.zeros(*shape, dtype=torch.int32, device=DEVICE))

    req_states = SimpleNamespace(
        all_token_ids=buf(max_reqs, max_len),
        total_len=buf(max_reqs),
        prompt_len=SimpleNamespace(np=np.zeros(max_reqs, dtype=np.int32)),
        req_id_to_index={},
    )
    return SuffixSpeculator(cfg, DEVICE, req_states), req_states


def _set(req_states, slot, prompt, output):
    toks = [int(t) for t in prompt] + [int(t) for t in output]
    req_states.all_token_ids.gpu[slot, : len(toks)] = torch.tensor(toks)
    req_states.total_len.gpu[slot] = len(toks)
    req_states.prompt_len.np[slot] = len(prompt)


def _finish(spec, req_states, slot_by_id):
    req_states.req_id_to_index = dict(slot_by_id)
    spec.on_requests_finished(list(slot_by_id))


def _propose(spec, req_states, slots, num_sampled=None, num_rejected=None):
    n = len(slots)
    max_reqs = req_states.total_len.gpu.shape[0]
    last_sampled = torch.zeros(max_reqs, 1, dtype=torch.int64, device=DEVICE)
    for s in slots:
        last = int(req_states.total_len.gpu[s]) - 1
        last_sampled[s, 0] = req_states.all_token_ids.gpu[s, last]
    batch = SimpleNamespace(
        num_reqs=n,
        idx_mapping=torch.tensor(slots, dtype=torch.int32, device=DEVICE),
        idx_mapping_np=np.array(slots),
    )
    if num_sampled is None:
        num_sampled = [1] * n
    if num_rejected is None:
        num_rejected = [0] * n
    drafts = spec.propose(
        batch,
        None,
        None,
        None,
        None,
        torch.tensor(num_sampled, dtype=torch.int32, device=DEVICE),
        torch.tensor(num_rejected, dtype=torch.int32, device=DEVICE),
        last_sampled,
        None,
        None,
        None,
    )
    return drafts.cpu().numpy(), spec.num_valid[:n].cpu().numpy()
