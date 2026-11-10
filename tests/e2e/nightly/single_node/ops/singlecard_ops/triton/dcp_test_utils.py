# SPDX-License-Identifier: Apache-2.0
"""Independent CPU references for DCP operator tests."""

import torch


def local_to_global(indices, world_size, rank, interleave):
    mapped = (indices // interleave * world_size + rank) * interleave + indices % interleave
    return torch.where(indices >= 0, mapped, -1).to(torch.int32)


def local_visible_lengths(visible, world_size, rank, interleave):
    cycles = visible // (world_size * interleave)
    tail = (visible % (world_size * interleave) - rank * interleave).clamp(0, interleave)
    return cycles * interleave + tail


def query_visible_lengths(seq_lens, cum_query_lens, rows, requests):
    visible = (seq_lens[requests] - cum_query_lens[requests] + rows + 1).clamp_min(0)
    return torch.where(rows < cum_query_lens[-1], visible, 0)


def remap_reference(indices, world_size, rank, interleave):
    result = torch.full_like(indices, -1)
    for source, target in zip(indices.reshape(-1, indices.shape[-1]), result.reshape(-1, indices.shape[-1])):
        owned = source[(source >= 0) & ((source // interleave) % world_size == rank)]
        target[: owned.numel()] = owned // (world_size * interleave) * interleave + owned % interleave
    return result


def merge_topk_candidates(indices, scores, topk, *, local_rank=None, world_size=1, interleave=1):
    indices, scores = indices.cpu(), scores.cpu().float()
    valid = (indices >= 0) & torch.isfinite(scores)
    values = torch.where(valid, scores, -torch.inf)
    keys = torch.where(valid, indices, torch.iinfo(torch.int32).max)
    by_id = keys.argsort(dim=-1, stable=True)
    by_score = values.gather(-1, by_id).argsort(dim=-1, descending=True, stable=True)[..., :topk]
    order = by_id.gather(-1, by_score)
    selected = torch.where(values.gather(-1, order).isfinite(), indices.gather(-1, order), -1).to(torch.int32)
    return selected if local_rank is None else remap_reference(selected, world_size, local_rank, interleave)


def assert_dcp_close(actual, golden):
    rtol, atol, fixed_limit = {
        torch.float16: (2**-9, 2**-9, 0.1),
        torch.bfloat16: (2**-6, 2**-6, 1.0),
        torch.float32: (2**-10, 2**-16, 0.01),
    }[actual.dtype]
    limit = max(fixed_limit, 32 * torch.finfo(actual.dtype).eps)
    error = (actual.cpu().double() - golden.cpu().double()).abs()
    assert torch.isfinite(error).all()
    if error.numel():
        passing = error <= atol + rtol * golden.cpu().double().abs()
        assert passing.double().mean() >= 0.99
        assert error.max() <= limit
