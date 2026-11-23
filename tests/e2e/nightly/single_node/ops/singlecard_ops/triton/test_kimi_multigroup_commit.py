# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-group recovery must match serial recovery on every physical page."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

import vllm_ascend.ops.triton.kda.recoverssm as ops


def make_contexts(module, device, heads, width, shared_slot=False):
    rng = torch.Generator().manual_seed(141)

    def rand(*shape):
        return torch.randn(shape, generator=rng).to(device)

    groups = []
    shared_kv = None
    for group, n_layers in enumerate((1, 2, 1)):
        layers = []
        pages = 25 if shared_slot and group == 0 else 13 + group
        for _ in range(n_layers):
            if shared_slot and group == 2:
                assert shared_kv is not None
                kv_cache = shared_kv
            else:
                state = rand(pages * (group + 1), heads, width, width)[:: group + 1]
                # Kimi convolution pages store Q, K and V in model dtype.
                conv = rand(pages, 6, 3 * heads * width).bfloat16()
                kv_cache = (conv.transpose(-1, -2) if is_conv_state_dim_first() else conv, state)
                if shared_slot and group == 0:
                    shared_kv = kv_cache
            layers.append(
                SimpleNamespace(
                    kv_cache=kv_cache,
                    recoverssm_records=SimpleNamespace(
                        correction=rand(7, heads, 4, width), key_gate=rand(7, heads, 4, width * 2).bfloat16()
                    ),
                    A_log=rand(heads),
                    dt_bias=rand(heads, width),
                    gate_lower_bound=-5.0,
                    local_num_heads=heads,
                    head_dim=width,
                )
            )
        groups.append(module.KDARecoverSSMCommitContext.create(layers, spec_query_len=4, max_num_reqs=7))
    return tuple(groups)


def make_metadata(device, step, shared_slot=False):
    metadata = []
    # The second batch interleaves four speculative rows with two other rows.
    positions = torch.tensor([0, 2, 4, 5]) if step else torch.arange(4)
    batch_size = 6 if step else 4
    for group in range(3):
        # Requests have different physical source pages and sampler row orders
        # in every group. Row 2 ends exactly at a boundary with no next page.
        order = torch.tensor([2, 0, 3, 1]).roll(group + step)
        identity = step == 0 and group == 0
        if identity:
            order = torch.arange(4)
        table = (torch.arange(12).reshape(4, 3) + 3 * group + step) % 12 + 1
        if shared_slot and group == 2:
            # The groups sharing one slot own disjoint pages in the block pool.
            table += 12
        table[2, 1] = 0
        indices = torch.zeros(4, 2, dtype=torch.int32)
        indices[:, 0] = table[order, 0]
        indices[order == 3, 0] = 0
        full_table = torch.zeros(batch_size, 3, dtype=torch.int32)
        full_table[positions] = table.int()
        computed = torch.zeros(batch_size, dtype=torch.int32)
        computed[positions] = torch.tensor([2, 3, 1, 1], dtype=torch.int32)
        align = SimpleNamespace(
            block_table=full_table.to(device), num_computed_tokens=computed.to(device), block_size=4
        )
        commit = SimpleNamespace(
            state_indices=indices.to(device),
            query_start_loc=torch.arange(5, device=device, dtype=torch.int32) * 4,
            request_indices=None if identity else positions[order].to(device, torch.int32),
            align=align,
        )
        metadata.append(
            SimpleNamespace(num_spec_decodes=4, num_prefills=step, num_decodes=step, recoverssm_commit=commit)
        )
    return metadata


def make_sampled(device, step):
    values = [2, 0, 3, 1, 3, 0] if step else [1, 4, 3, 0]
    return torch.tensor(values, dtype=torch.int32, device=device)


class CountKernel:
    def __init__(self, kernel, launches):
        self.kernel = kernel
        self.launches = launches

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def invoke(*args, **kwargs):
            self.launches.append(kwargs["ALIGN_MODE"])
            return launch(*args, **kwargs)

        return invoke


def check_multi_group(device, heads, width, reference=ops, shared_slot=False):
    serial = make_contexts(reference, device, heads, width, shared_slot)
    groups = make_contexts(ops, device, heads, width, shared_slot)
    if shared_slot:
        assert groups[0].checkpoints[0].data_ptr() == groups[2].checkpoints[0].data_ptr()
    merged = ops.MultiGroupKDACommit.create(groups)
    assert merged is not None
    grouped_heads = ops.is_950() and heads in (32, 64) and width == 128
    kernel = "_commit_kda_state_grouped_kernel" if grouped_heads else "_commit_kda_state_kernel"
    for step in range(2):
        metadata = make_metadata(device, step, shared_slot)
        sampled = make_sampled(device, step)
        serial_calls, merged_calls = [], []
        with patch.object(reference, kernel, CountKernel(getattr(reference, kernel), serial_calls)):
            for m, ctx in zip(metadata, serial, strict=True):
                c = m.recoverssm_commit
                ctx.commit(
                    sampled,
                    c.state_indices[:, 0],
                    c.query_start_loc,
                    c.request_indices,
                    c.align.block_table,
                    c.align.num_computed_tokens,
                    c.align.block_size,
                )
        with patch.object(ops, kernel, CountKernel(getattr(ops, kernel), merged_calls)):
            merged.commit(metadata, sampled)

        assert serial_calls == [False, True] * 3
        assert merged_calls == [False, True]
        for ref, actual, plan in zip(serial, groups, merged.plans, strict=True):
            for name in (
                "commit_lens",
                "state_commit_lens",
                "boundary_commit_lens",
                "final_state_indices",
                "boundary_state_indices",
                "boundary_recovery_lens",
            ):
                torch.testing.assert_close(getattr(ref, name)[:4], getattr(plan, name)[:4], rtol=0, atol=0)
            for a, b in zip(ref.checkpoints + ref.conv_states, actual.checkpoints + actual.conv_states, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize(("heads", "shared_slot"), [(12, False), (12, True), (32, False)])
def test_multigroup_commit_matches_serial_on_all_pages(heads, shared_slot):
    check_multi_group(torch.device("npu"), heads, 128, shared_slot=shared_slot)


def test_draft_scope_orders_next_state_read(monkeypatch):
    from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadata
    from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState

    from vllm_ascend.models.kimi_k3.dspark_speculator import KimiDSparkSpeculator
    from vllm_ascend.models.kimi_k3.recoverssm import KimiDSparkRecoverSSMState, KimiRecoverSSMModelState
    from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator

    device = torch.device("npu")
    workers = (RecoverSSMState(), KimiDSparkRecoverSSMState(device))
    target = object.__new__(KimiRecoverSSMModelState)
    target.recoverssm = workers[1]
    proposer = object.__new__(KimiDSparkSpeculator)
    # Enqueue independent draft work inside the real Kimi proposal wrapper.
    monkeypatch.setattr(AscendDSparkSpeculator, "propose", lambda self, sampled, step: sampled + step)
    groups = [make_contexts(ops, device, 12, 128) for _ in workers]
    states = [torch.full((7,), -1, dtype=torch.int32, device=device) for _ in workers]
    counts = [torch.full_like(state, 11) for state in states]
    names = ["group_a", "group_b", "group_c"]
    attention_groups = [[SimpleNamespace(layer_names=[name])] for name in names]

    for step in range(2):
        idx = torch.tensor([1, 0, 5, 6, 2, 4] if step else [1, 5, 2, 4], dtype=torch.int32, device=device)
        sampled = make_sampled(device, step)
        next_reads = []
        for i, (worker, contexts) in enumerate(zip(workers, groups, strict=True)):
            metadata = {}
            for name, raw, ctx in zip(names, make_metadata(device, step), contexts, strict=True):
                # Only the fields consumed by the sampler's postprocess hook.
                item = object.__new__(KimiK3KDAMetadata)
                item.num_spec_decodes, item.num_prefills, item.num_decodes = (
                    raw.num_spec_decodes,
                    raw.num_prefills,
                    raw.num_decodes,
                )
                item.recoverssm_context, item.recoverssm_commit = ctx, raw.recoverssm_commit
                metadata[name] = item
            worker.record_step(metadata, attention_groups, for_capture=False)
            with proposer.sampling_scope(target) if i else nullcontext():
                with worker.postprocess_stream(idx, sampled, None) if i else nullcontext():
                    worker.commit_step(sampled, idx, state_indices=states[i], num_accepted_tokens=counts[i])
                if i:
                    draft_result = proposer.propose(sampled, step)
                # Read before the outer scope's exception-safe fence, as publication would.
                next_reads.append([page.clone() for ctx in contexts for page in ctx.checkpoints + ctx.conv_states])
        for a, b in zip(*next_reads, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
        torch.testing.assert_close(counts[0], counts[1], rtol=0, atol=0)
        torch.testing.assert_close(draft_result, sampled + step, rtol=0, atol=0)
