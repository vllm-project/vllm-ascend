# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drop-in execution inside the existing vLLM-Ascend RecoverSSM commit context."""

from .verify import layout


def verify_step(A_log, a, b, dt_bias, q, k, v, **kwargs):
    from .cube_mixed import cube_verify

    return cube_verify(A_log, a, b, dt_bias, q, k, v, **kwargs)


def prepare_step(ctx, accepted, indices, offsets, mapping, table, computed, block_size, commit_conv):
    cache = ctx.layer_caches[0]
    batch = indices.numel()
    capacity = ctx.commit_plan.capacity
    pages, hv, v, k = cache.checkpoint.shape
    if (v, k) != (128, 128):
        raise ValueError("A5 TileLang GDN expects K=V=128")
    # Context.allocate owns a contiguous six-row plan slab and graph-stable pointer tables.
    plan = ctx.commit_plan.commit_lens.as_strided((6 * capacity,), (1,))
    align = table is not None
    args = (
        accepted,
        indices,
        offsets,
        mapping if mapping is not None else indices,
        table if align else indices,
        computed if align else indices,
    )
    desc = tuple(layout(t) for t in args)
    plan_cfg = (
        desc,
        batch,
        capacity,
        table.shape[1] if align else 1,
        ctx.spec_query_len,
        block_size if align else 1,
        mapping is not None,
        align,
    )
    plan_args = tuple(t.as_strided((d[0],), (1,)) for t, d in zip(args, desc)) + (plan,)
    commit_cfg = (
        batch,
        len(ctx.layer_caches),
        pages,
        hv,
        ctx.spec_query_len,
        cache.conv.shape[1],
        ctx.conv_history_len,
        capacity,
        layout(indices),
        str(cache.checkpoint.dtype).removeprefix("torch."),
        str(cache.conv.dtype).removeprefix("torch."),
        tuple(cache.checkpoint.stride()[1:]),
        tuple(cache.replay.stride()[1:]),
        align,
        commit_conv,
    )
    commit_args = (
        ctx._state_commit_cache_address_stride_table.view(-1),
        ctx._conv_cache_address_stride_table.view(-1),
        plan_args[1],
        plan,
    )
    return plan_cfg, plan_args, commit_cfg, commit_args


def commit_step(
    ctx,
    accepted,
    indices,
    offsets,
    mapping=None,
    table=None,
    computed=None,
    block_size=None,
    commit_conv=True,
):
    if not indices.numel():
        return
    from .commit_kernel import compiled_commit, compiled_plan

    cfg, inputs, cc, ci = prepare_step(
        ctx, accepted, indices, offsets, mapping, table, computed, block_size, commit_conv
    )
    compiled_plan(*cfg)(*inputs)
    compiled_commit(*cc)(*ci)
