# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify/commit correctness, including grouped heads and paged cache strides."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

import vllm_ascend.ops.triton.gdn.recoverssm_gdn as recovery  # noqa: E402


def paged(shape, *, dtype=torch.float32):
    elements = 1
    strides = []
    for size in reversed(shape[1:]):
        strides.append(elements)
        elements *= size
    backing = torch.zeros(shape[0] * (elements + 128), dtype=dtype, device="npu")
    return backing.as_strided(shape, (elements + 128, *reversed(strides)))


def padded_inner_page(shape, *, inner_padding=3, block_padding=128):
    strides = [0] * len(shape)
    inner_span = 1
    for dim in range(len(shape) - 1, 0, -1):
        strides[dim] = inner_span + inner_padding
        inner_span += (shape[dim] - 1) * strides[dim]
    strides[0] = inner_span + block_padding
    storage_size = (shape[0] - 1) * strides[0] + inner_span
    backing = torch.zeros(storage_size, dtype=torch.float32, device="npu")
    return backing.as_strided(shape, strides)


def reference(q, k, v, g, beta, initial, starts, indices, scale=None):
    out = torch.zeros_like(v, dtype=torch.float32)
    states = []
    ratio = v.shape[2] // k.shape[2]
    if scale is None:
        scale = k.shape[-1] ** -0.5
    for row, block in enumerate(indices):
        state = initial[block].float().clone()
        history = [state.clone()]
        for token in range(starts[row], starts[row + 1]):
            key = k[0, token].float().repeat_interleave(ratio, 0)
            query = q[0, token].float().repeat_interleave(ratio, 0)
            state = state * g[0, token].float().exp()[:, None, None]
            residual = (v[0, token].float() - torch.einsum("hvk,hk->hv", state, key)) * beta[0, token].float()[:, None]
            state = state + residual[:, :, None] * key[:, None, :]
            out[0, token] = torch.einsum("hvk,hk->hv", state, query) * scale
            history.append(state.clone())
        states.append(history)
    return out, states


def verify_from_activated(q, k, v, g, beta, state, replay, starts, indices, window):
    """Adapt existing activated-gate cases to the raw upstream interface."""
    hv = v.shape[2]
    A_log = torch.zeros(hv, dtype=torch.float32, device=v.device)
    dt_bias = torch.zeros_like(A_log)
    a = torch.log(torch.expm1(-g.float()))
    b = torch.logit(beta.float())
    return recovery.gdn_recoverssm_verify(
        A_log,
        a,
        b,
        dt_bias,
        q,
        k,
        v,
        checkpoint_state=state,
        replay_cache=replay,
        query_start_loc=starts,
        state_indices=indices,
        spec_query_len=window,
        use_qk_l2norm_in_kernel=False,
    )


def test_verify_and_commit_support_padded_inner_page_strides(monkeypatch):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: True)
    torch.manual_seed(41)
    blocks, hk, hv, dk, dv, window, history = 4, 1, 2, 33, 47, 3, 3
    channels = 2 * hk * dk + hv * dv
    layers = []
    for layer_idx in range(2):
        conv = paged((blocks, channels, history + window - 1), dtype=torch.bfloat16)
        conv.copy_(torch.randn_like(conv))
        state = padded_inner_page((blocks, hv, dv, dk), block_padding=128 + layer_idx * 64)
        state.copy_(torch.randn_like(state) * 0.1)
        records = (padded_inner_page((blocks, hv, window, dv + dk + 1), block_padding=160 + layer_idx * 64),)
        layers.append(SimpleNamespace(num_k_heads=hk, tp_size=1, kv_cache=(conv, state, *records)))

    context = recovery.GDNRecoverSSMCommitContext.create(layers, spec_query_len=window, max_num_reqs=1)
    indices = torch.tensor([1], dtype=torch.int32, device="npu")
    starts = torch.tensor([0, window], dtype=torch.int32, device="npu")
    expected_states = []
    for cache in context.layer_caches:
        state = cache.checkpoint
        replay = cache.replay
        initial = state.clone()
        q = torch.nn.functional.normalize(torch.randn(1, window, hk, dk, device="npu"), dim=-1).bfloat16()
        k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1).bfloat16()
        v = torch.randn(1, window, hv, dv, dtype=torch.bfloat16, device="npu")
        g = -torch.rand(1, window, hv, device="npu")
        beta = torch.rand(1, window, hv, dtype=torch.bfloat16, device="npu")
        expected_out, histories = reference(q, k, v, g, beta, initial, [0, window], [1])
        observed = verify_from_activated(q, k, v, g, beta, state, replay, starts, indices, window)
        torch.testing.assert_close(observed.float(), expected_out, atol=3e-3, rtol=2e-2)
        torch.testing.assert_close(state, initial, atol=0, rtol=0)
        final = initial.clone()
        final[1] = histories[0][2]
        expected_states.append(final)

    context.commit(
        torch.tensor([2], dtype=torch.int32, device="npu"),
        indices,
        starts,
    )
    for cache, expected_state in zip(context.layer_caches, expected_states):
        torch.testing.assert_close(cache.checkpoint, expected_state, atol=3e-5, rtol=3e-4)


def test_commit_without_conv_still_updates_checkpoint(monkeypatch):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: True)
    torch.manual_seed(59)
    blocks, hk, hv, dim, window, history = 3, 1, 2, 8, 3, 3
    channels = 2 * hk * dim + hv * dim
    conv = paged((blocks, channels, history + window - 1), dtype=torch.bfloat16)
    conv.copy_(torch.randn_like(conv))
    state = paged((blocks, hv, dim, dim))
    state.copy_(torch.randn_like(state) * 0.1)
    replay = paged((blocks, hv, window, 2 * dim + 1))
    layer = SimpleNamespace(num_k_heads=hk, tp_size=1, kv_cache=(conv, state, replay))
    context = recovery.GDNRecoverSSMCommitContext.create([layer], spec_query_len=window, max_num_reqs=1)

    initial_state = state.clone()
    initial_conv = conv.clone()
    q = torch.nn.functional.normalize(torch.randn(1, window, hk, dim, device="npu"), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1).bfloat16()
    v = torch.randn(1, window, hv, dim, dtype=torch.bfloat16, device="npu")
    g = -torch.rand(1, window, hv, device="npu")
    beta = torch.rand(1, window, hv, dtype=torch.bfloat16, device="npu")
    starts = torch.tensor([0, window], dtype=torch.int32, device="npu")
    indices = torch.tensor([1], dtype=torch.int32, device="npu")
    _, histories = reference(q, k, v, g, beta, initial_state, [0, window], [1])
    verify_from_activated(q, k, v, g, beta, state, replay, starts, indices, window)

    request_indices = torch.tensor([0, 7], dtype=torch.int32, device="npu")
    context.commit(
        torch.tensor([2], dtype=torch.int32, device="npu"),
        indices,
        starts,
        request_indices,
        None,
        None,
        None,
        False,
    )

    expected_state = initial_state.clone()
    expected_state[1] = histories[0][2]
    torch.testing.assert_close(state, expected_state, atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(conv, initial_conv, atol=0, rtol=0)


def test_commit_context_rejects_mismatched_inner_page_strides(monkeypatch):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: True)
    blocks, hk, hv, dim, window, history = 3, 1, 2, 8, 3, 3
    channels = 2 * hk * dim + hv * dim
    layers = []
    for inner_padding in (2, 5):
        layer = SimpleNamespace(
            num_k_heads=hk,
            tp_size=1,
            kv_cache=(
                paged((blocks, channels, history + window - 1), dtype=torch.bfloat16),
                padded_inner_page((blocks, hv, dim, dim), inner_padding=inner_padding),
                padded_inner_page((blocks, hv, window, 2 * dim + 1), inner_padding=inner_padding),
            ),
        )
        layers.append(layer)

    with pytest.raises(ValueError, match="checkpoint page inner strides"):
        recovery.GDNRecoverSSMCommitContext.create(layers, spec_query_len=window, max_num_reqs=1)


@pytest.mark.parametrize("accepted", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("hk,hv,dk,dv", [(2, 4, 128, 128), (1, 2, 33, 47)])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_verify_and_multilayer_commit_preserve_accepted_prefix(
    monkeypatch, accepted, dim_first, hk, hv, dk, dv, state_dtype
):
    torch.manual_seed(73)
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: dim_first)
    blocks, window, history = 5, 5, 3
    channels = 2 * hk * dk + hv * dv
    layers = []
    for _ in range(2):
        conv = paged((blocks, channels, history + window - 1), dtype=torch.bfloat16)
        conv.copy_(torch.randn_like(conv))
        state = paged((blocks, hv, dv, dk), dtype=state_dtype)
        state.copy_(torch.randn_like(state) * 0.1)
        records = (paged((blocks, hv, window, dv + dk + 1)),)
        layers.append(
            SimpleNamespace(
                num_k_heads=hk, tp_size=1, kv_cache=(conv if dim_first else conv.transpose(-1, -2), state, *records)
            )
        )
    context = recovery.GDNRecoverSSMCommitContext.create(layers, spec_query_len=window, max_num_reqs=4)
    # Reorder physical pages between rounds; sampler rows are in a different order.
    for block_ids in ([3, 1, 0], [1, 3, 0]):
        indices = torch.tensor(block_ids, dtype=torch.int32, device="npu")
        starts = torch.tensor([0, 5, 8, 8], dtype=torch.int32, device="npu")
        counts = [accepted, min(accepted, 3)]
        expected_states, expected_convs = [], []
        for layer, cache in zip(layers, context.layer_caches):
            conv = cache.conv
            state = cache.checkpoint
            replay = cache.replay
            initial, old_conv = state.clone(), conv.clone()
            # Poison previous-round records: rejected/padded entries must not leak.
            replay.fill_(float("nan"))
            q = torch.nn.functional.normalize(torch.randn(1, 8, hk, dk, device="npu"), dim=-1).to(torch.bfloat16)
            k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1).to(torch.bfloat16)
            v = torch.randn(1, 8, hv, dv, device="npu", dtype=torch.bfloat16)
            g = -torch.rand(1, 8, hv, device="npu")
            beta = torch.rand(1, 8, hv, device="npu", dtype=torch.bfloat16)
            expected, histories = reference(q, k, v, g, beta, initial, [0, 5, 8, 8], block_ids)
            observed = verify_from_activated(q, k, v, g, beta, state, replay, starts, indices, window)
            torch.testing.assert_close(observed.float(), expected, atol=3e-3, rtol=2e-2)
            # RecoverSSM records the effective decay so commit replays the exact
            # value instead of evaluating exp(g) a second time.
            for row, block in enumerate(block_ids):
                begin = starts[row].item()
                length = starts[row + 1].item() - begin
                expected_key = k[0, begin : begin + length].float().repeat_interleave(hv // hk, dim=1).transpose(0, 1)
                torch.testing.assert_close(replay[block, :, :length, dv : dv + dk], expected_key, atol=0, rtol=0)
                expected_decay = g[0, begin : begin + length].float().exp().transpose(0, 1)
                torch.testing.assert_close(
                    replay[block, :, :length, dv + dk],
                    expected_decay,
                    atol=2e-3,
                    rtol=2e-3,
                )
            torch.testing.assert_close(state, initial, atol=0, rtol=0)
            final = initial.clone()
            conv_final = old_conv.clone()
            for row, count in enumerate(counts):
                block = block_ids[row]
                final[block] = histories[row][count]
                if count:
                    conv_final[block, :, :history] = old_conv[block, :, count - 1 : count - 1 + history]
            expected_states.append(final)
            expected_convs.append(conv_final)
        sampled = torch.tensor([counts[1], 99, counts[0]], dtype=torch.int32, device="npu")
        mapping = torch.tensor([2, 0], dtype=torch.int32, device="npu")
        context.commit(sampled, indices[:2], starts[:3], request_indices=mapping)
        for cache, expected_state, expected_conv in zip(context.layer_caches, expected_states, expected_convs):
            torch.testing.assert_close(
                cache.checkpoint,
                expected_state,
                atol=2e-3 if state_dtype == torch.bfloat16 else 3e-5,
                rtol=1e-2 if state_dtype == torch.bfloat16 else 3e-4,
            )
            torch.testing.assert_close(cache.conv, expected_conv, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["cross", "exact", "no_cross", "inplace", "zero", "full"])
@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("hk,hv,dk,dv", [(1, 2, 33, 47), (2, 4, 128, 128), (2, 6, 128, 128)])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_align_commit_restores_boundary_and_final_pages(
    monkeypatch, mode, dim_first, graph_mode, hk, hv, dk, dv, state_dtype
):
    torch.manual_seed(91)
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: dim_first)
    blocks, window, history, block_size = 8, 5, 3, 8
    channels = 2 * hk * dk + hv * dv
    computed, accepted = {
        "cross": (6, 4),
        "exact": (6, 2),
        "no_cross": (3, 2),
        "inplace": (3, 2),
        "zero": (6, 0),
        "full": (6, 5),
    }[mode]
    # Spec row 0 maps to sampler row 2; include a second request and a null row.
    mapping = torch.tensor([2, 0, 1], dtype=torch.int32, device="npu")
    counts = torch.tensor([1, 4, accepted], dtype=torch.int32, device="npu")
    computed_tokens = torch.tensor([3, 0, computed], dtype=torch.int32, device="npu")
    table_row = [1, 2, 3] if mode != "no_cross" else [2, 3, 1]
    table = torch.tensor([[4, 5, 6], [0, 0, 0], table_row], dtype=torch.int32, device="npu")
    indices = torch.tensor([1, 4, 0], dtype=torch.int32, device="npu")
    starts = torch.tensor([0, 5, 10, 10], dtype=torch.int32, device="npu")
    layers, originals, expected = [], [], []
    for layer_idx in range(2):
        # Exercise different physical channel strides in one batched commit.
        if layer_idx == 0:
            conv = paged((blocks, channels, history + window - 1), dtype=torch.bfloat16)
        else:
            conv = paged((blocks, history + window - 1, channels), dtype=torch.bfloat16).transpose(1, 2)
        conv.copy_(torch.randn_like(conv))
        state = paged((blocks, hv, dv, dk), dtype=state_dtype)
        state.copy_(torch.randn_like(state) * 0.1)
        records = (paged((blocks, hv, window, dv + dk + 1)),)
        for record in records:
            record.fill_(float("nan"))
        q = torch.nn.functional.normalize(torch.randn(1, 10, hk, dk, device="npu"), dim=-1).bfloat16()
        k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1).bfloat16()
        v = torch.randn(1, 10, hv, dv, device="npu", dtype=torch.bfloat16)
        g = -torch.rand(1, 10, hv, device="npu")
        beta = torch.rand(1, 10, hv, device="npu", dtype=torch.bfloat16)
        initial, old_conv = state.clone(), conv.clone()
        _, states = reference(q, k, v, g, beta, initial, [0, 5, 10, 10], [1, 4, 0])
        verify_from_activated(q, k, v, g, beta, state, *records, starts, indices, window)
        final, final_conv = initial.clone(), old_conv.clone()
        for row, (source, n, count, destinations) in enumerate(
            [(1, computed, accepted, table_row), (4, 3, 1, [4, 5, 6])]
        ):
            if count == 0:
                continue
            boundary_len = (n // block_size + 1) * block_size - n
            targets = [(destinations[(n + count - 1) // block_size], count)]
            if count >= boundary_len:
                targets.insert(0, (destinations[n // block_size], boundary_len))
            for target, length in targets:
                final[target] = states[row][length]
                final_conv[target, :, :history] = old_conv[source, :, length - 1 : length - 1 + history]
        layers.append(
            SimpleNamespace(
                num_k_heads=hk, tp_size=1, kv_cache=(conv if dim_first else conv.transpose(1, 2), state, *records)
            )
        )
        originals.append((initial, old_conv))
        expected.append((final, final_conv))
    context = recovery.GDNRecoverSSMCommitContext.create(layers, spec_query_len=window, max_num_reqs=4)
    assert context.conv_shared_dim_stride is None
    launches = []

    class RecordingKernel:
        def __init__(self, name, kernel):
            self.name, self.kernel = name, kernel

        def __getitem__(self, grid):
            def run(*args, **kwargs):
                launches.append((self.name, grid, kwargs))
                return self.kernel[grid](*args, **kwargs)

            return run

    for name in ("_commit_gdn_state_kernel", "_commit_gdn_state_grouped_kernel"):
        monkeypatch.setattr(recovery, name, RecordingKernel(name, getattr(recovery, name)))

    def launch():
        context.commit(
            counts,
            indices,
            starts,
            request_indices=mapping,
            block_table=table,
            num_computed_tokens=computed_tokens,
            mamba_block_size=block_size,
        )

    if graph_mode:
        launch()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            launch()
        for cache, (initial, old_conv) in zip(context.layer_caches, originals):
            cache.checkpoint.copy_(initial)
            cache.conv.copy_(old_conv)
        graph.replay()
    else:
        launch()
    grouped = recovery.is_950() and dk == dv == 128 and hv % 2 == 0
    selected = "_commit_gdn_state_grouped_kernel" if grouped else "_commit_gdn_state_kernel"
    assert launches and {name for name, _, _ in launches} == {selected}
    head_groups = hv // recovery.RECOVERSSM_COMMIT_HEAD_TILE if grouped else hv
    for _, grid, kernel_kwargs in launches:
        expected_grid = (
            (dv + kernel_kwargs["BV"] - 1) // kernel_kwargs["BV"] * len(layers) * head_groups * indices.numel(),
        )
        assert grid == expected_grid
    for cache, (final, final_conv) in zip(context.layer_caches, expected):
        torch.testing.assert_close(
            cache.checkpoint,
            final,
            atol=2e-3 if state_dtype == torch.bfloat16 else 3e-5,
            rtol=1e-2 if state_dtype == torch.bfloat16 else 3e-4,
        )
        torch.testing.assert_close(cache.conv, final_conv, atol=0, rtol=0)


@pytest.mark.parametrize("ratio", [1, 2, 3])
@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("accepted", [1, 2, 5, 16])
def test_reverse_commit_grouped_and_ordinary_match_forward_reference(monkeypatch, ratio, align, accepted):
    if not recovery.is_950():
        pytest.skip("Grouped state commit is selected on A5 only")
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: True)
    torch.manual_seed(117)
    blocks, hk, hv, dim, window, history = 4, 2, 2 * ratio, 128, 16, 3
    channels = 2 * hk * dim + hv * dim
    conv = paged((blocks, channels, history + window - 1), dtype=torch.bfloat16)
    conv.copy_(torch.randn_like(conv))
    state = paged((blocks, hv, dim, dim))
    state.copy_(torch.randn_like(state) * 0.1)
    replay = paged((blocks, hv, window, 2 * dim + 1))
    corrections = replay[..., :dim]
    corrections.copy_(torch.randn_like(corrections) * 0.1)
    keys = replay[..., dim : 2 * dim]
    shared_keys = torch.nn.functional.normalize(torch.randn(blocks, hk, window, dim, device="npu"), dim=-1)
    keys.copy_(shared_keys.repeat_interleave(ratio, dim=1))
    decays = replay[..., 2 * dim]
    decays.copy_((-torch.rand_like(decays) * 0.1).exp())
    layer = SimpleNamespace(num_k_heads=hk, tp_size=1, kv_cache=(conv, state, replay))
    context = recovery.GDNRecoverSSMCommitContext.create([layer], spec_query_len=window, max_num_reqs=1)
    initial, old_conv = state.clone(), conv.clone()
    expected, expected_conv = initial.clone(), old_conv.clone()
    recovered = initial[1].double()
    for pos in range(accepted):
        key = keys[1, :, pos].double()
        recovered = (
            recovered * decays[1, :, pos].double()[:, None, None]
            + corrections[1, :, pos].double()[:, :, None] * key[:, None, :]
        )
        if align and pos + 1 == 2:
            expected[1] = recovered.float()
            expected_conv[1, :, :history] = old_conv[1, :, 1 : 1 + history]
    # Exactly reaching the boundary stays on its completed block's page.
    final_page = 2 if align and accepted > 2 else 1
    expected[final_page] = recovered.float()
    expected_conv[final_page, :, :history] = old_conv[1, :, accepted - 1 : accepted - 1 + history]
    kwargs = {}
    if align:
        kwargs = dict(
            block_table=torch.tensor([[1, 2, 3]], dtype=torch.int32, device="npu"),
            num_computed_tokens=torch.tensor([14], dtype=torch.int32, device="npu"),
            mamba_block_size=16,
        )
    results = []
    for grouped in (False, True):
        # Both paths are valid on A5; keep the hardware-selected policy outside this comparison.
        monkeypatch.setattr(recovery, "is_950", lambda grouped=grouped: grouped)
        state.copy_(initial)
        conv.copy_(old_conv)
        context.commit(
            torch.tensor([accepted], dtype=torch.int32, device="npu"),
            torch.tensor([1], dtype=torch.int32, device="npu"),
            torch.tensor([0, window], dtype=torch.int32, device="npu"),
            **kwargs,
        )
        torch.testing.assert_close(state, expected, atol=3e-5, rtol=3e-4)
        torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
        results.append(state.clone())
    torch.testing.assert_close(results[0], results[1], atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("use_defaults", [True, False])
def test_verify_raw_interface_defaults_custom_options_and_out(use_defaults):
    torch.manual_seed(131)
    blocks, tokens, hk, hv, dk, dv, window = 3, 3, 1, 2, 33, 47, 5
    A_log = torch.randn(hv, dtype=torch.float32, device="npu") * 0.1
    dt_bias = torch.randn(hv, dtype=torch.float32, device="npu") * 0.1
    a = torch.randn(tokens, hv, dtype=torch.bfloat16, device="npu")
    b = torch.randn(1, tokens, hv, dtype=torch.bfloat16, device="npu")
    q = torch.randn(1, tokens, hk, dk, dtype=torch.bfloat16, device="npu")
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, hv, dv, dtype=torch.bfloat16, device="npu")
    state = torch.randn(blocks, hv, dv, dk, dtype=torch.float32, device="npu") * 0.1
    initial = state.clone()
    replay = torch.empty(blocks, hv, window, dv + dk + 1, dtype=torch.float32, device="npu")
    starts = torch.tensor([0, tokens], dtype=torch.int32, device="npu")
    indices = torch.tensor([1], dtype=torch.int32, device="npu")
    supplied_out = torch.empty_like(v)

    kwargs = {}
    softplus_beta, softplus_threshold = 1.0, 20.0
    scale = dk**-0.5
    normalize = True
    if not use_defaults:
        softplus_beta, softplus_threshold = 0.5, 2.0
        scale = 0.25
        normalize = False
        kwargs = dict(
            scale=scale,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            use_qk_l2norm_in_kernel=False,
        )

    observed = recovery.gdn_recoverssm_verify(
        A_log,
        a,
        b,
        dt_bias,
        q,
        k,
        v,
        checkpoint_state=state,
        replay_cache=replay,
        query_start_loc=starts,
        state_indices=indices,
        spec_query_len=window,
        out=supplied_out,
        **kwargs,
    )
    x = a.float() + dt_bias
    softplus = torch.nn.functional.softplus(x, beta=softplus_beta, threshold=softplus_threshold)
    g = -A_log.exp() * softplus
    beta = b.float().sigmoid()
    q_ref, k_ref = q.float(), k.float()
    if normalize:
        q_ref = q_ref * torch.rsqrt((q_ref * q_ref).sum(-1, keepdim=True) + 1e-6)
        k_ref = k_ref * torch.rsqrt((k_ref * k_ref).sum(-1, keepdim=True) + 1e-6)
    expected, _ = reference(q_ref, k_ref, v, g.unsqueeze(0), beta, initial, [0, tokens], [1], scale=scale)

    assert observed.data_ptr() == supplied_out.data_ptr()
    torch.testing.assert_close(observed.float(), expected, atol=3e-3, rtol=2e-2)
    torch.testing.assert_close(state, initial, atol=0, rtol=0)


def test_empty_verification_does_not_launch_or_mutate():
    q = torch.empty(1, 0, 1, 128, dtype=torch.bfloat16, device="npu")
    v = torch.empty(1, 0, 2, 128, dtype=torch.bfloat16, device="npu")
    g = torch.empty(1, 0, 2, device="npu")
    state = torch.randn(3, 2, 128, 128, device="npu")
    before = state.clone()
    out = verify_from_activated(
        q,
        q,
        v,
        g,
        g,
        state,
        torch.empty(3, 2, 5, 257, device="npu"),
        torch.zeros(1, dtype=torch.int32, device="npu"),
        torch.empty(0, dtype=torch.int32, device="npu"),
        5,
    )
    assert out.shape == v.shape
    torch.testing.assert_close(state, before, atol=0, rtol=0)


def test_nonempty_capture_rows_with_null_pages_produce_zero_outputs():
    q = torch.randn(1, 3, 1, 128, dtype=torch.bfloat16, device="npu")
    v = torch.randn(1, 3, 2, 128, dtype=torch.bfloat16, device="npu")
    g = torch.zeros(1, 3, 2, device="npu")
    state = torch.randn(3, 2, 128, 128, device="npu")
    before = state.clone()
    out = verify_from_activated(
        q,
        q,
        v,
        g,
        g,
        state,
        torch.empty(3, 2, 5, 257, device="npu"),
        torch.tensor([0, 3], dtype=torch.int32, device="npu"),
        torch.zeros(1, dtype=torch.int32, device="npu"),
        5,
    )
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
    torch.testing.assert_close(state, before, atol=0, rtol=0)
