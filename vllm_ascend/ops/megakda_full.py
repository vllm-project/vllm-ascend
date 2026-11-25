# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 INT8 MegaKDA verification with request-indexed RecoverSSM records."""

import torch
from vllm.triton_utils import tl, triton


def supports(layer):
    from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod

    qkv = layer.in_proj_qkvgfab
    qmethod = getattr(qkv.quant_method, "quant_method", qkv.quant_method)
    heads, dim, hidden = layer.local_num_heads, layer.head_dim, layer.hidden_size
    p = heads * dim
    conv, state = layer.kv_cache
    w = layer.num_spec + 1
    u = layer.recoverssm_records.correction
    return (
        type(qmethod) is AscendW8A8DynamicLinearMethod
        and qmethod.act_quant_type == torch.int8
        and qkv.weight.dtype == torch.int8
        and qkv.weight_scale.dtype == torch.bfloat16
        and qkv.weight.shape in ((hidden, 3 * p), (3 * p // 32, hidden // 16, 16, 32))
        and qkv.weight_scale.numel() == 3 * p
        and layer.o_proj.bias is None
        and layer.o_proj.weight.dtype == torch.bfloat16
        and layer.o_proj.weight.shape == (hidden, p)
        and layer.fused_bfg_proj.weight.dtype == torch.bfloat16
        and layer.fused_bfg_proj.weight.shape == (heads + 2 * p, hidden)
        and state.shape[1:] == (heads, dim, dim)
        and state.dtype == torch.float32
        and len(state) == len(conv)
        and conv.dtype == torch.bfloat16
        and conv.shape[1:] in ((w + 2, 3 * p), (3 * p, w + 2))
        and u.dtype == torch.float32
        and u.is_contiguous()
        and u.shape[1:] == (heads, w, dim)
    )


def bind(layer):
    """Called after weight loading, when the worker binds its RecoverSSM cache."""
    import torch_npu
    from fla_npu.ops.ascendc import npu_mega_recurrent_kda, npu_mega_recurrent_kda_metadata

    from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod

    qkv = layer.in_proj_qkvgfab
    qmethod = getattr(qkv.quant_method, "quant_method", qkv.quant_method)
    heads, dim, hidden = layer.local_num_heads, layer.head_dim, layer.hidden_size
    p = heads * dim
    if type(qmethod) is not AscendW8A8DynamicLinearMethod or qmethod.act_quant_type != torch.int8:
        raise ValueError("MegaKDA on A3 requires dynamic INT8 QKV")
    q = qkv.weight
    if q.dtype != torch.int8 or q.shape not in ((hidden, 3 * p), (3 * p // 32, hidden // 16, 16, 32)):
        raise ValueError(f"Incompatible post-load QKV weight {q.shape}, {q.dtype}")
    if q.ndim == 2 and torch_npu.get_npu_format(q) != 29:
        q = q.reshape(hidden // 16, 16, 3 * p // 32, 32).permute(2, 0, 1, 3).contiguous()

    def plain(t):
        return t if torch_npu.get_npu_format(t) in (0, 2) else torch_npu.npu_format_cast(t, 2)

    aux = layer.fused_bfg_proj
    beta, _, gate = plain(aux.weight).split([heads, p, p])
    conv = layer.get_parameter("ascend_conv1d_weight")
    weights = dict(
        qkv_projection_weight=q,
        qkv_projection_weight_scale=plain(qkv.weight_scale).reshape(-1),
        decay_projection_a_weight=plain(aux.f_a_weight),
        decay_projection_b_weight=plain(aux.f_b_weight),
        beta_projection_weight=beta,
        output_gate_projection_weight=gate,
        output_norm_weight=plain(layer.o_norm.weight).float(),
        output_projection_weight=plain(layer.o_proj.weight),
        conv1d_weight=conv,
        a_log=layer.A_log.float().reshape(heads),
        dt_bias=layer.dt_bias.float().reshape(heads, dim),
    )
    shapes = [
        (3 * p,),
        (dim, hidden),
        (p, dim),
        (heads, hidden),
        (p, hidden),
        (dim,),
        (hidden, p),
        (4, 3 * p),
        (heads,),
        (heads, dim),
    ]
    for (name, tensor), shape in zip(list(weights.items())[1:], shapes, strict=True):
        dtype = torch.float32 if name in ("a_log", "dt_bias", "output_norm_weight") else torch.bfloat16
        if tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError(f"MegaKDA {name} needs {shape} {dtype}, got {tensor.shape} {tensor.dtype}")
        weights[name] = tensor.contiguous()
    u = layer.recoverssm_records.correction
    if not u.is_contiguous() or u.dtype != torch.float32:
        raise ValueError("MegaKDA requires contiguous row-indexed FP32 correction records")
    _, inner1, inner2 = layer.kv_cache[0].shape
    channels = 3 * p
    history = inner1 if inner2 == channels else inner2
    if channels not in (inner1, inner2) or history != u.shape[2] + 2 or layer.kv_cache[0].dtype != torch.bfloat16:
        raise ValueError("MegaKDA needs four-tap BF16 convolution state")
    pending_conv = conv.new_empty((len(u), history, channels))
    layer.mega_weights = weights
    layer.mega_op = npu_mega_recurrent_kda
    layer.mega_metadata = npu_mega_recurrent_kda_metadata
    layer.mega_records = (
        torch.empty_like(u, dtype=torch.float16),
        torch.empty_like(u),
        torch.zeros((), device=u.device, dtype=torch.int32),
        pending_conv,
    )
    return layer.mega_records


@triton.jit
def _mega_plan(
    Starts,
    Slots,
    Counts,
    Indices,
    Accepted,
    OutputMap,
    B: tl.constexpr,
    W: tl.constexpr,
    T: tl.constexpr,
    N: tl.constexpr,
    DIRECT: tl.constexpr,
    TH: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    row = tl.program_id(0)
    ids = tl.arange(0, BLOCK_B)
    starts = tl.load(Starts + ids, ids < B, 0)
    ends = tl.load(Starts + ids + 1, ids < B, 0)
    slots = tl.load(Slots + ids, ids < B, -1)
    live = (ids < B) & (ends > starts) & (slots > 0)
    if row < B:
        slot = tl.load(Slots + row)
        active = (tl.load(Starts + row + 1) > tl.load(Starts + row)) & (slot > 0)
        idx = row
        if DIRECT:
            idx = slot
            if not active:
                # At least one of the first B slots is free for each dummy row.
                # Assign free slots in row order. Live physical checkpoints remain read-only.
                used = tl.full((BLOCK_B,), 0, tl.int32)
                for j in range(B):
                    s = tl.load(Slots + j)
                    valid = (s > 0) & (tl.load(Starts + j + 1) > tl.load(Starts + j))
                    used = used | ((ids == s) & valid).to(tl.int32)
                ordinal = tl.sum(((ids <= row) & ~live).to(tl.int32), 0)
                free = (ids < B) & (used == 0)
                rank = tl.cumsum(free.to(tl.int32), 0)
                idx = tl.sum(tl.where(free & (rank == ordinal), ids, 0), 0)
        tl.store(Indices + row, idx)
        tl.store(Accepted + row, tl.where(active, tl.load(Counts + row), 1))
    if row < T:
        if TH:
            source = tl.where(row < N, row, -1)
        else:
            valid = live & (row >= starts) & (row < ends)
            source = tl.sum(tl.where(valid, ids * W + row - starts + 1, 0), 0) - 1
        tl.store(OutputMap + row, source)


@triton.jit
def _mega_pack(
    Hidden,
    Packed,
    Conv,
    ScratchConv,
    State,
    ScratchState,
    Starts,
    Slots,
    Indices,
    T: tl.constexpr,
    W: tl.constexpr,
    HIDDEN: tl.constexpr,
    HIST: tl.constexpr,
    C: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    HS0: tl.constexpr,
    HS1: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    CS2: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    SS2: tl.constexpr,
    SS3: tl.constexpr,
    TH: tl.constexpr,
    DIRECT: tl.constexpr,
    TILE: tl.constexpr,
):
    task = tl.program_id(0)
    lane = tl.arange(0, TILE)
    channel_tiles: tl.constexpr = triton.cdiv(C, TILE)
    hidden_tiles: tl.constexpr = triton.cdiv(HIDDEN, TILE)
    conv_tiles: tl.constexpr = HIST * channel_tiles
    x_tiles: tl.constexpr = 0 if TH else W * hidden_tiles
    state_tiles: tl.constexpr = 0 if DIRECT else triton.cdiv(HEADS * D * D, TILE)
    tasks_per_row: tl.constexpr = conv_tiles + x_tiles + state_tiles
    row = task // tasks_per_row
    tile = task % tasks_per_row
    if tile < conv_tiles:
        history = tile // channel_tiles
        channel = (tile % channel_tiles) * TILE + lane
        physical = tl.maximum(0, tl.load(Slots + row)).to(tl.int64)
        slot = tl.load(Indices + row).to(tl.int64)
        conv_values = tl.load(Conv + physical * CS0 + history * CS1 + channel * CS2, channel < C, 0)
        tl.store(ScratchConv + (slot * HIST + history) * C + channel, conv_values, channel < C)
    elif tile < conv_tiles + x_tiles:
        local_token = (tile - conv_tiles) // hidden_tiles
        col = ((tile - conv_tiles) % hidden_tiles) * TILE + lane
        token = tl.minimum(tl.load(Starts + row) + local_token, T - 1)
        values = tl.load(Hidden + token * HS0 + col * HS1, col < HIDDEN, 0)
        tl.store(Packed + (row * W + local_token) * HIDDEN + col, values, col < HIDDEN)
    else:
        if not DIRECT:
            i = (tile - conv_tiles - x_tiles) * TILE + lane
            pos = (i // (D * D)) * SS1 + ((i // D) % D) * SS2 + (i % D) * SS3
            page = tl.maximum(0, tl.load(Slots + row)).to(tl.int64)
            state_values = tl.load(State + page * SS0 + pos, i < HEADS * D * D, 0)
            tl.store(ScratchState + row * HEADS * D * D + i, state_values, i < HEADS * D * D)


@triton.jit
def _mega_restore(
    Out,
    Result,
    Conv,
    U,
    K,
    Decay,
    Pending,
    RecordU,
    RecordK,
    RecordDecay,
    Flag,
    Indices,
    OutputMap,
    B: tl.constexpr,
    T: tl.constexpr,
    HIDDEN: tl.constexpr,
    CONV_SIZE: tl.constexpr,
    REPLAY_SIZE: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    TILE: tl.constexpr,
):
    task = tl.program_id(0)
    lane = tl.arange(0, TILE)
    nc: tl.constexpr = triton.cdiv(CONV_SIZE, TILE)
    nr: tl.constexpr = triton.cdiv(REPLAY_SIZE, TILE)
    no: tl.constexpr = triton.cdiv(HIDDEN, TILE)
    if task < B * nc:
        row = task // nc
        i = (task % nc) * TILE + lane
        slot = tl.load(Indices + row).to(tl.int64)
        val = tl.load(Conv + slot * CONV_SIZE + i, i < CONV_SIZE, 0)
        tl.store(Pending + row * CONV_SIZE + i, val, i < CONV_SIZE)
        if task == 0:
            tl.store(Flag, 1)
    elif task < B * (nc + nr):
        r = task - B * nc
        row = r // nr
        i = (r % nr) * TILE + lane
        slot = tl.load(Indices + row).to(tl.int64)
        src = slot * REPLAY_SIZE + i
        dst = row * REPLAY_SIZE + i
        mask = i < REPLAY_SIZE
        tl.store(RecordU + dst, tl.load(U + src, mask, 0), mask)
        tl.store(RecordK + dst, tl.load(K + src, mask, 0), mask)
        tl.store(RecordDecay + dst, tl.load(Decay + src, mask, 0), mask)
    else:
        r = task - B * (nc + nr)
        token = r // no
        col = (r % no) * TILE + lane
        packed_token = tl.load(OutputMap + token)
        val = tl.load(Out + tl.maximum(packed_token, 0) * HIDDEN + col, (packed_token >= 0) & (col < HIDDEN), 0)
        tl.store(Result + token * OS0 + col * OS1, val, col < HIDDEN)


def _view_key(tensor):
    return (tensor.untyped_storage(), tensor.storage_offset(), tensor.shape, tensor.stride(), tensor.dtype)


def _native_verify(layer, hidden, output, conv, state, starts, slots, counts):
    import importlib

    from fla_npu.ops.ascendc import npu_mega_recurrent_kda

    # Resolve the registered namespace from the active FLA wheel layer.
    provider = npu_mega_recurrent_kda.__module__.split(".ops.ascendc")[0]
    stable = importlib.import_module(provider + ".ops.ascendc._stable")
    native = stable._op("npu_mega_recurrent_kda_verify_a3")
    b, w = slots.numel(), layer.num_spec + 1
    x = hidden.view(b, w, -1) if hidden.shape[0] == b * w else hidden
    k, decay, flag, pending = layer.mega_records
    u = layer.recoverssm_records.correction
    args = layer.mega_weights
    native(
        x,
        args["qkv_projection_weight"],
        args["decay_projection_a_weight"],
        args["decay_projection_b_weight"],
        args["beta_projection_weight"],
        args["output_gate_projection_weight"],
        args["output_norm_weight"],
        args["output_projection_weight"],
        args["conv1d_weight"],
        pending[:b],
        state,
        u[:b],
        k[:b],
        decay[:b],
        slots,
        counts,
        args["a_log"],
        args["dt_bias"],
        None,
        None,
        args["qkv_projection_weight_scale"],
        layer.head_dim**-0.5,
        layer.gate_lower_bound,
        layer.o_norm.eps,
        conv,
        starts,
        flag.view(1),
        output.view_as(x),
        stable._current_stream_ptr(),
    )
    return output


def verify(layer, hidden, meta, output=None, cache=None):
    desc = meta.spec_decode_metadata.spec_causal_conv1d
    starts = desc.query_start_loc.contiguous()
    slots = desc.cache_indices.reshape(-1).contiguous()
    counts = desc.num_accepted_tokens.reshape(-1).contiguous()
    b, w = slots.numel(), layer.num_spec + 1
    t, channels = hidden.shape
    conv, state = layer.kv_cache
    if conv.size(-1) != 3 * layer.local_projection_size:
        conv = conv.transpose(1, 2)
    capacity, heads, d, _ = state.shape
    hist, c = conv.shape[1:]
    direct = state.stride()[1:] == (d * d, d, 1) and state.stride(0) >= heads * d * d and capacity >= b
    th = layer.model_config.enforce_eager
    if (
        not th
        and w == 4
        and direct
        and hidden.is_contiguous()
        and conv.stride()[1:] == (c, 1)
        and conv.stride(0) >= hist * c
        and starts.dtype == slots.dtype == counts.dtype == torch.int32
        and (output is None or output.is_contiguous())
    ):
        if output is None:
            output = torch.empty_like(hidden)
        return _native_verify(layer, hidden, output, conv, state, starts, slots, counts)
    n = meta.num_spec_decode_tokens if th else t
    if th:
        assert b == meta.num_spec_decodes
    if cache is None:
        cache = {}
    stream = torch.npu.current_stream(hidden.device).npu_stream
    key = ("plan", stream, _view_key(starts), _view_key(slots), _view_key(counts), t, n, w, direct, th, capacity)
    plan = cache.get(key)
    if plan is None:
        indices = slots.new_empty(b, dtype=torch.int32)
        accepted = torch.empty_like(indices)
        output_map = indices.new_empty(t)
        _mega_plan[(max(b, t),)](
            starts, slots, counts, indices, accepted, output_map, b, w, t, n, direct, th, triton.next_power_of_2(b)
        )
        metadata = layer.mega_metadata(starts, tokens=n) if th else None
        plan = cache[key] = indices, accepted, output_map, metadata
    indices, accepted, output_map, metadata = plan
    u = layer.recoverssm_records.correction
    k, decay, mode, pending = layer.mega_records
    size = capacity if direct else b
    pool_key = ("scratch", stream, size, hist, c, heads, w, d, hidden.device)
    pools = cache.get(pool_key)
    if pools is None:
        pools = cache[pool_key] = tuple(a.new_empty((size, *a.shape[1:])) for a in (conv, u, k, decay))
    c_buf, u_buf, k_buf, d_buf = pools
    checkpoint = state if direct else state.new_empty((b, heads, d, d))
    x = hidden[:n].contiguous() if th else hidden.new_empty((b, w, channels))
    pack_tile = 2048
    tasks = hist * triton.cdiv(c, pack_tile)
    if not th:
        tasks += w * triton.cdiv(channels, pack_tile)
    if not direct:
        tasks += triton.cdiv(heads * d * d, pack_tile)
    _mega_pack[(tasks * b,)](
        hidden,
        x,
        conv,
        c_buf,
        state,
        checkpoint,
        starts,
        slots,
        indices,
        t,
        w,
        channels,
        hist,
        c,
        heads,
        d,
        *hidden.stride(),
        *conv.stride(),
        *state.stride(),
        th,
        direct,
        pack_tile,
    )
    out, _, _, _, _ = layer.mega_op(
        hidden_states=x,
        **layer.mega_weights,
        conv_state=c_buf,
        checkpoint_state=checkpoint,
        replay_u=u_buf,
        replay_k=k_buf,
        replay_decay=d_buf,
        state_indices=indices,
        conv_num_accepted_tokens=accepted,
        cu_seqlens=starts if th else None,
        metadata=metadata,
        scale=layer.head_dim**-0.5,
        lower_bound=layer.gate_lower_bound,
        rms_norm_eps=layer.o_norm.eps,
    )
    if output is None:
        output = torch.empty_like(hidden)
    tile = 2048
    tasks = b * (triton.cdiv(hist * c, tile) + triton.cdiv(heads * w * d, tile)) + t * triton.cdiv(channels, tile)
    _mega_restore[(tasks,)](
        out,
        output,
        c_buf,
        u_buf,
        k_buf,
        d_buf,
        pending,
        u,
        k,
        decay,
        mode,
        indices,
        output_map,
        b,
        t,
        channels,
        hist * c,
        heads * w * d,
        *output.stride(),
        tile,
    )
    return output
