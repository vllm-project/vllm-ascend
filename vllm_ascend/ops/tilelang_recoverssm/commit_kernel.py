# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 device-side acceptance plan and cross-layer state/conv commit."""

from functools import lru_cache

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language.kernel import get_thread_binding


def plan_kernel(desc, batch, capacity, width, window, block_size, mapped, align):
    sizes, strides, types = zip(*desc)

    @T.prim_func
    def plan(
        A: T.Tensor((sizes[0],), types[0]),
        PageIds: T.Tensor((sizes[1],), types[1]),
        Offsets: T.Tensor((sizes[2],), types[2]),
        M: T.Tensor((sizes[3],), types[3]),
        Table: T.Tensor((sizes[4],), types[4]),
        N: T.Tensor((sizes[5],), types[5]),
        P: T.Tensor((6 * capacity,), "int32"),
    ):
        with T.Kernel((batch + 31) // 32) as block:  # noqa: SIM117 - preserve nested TileLang IR scopes
            with T.SimtVF(threads=32):
                row = block * 32 + get_thread_binding(0)
                if row < batch:
                    request = T.cast(M[row * strides[3][0]], "int64") if mapped else T.cast(row, "int64")
                    source = PageIds[row * strides[1][0]]
                    length = T.min(
                        window,
                        T.min(
                            T.max(A[request * strides[0][0]], 0),
                            Offsets[(row + 1) * strides[2][0]] - Offsets[row * strides[2][0]],
                        ),
                    )
                    boundary = T.alloc_var("int32", init=0)
                    recovery = T.alloc_var("int32", init=0)
                    final = T.alloc_var("int32", init=T.cast(source, "int32"))
                    if align:
                        start = T.cast(N[request * strides[5][0]], "int64")
                        end = start + length
                        col = T.min(T.max(end - 1, 0) // block_size, width - 1)
                        final = T.cast(Table[request * strides[4][0] + col * strides[4][-1]], "int32")
                        boundary_col = start // block_size
                        next_end = (boundary_col + 1) * block_size
                        recovery = T.cast(next_end - start, "int32")
                        if end >= next_end and boundary_col < width:
                            boundary = T.cast(Table[request * strides[4][0] + boundary_col * strides[4][-1]], "int32")
                    valid = source > 0 and length > 0
                    has_boundary = boundary > 0 and recovery > 0
                    # At an exact block end the boundary page is the final
                    # destination. It needs only the ordinary final write.
                    # Preserve boundary metadata for the common compact ABI.
                    needs_snapshot = has_boundary and boundary != final
                    P[row] = T.cast(T.if_then_else(valid, length, 0), "int32")
                    P[capacity + row] = T.cast(T.if_then_else(valid and not needs_snapshot, length, 0), "int32")
                    P[2 * capacity + row] = T.cast(T.if_then_else(valid and needs_snapshot, length, 0), "int32")
                    P[3 * capacity + row] = T.if_then_else(valid, final, 0)
                    P[4 * capacity + row] = T.if_then_else(valid, boundary, 0)
                    P[5 * capacity + row] = T.if_then_else(valid, recovery, 0)

    return plan


def commit_kernel(
    batch,
    layers,
    pages,
    hv,
    window,
    channels,
    hist,
    capacity,
    index_layout,
    state_dtype,
    conv_dtype,
    ss,
    rs,
    align,
    conv,
    value_tile=None,
    threads=512,
    cap=None,
):
    value_tile = (64 if window <= 4 else 32) if value_tile is None else value_tile
    cap = (64 if window <= 4 else 128) if cap is None else cap
    size, index_stride, index_dtype = index_layout
    state_tasks = hv * (128 // value_tile)
    rows_per_warp_group = threads // 128
    reg_rows = value_tile // rows_per_warp_group
    conv_tasks = (channels + threads - 1) // threads if conv else 0
    per_layer = state_tasks + conv_tasks
    tasks = batch * layers * per_layer
    blocks = min(cap, tasks) if cap else tasks

    @T.prim_func
    def commit(
        ST: T.Tensor((4 * layers,), "int64"),
        CT: T.Tensor((4 * layers,), "int64"),
        PageIds: T.Tensor((size,), index_dtype),
        P: T.Tensor((6 * capacity,), "int32"),
    ):
        with T.Kernel(blocks) as block:  # noqa: SIM117 - preserve nested TileLang IR scopes
            with T.SimtVF(threads=threads):
                tid = get_thread_binding(0)
                lane = tid % 128
                for wave in T.serial((tasks - 1 - block) // blocks + 1):
                    task = wave * blocks + block
                    row = task // (layers * per_layer)
                    layer = (task // per_layer) % layers
                    part = task % per_layer
                    count = P[row]
                    src = T.cast(PageIds[row * index_stride[0]], "int64")
                    dst = T.cast(P[3 * capacity + row], "int64")
                    mid = T.cast(P[4 * capacity + row], "int64")
                    cut = P[5 * capacity + row]
                    boundary = align and mid > 0 and mid < pages and cut > 0 and cut <= count and mid != dst
                    if count > 0 and src > 0 and src < pages and dst > 0 and dst < pages:
                        if part < state_tasks:
                            head = part // (128 // value_tile)
                            v0 = (part % (128 // value_tile)) * value_tile + tid // 128
                            state = T.make_tensor_from_addr(
                                ST[0 * layers + layer],
                                (pages, hv, 128, 128),
                                state_dtype,
                                strides=(ST[1 * layers + layer], ss[0], ss[1], ss[2]),
                            )
                            replay = T.make_tensor_from_addr(
                                ST[2 * layers + layer],
                                (pages, hv, window, 257),
                                "float32",
                                strides=(ST[3 * layers + layer], rs[0], rs[1], rs[2]),
                            )
                            reg = T.alloc_local((reg_rows,), "float32")
                            for j in T.unroll(reg_rows):
                                reg[j] = T.cast(state[src, head, v0 + j * rows_per_warp_group, lane], "float32")
                            for pos in T.unroll(window) if window <= 4 else T.serial(count):
                                if window > 4 or pos < count:
                                    decay = replay[src, head, pos, 256]
                                    key = replay[src, head, pos, 128 + lane]
                                    for j in T.unroll(reg_rows):
                                        reg[j] = (
                                            reg[j] * decay + replay[src, head, pos, v0 + j * rows_per_warp_group] * key
                                        )
                                    if boundary and pos + 1 == cut:
                                        for j in T.unroll(reg_rows):
                                            state[mid, head, v0 + j * rows_per_warp_group, lane] = T.cast(
                                                reg[j], state_dtype
                                            )
                            for j in T.unroll(reg_rows):
                                state[dst, head, v0 + j * rows_per_warp_group, lane] = T.cast(reg[j], state_dtype)
                        elif conv:
                            channel = (part - state_tasks) * threads + tid
                            if channel < channels:
                                cache = T.make_tensor_from_addr(
                                    CT[0 * layers + layer],
                                    (pages, channels, hist + window - 1),
                                    conv_dtype,
                                    strides=(
                                        CT[1 * layers + layer],
                                        CT[2 * layers + layer],
                                        CT[3 * layers + layer],
                                    ),
                                )
                                final_hist = T.alloc_local((hist,), conv_dtype)
                                mid_hist = T.alloc_local((hist,), conv_dtype)
                                for j in T.unroll(hist):
                                    final_hist[j] = cache[src, channel, count - 1 + j]
                                    if boundary:
                                        mid_hist[j] = cache[src, channel, cut - 1 + j]
                                if boundary:
                                    for j in T.unroll(hist):
                                        cache[mid, channel, j] = mid_hist[j]
                                for j in T.unroll(hist):
                                    cache[dst, channel, j] = final_hist[j]

    return commit


@lru_cache(maxsize=64)
def compiled_plan(*config):
    return tilelang.compile(plan_kernel(*config), target="ascend")


@lru_cache(maxsize=64)
def compiled_commit(*config, split_boundary=False, row_group=4, conv_dim_stride=None):
    # Block DMA and raw SIMD require contiguous K rows and packed replay records.
    # Keep the existing strided SIMT path for FP32 checkpoints and long windows.
    if len(config) != 15:
        return tilelang.compile(commit_kernel(*config), target="ascend")
    window, state_dtype, state_strides, replay_strides = config[4], config[9], config[11], config[12]
    if window > 4 or state_dtype != "bfloat16" or state_strides[-2:] != (128, 1) or replay_strides[-1] != 1:
        return tilelang.compile(commit_kernel(*config), target="ascend")
    from .compact import compact_kernel
    from .state_commit import state_commit_kernel

    state_config = (*config[:14], False)
    if config[13] and split_boundary:
        # Reuse the plan's disjoint on-device filters. Do not copy lengths to
        # CPU or allocate a new graph buffer. Keep single-launch as the default:
        # GLM's measurements include regressions from the extra empty launch.
        plain_config = (*state_config[:13], False, False)
        plain = tilelang.compile(
            state_commit_kernel(*plain_config, commit_lens_row=1, row_group=row_group),
            target="ascend",
            pass_configs={"tl.enable_auto_schedule": False},
        )
        boundary = tilelang.compile(
            state_commit_kernel(*state_config, commit_lens_row=2, row_group=row_group),
            target="ascend",
            pass_configs={"tl.enable_auto_schedule": False},
        )

        def state(*inputs):
            plain(*inputs)
            boundary(*inputs)
    else:
        state = tilelang.compile(
            state_commit_kernel(*state_config, row_group=row_group),
            target="ascend",
            pass_configs={"tl.enable_auto_schedule": False},
        )
    if not config[14]:
        return state
    compact = tilelang.compile(
        compact_kernel(*config, shared_dim_stride=conv_dim_stride),
        target="ascend",
        pass_configs={"tl.enable_auto_schedule": False},
    )

    def run(*inputs):
        compact(*inputs)
        state(*inputs)

    return run
