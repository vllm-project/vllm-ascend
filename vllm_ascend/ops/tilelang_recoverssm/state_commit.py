# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline BF16 checkpoints and accepted replay records through two UB slots.

The caller disables TileLang auto scheduling for this manually synchronized
kernel. Each slot follows MTE2 -> V -> MTE3 -> MTE2; the final waits drain both
slots, including slots skipped by empty requests. No checkpoint/replay layout
or recurrence arithmetic changes are needed.
"""

import tilelang.ascend.language as T


def state_task_geometry(batch, layers, heads, value_tile, cap, group_limit=4):
    """Reuse one request/layer metadata load across a bounded head group.

    Keep at least the original launch extent and cap the group at four state
    tiles, avoiding the long-tail imbalance of assigning all heads to a core.
    """
    parts = heads * (128 // value_tile)
    target = min(cap, batch * layers * parts)
    group = next(
        g for g in range(min(group_limit, parts), 0, -1) if parts % g == 0 and batch * layers * (parts // g) >= target
    )
    groups = parts // group
    return group, groups, min(cap, batch * layers * groups)


def state_commit_kernel(
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
    value_tile=128,
    threads=256,
    cap=64,
    commit_lens_row=0,
    row_group=4,
):
    if commit_lens_row not in (0, 1, 2):
        raise ValueError("commit length row must be full, ordinary or boundary")
    if row_group not in (2, 4) or value_tile % row_group:
        raise ValueError("state row group must be 2/4 and divide the value tile")
    size, index_stride, index_dtype = index_layout
    parts_per_task, part_groups, blocks = state_task_geometry(batch, layers, hv, value_tile, cap)
    tasks = batch * layers * part_groups

    @T.macro
    def recur(state_ub, smid, records, slot, count, vbase, snapshot):
        with T.SimdVF():
            # These records are identical for every value row.
            # Keep their vectors live instead of loading and
            # permuting K again for each pair of state rows.
            keys0 = T.simd.alloc_local((window,), "float32")
            keys1 = T.simd.alloc_local((window,), "float32")
            decays = T.simd.alloc_local((window,), "float32")
            for pos in T.unroll(window, explicit=True):
                if pos < count:
                    decays[pos] = T.simd.vld(records[slot, pos, 256], dist="BRC_B32")
                    # Packed BF16 conversion selects even/odd
                    # lanes, so apply the same permutation to K.
                    key0, key1 = T.simd.vdintlv(
                        T.simd.vld(records[slot, pos, 128]),
                        T.simd.vld(records[slot, pos, 192]),
                    )
                    keys0[pos] = key0
                    keys1[pos] = key1
            # Four independent V rows share the same K/decay vectors and
            # one acceptance branch. Recurrence order within each row is
            # unchanged: FP32 multiply, multiply, then add, from pos 0 onward.
            for vr in T.serial(value_tile // row_group):
                even_state = T.simd.alloc_local((row_group,), "float32")
                odd_state = T.simd.alloc_local((row_group,), "float32")
                for r in T.unroll(row_group, explicit=True):
                    packed = T.simd.vld(state_ub[slot, vr * row_group + r, 0])
                    even_state[r] = T.simd.vcvt(packed, "float32", part=0)
                    odd_state[r] = T.simd.vcvt(packed, "float32", part=1)
                for pos in T.unroll(window, explicit=True):
                    if pos < count:
                        decay = decays[pos]
                        key0 = keys0[pos]
                        key1 = keys1[pos]
                        for r in T.unroll(row_group, explicit=True):
                            delta = T.simd.vld(records[slot, pos, vbase + vr * row_group + r], dist="BRC_B32")
                            even_state[r] = T.simd.vadd(T.simd.vmul(even_state[r], decay), T.simd.vmul(delta, key0))
                            odd_state[r] = T.simd.vadd(T.simd.vmul(odd_state[r], decay), T.simd.vmul(delta, key1))
                        if snapshot > 0 and pos + 1 == snapshot:
                            for r in T.unroll(row_group, explicit=True):
                                even = T.reinterpret("uint16x128", T.simd.vcvt(even_state[r], "bfloat16", part=0))
                                odd = T.reinterpret("uint16x128", T.simd.vcvt(odd_state[r], "bfloat16", part=1))
                                T.simd.vsts(
                                    smid[slot, vr * row_group + r, 0],
                                    T.reinterpret("bfloat16x128", T.simd.vor(even, odd)),
                                    dist="NORM_B16",
                                )
                for r in T.unroll(row_group, explicit=True):
                    even = T.reinterpret("uint16x128", T.simd.vcvt(even_state[r], "bfloat16", part=0))
                    odd = T.reinterpret("uint16x128", T.simd.vcvt(odd_state[r], "bfloat16", part=1))
                    T.simd.vsts(
                        state_ub[slot, vr * row_group + r, 0],
                        T.reinterpret("bfloat16x128", T.simd.vor(even, odd)),
                        dist="NORM_B16",
                    )

    @T.prim_func
    def commit(
        ST: T.Tensor((4 * layers,), "int64"),
        CT: T.Tensor((4 * layers,), "int64"),
        PageIds: T.Tensor((size,), index_dtype),
        P: T.Tensor((6 * capacity,), "int32"),
    ):
        with T.Kernel(blocks) as block:  # noqa: SIM117 - preserve TileLang scopes
            state_ub = T.alloc_shared((2, value_tile, 128), state_dtype)
            smid = T.alloc_shared((2, value_tile, 128), state_dtype)
            records = T.alloc_shared((2, window, 264), "float32")
            # A 264-float pitch keeps each packed 257-float replay row 32-byte aligned.
            # The checkpoint is read and updated in the same slot. A previous
            # writeback must finish before that slot is reused two waves later.
            for slot in T.unroll(2):
                T.ascend_set_flag("MTE3_MTE2", slot)
            for wave in T.serial((tasks - 1 - block) // blocks + 1):
                task = wave * blocks + block
                row = task // (layers * part_groups)
                layer = (task // part_groups) % layers
                part_group = task % part_groups
                count = P[commit_lens_row * capacity + row]
                if count > 0:
                    src = T.cast(PageIds[row * index_stride[0]], "int64")
                    dst = T.cast(P[3 * capacity + row], "int64")
                    mid = T.cast(P[4 * capacity + row], "int64") if align else T.int64(0)
                    cut = P[5 * capacity + row] if align else 0
                    boundary = align and mid > 0 and mid < pages and cut > 0 and cut <= count and mid != dst
                    if src > 0 and src < pages and dst > 0 and dst < pages:
                        state = T.make_tensor_from_addr(
                            ST[layer],
                            (pages, hv, 128, 128),
                            state_dtype,
                            strides=(ST[layers + layer], ss[0], ss[1], ss[2]),
                        )
                        replay = T.make_tensor_from_addr(
                            ST[2 * layers + layer],
                            (pages, hv, window, 257),
                            "float32",
                            strides=(ST[3 * layers + layer], rs[0], rs[1], rs[2]),
                        )
                        for local_part in T.serial(parts_per_task):
                            part = part_group * parts_per_task + local_part
                            head = part // (128 // value_tile)
                            vbase = (part % (128 // value_tile)) * value_tile
                            slot = (wave * parts_per_task + local_part) % 2
                            T.ascend_wait_flag("MTE3_MTE2", slot)
                            T.copy(state[src, head, vbase : vbase + value_tile, 0:128], state_ub[slot, :, :])
                            # One strided DMA of the accepted records, not window
                            # separate copies and not reads of rejected records.
                            T.copy(replay[src, head, 0:count, 0:257], records[slot, 0:count, 0:257])
                            T.ascend_set_flag("MTE2_V", slot)
                            T.ascend_wait_flag("MTE2_V", slot)
                            if boundary and cut < count:
                                for snapshot in T.unroll(1, window, explicit=True):
                                    if cut == snapshot:
                                        recur(state_ub, smid, records, slot, count, vbase, snapshot)
                            else:
                                recur(state_ub, smid, records, slot, count, vbase, 0)
                            T.ascend_set_flag("V_MTE3", slot)
                            T.ascend_wait_flag("V_MTE3", slot)
                            if boundary:
                                if window == 1:
                                    T.copy(state_ub[slot, :, :], state[mid, head, vbase : vbase + value_tile, 0:128])
                                else:
                                    if cut == count:
                                        T.copy(
                                            state_ub[slot, :, :], state[mid, head, vbase : vbase + value_tile, 0:128]
                                        )
                                    else:
                                        T.copy(smid[slot, :, :], state[mid, head, vbase : vbase + value_tile, 0:128])
                            T.copy(state_ub[slot, :, :], state[dst, head, vbase : vbase + value_tile, 0:128])
                            T.ascend_set_flag("MTE3_MTE2", slot)
            for slot in T.unroll(2):
                T.ascend_wait_flag("MTE3_MTE2", slot)

    names = ("state_commit", "state_plain", "state_boundary")
    return commit.with_attr(
        "global_symbol", names[commit_lens_row] + "_r" + str(row_group) + "_g" + str(parts_per_task)
    )
