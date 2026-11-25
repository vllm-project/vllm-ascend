# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from . import (
    FRAGMENTED_INDEX_SEGMENT,
    INDEX_COVERAGE_SCALE,
    INDEX_VOTE_STRIDE,
    LINKED_PAIR_LEADER,
    SHARED_INDEX_SEGMENT,
    SHARED_INDEX_WEIGHT,
    use_four_request_index_scan,
    use_segmented_index_scan,
)


def state_update(
    batch,
    pages,
    page,
    context,
    tail,
    parts,
    table_columns,
    *,
    tail_layout=None,
    index_norm_eps=1e-5,
    length_offset=0,
    split_descriptors=False,
):
    mode = "div"
    kind = "quant_div"
    precise = kind != "legacy"
    raw_round = "ROUND_R" if kind == "vec_even" else "ROUND_A"
    precise_pool = kind in ("vec_pool", "vec_quant", "quant_div", "quant_ieee")
    precise_quant = kind in ("vec_quant", "quant_div", "quant_ieee")
    pool_page_rows = page // 4
    active_columns = context // page
    dim, capacity, eps = 128, 4, index_norm_eps
    tail_count, tail_stride, tail_id_stride = (batch, 4 * 256, 1) if tail_layout is None else tail_layout

    share_keys = use_segmented_index_scan(batch, context)
    descriptor_width = context // 512 + context // 16384
    count_qualifying = share_keys and batch < 512
    vote_rows = (1 if count_qualifying else (batch + descriptor_width - 1) // descriptor_width) if share_keys else 0
    # A shared segment saves more transfers than widening one fragmented segment.
    # Select the segmented scan above one quarter weighted coverage.

    @T.macro
    def describe_segments(
        bx, table_row, peer_row, descriptors, shared_length, Runs, compare_peer, mapped, leader_row, compare_leader
    ):
        with T.SimtVF(threads=32):
            needs_ordinary = T.alloc_fragment((1,), T.int32)
            votes = T.alloc_fragment((1,), T.int32)
            if share_keys:
                for lane in T.Parallel(1):
                    votes[lane] = 0
                    if not count_qualifying:
                        needs_ordinary[lane] = 0
            for segment in T.serial(context // 16384):
                scattered = T.alloc_fragment((1, 32), T.int32)
                total_scatter = T.alloc_fragment((1,), T.int32)
                for lane in T.Parallel(32):
                    is_scattered = T.cast(descriptors[segment * 32 + lane] < 0, T.int32)
                    if compare_peer:
                        mismatches = T.alloc_var(T.int32, init=0)
                        for page_in_run in T.unroll(4):
                            column = (segment * 32 + lane) * 4 + page_in_run
                            mismatches += T.cast(table_row[column] != peer_row[column], T.int32)
                            if compare_leader:
                                mismatches += 256 * T.cast(table_row[column] != leader_row[column], T.int32)
                        # Bits 0:6 count scattered runs, 6:14 count pair mismatches, and the rest compare leaders.
                        scattered[0, lane] = 64 * mismatches + is_scattered
                    else:
                        scattered[0, lane] = is_scattered
                T.reduce_sum(scattered, total_scatter, dim=1)
                for lane in T.Parallel(1):
                    total = total_scatter[lane] % 16384 if compare_leader else total_scatter[lane]
                    if share_keys:
                        flag = T.if_then_else(
                            total == 32 and shared_length >= (segment + 1) * 16384,
                            SHARED_INDEX_SEGMENT,
                            T.if_then_else(
                                total % 64 == 32, FRAGMENTED_INDEX_SEGMENT, T.cast(total % 64 == 0, T.int32)
                            ),
                        )
                        if compare_leader:
                            Runs[bx, context // 512 + segment] = T.if_then_else(
                                flag == SHARED_INDEX_SEGMENT and total_scatter[lane] == 32, LINKED_PAIR_LEADER, flag
                            )
                        else:
                            Runs[bx, context // 512 + segment] = flag
                        if (segment + 1) * 4096 <= mapped:
                            if count_qualifying:
                                votes[lane] += T.if_then_else(
                                    flag == SHARED_INDEX_SEGMENT,
                                    SHARED_INDEX_WEIGHT * INDEX_VOTE_STRIDE + 1,
                                    T.if_then_else(flag == FRAGMENTED_INDEX_SEGMENT, INDEX_VOTE_STRIDE, 0),
                                )
                            else:
                                if flag != SHARED_INDEX_SEGMENT and flag != FRAGMENTED_INDEX_SEGMENT:
                                    needs_ordinary[lane] = 1
                                votes[lane] += T.if_then_else(
                                    flag == SHARED_INDEX_SEGMENT,
                                    (INDEX_COVERAGE_SCALE * SHARED_INDEX_WEIGHT - 1) * INDEX_VOTE_STRIDE + 1,
                                    T.if_then_else(
                                        flag == FRAGMENTED_INDEX_SEGMENT,
                                        (INDEX_COVERAGE_SCALE - 1) * INDEX_VOTE_STRIDE,
                                        -INDEX_VOTE_STRIDE,
                                    ),
                                )
                    else:
                        Runs[bx, context // 512 + segment] = T.cast(total == 0, T.int32)

            if share_keys:
                for lane in T.Parallel(1):
                    if count_qualifying:
                        if votes[lane] > 0:
                            T.atomic_add(Runs[batch, 0], votes[lane])
                    else:
                        # Bit zero records a full descriptor not covered by a fragmented or shared scan.
                        Runs[batch + bx // descriptor_width, bx % descriptor_width] = (
                            votes[lane] * 2 + needs_ordinary[lane]
                        )

    @T.macro
    def publish_descriptors(bx, ITable, Len, Runs):
        table_row = T.alloc_shared((active_columns,), T.int32)
        descriptors = T.alloc_shared((context // 512,), T.int32)
        T.copy(ITable[bx, :active_columns], table_row)
        with T.SimtVF(threads=64):
            for group in T.Parallel(context // 512):
                first = table_row[group * 4]
                errors = T.alloc_var(T.int32, init=0)
                for n in T.unroll(3):
                    errors += T.cast(table_row[group * 4 + n + 1] != first + n + 1, T.int32)
                descriptors[group] = T.if_then_else((first >= 0) & (first <= pages - 4) & (errors == 0), first, -1)
        T.copy(descriptors, Runs[bx, : context // 512])
        if share_keys:
            if count_qualifying:
                own_length = Len[length_offset + bx]
                mapped = T.ceildiv(own_length // 4, 32) * 32
                peer = T.min(bx // 2 * 2 + 1 - bx % 2, batch - 1)
                shared_length = T.if_then_else(
                    peer != bx and own_length >= 16384 and table_row[0] == ITable[peer, 0],
                    T.min(own_length, Len[length_offset + peer]),
                    0,
                )
            else:
                mapped = T.ceildiv(Len[length_offset + bx] // 4, 32) * 32
                peer = T.min(bx // 2 * 2 + 1 - bx % 2, batch - 1)
                shared_length = T.if_then_else(peer == bx, 0, T.min(Len[length_offset + bx], Len[length_offset + peer]))
            peer_row = T.alloc_shared((active_columns,), T.int32)
            if shared_length >= 16384 and (count_qualifying or ITable[peer, 0] == ITable[bx, 0]):
                T.copy(ITable[peer, :active_columns], peer_row)
                if use_four_request_index_scan(batch, context):
                    if bx % 4 == 0 and bx + 3 < batch and ITable[T.min(bx + 2, batch - 1), 0] == table_row[0]:
                        leader_row = T.alloc_shared((active_columns,), T.int32)
                        T.copy(ITable[bx + 2, :active_columns], leader_row)
                        describe_segments(
                            bx, table_row, peer_row, descriptors, shared_length, Runs, True, mapped, leader_row, True
                        )
                    else:
                        describe_segments(
                            bx, table_row, peer_row, descriptors, shared_length, Runs, True, mapped, table_row, False
                        )
                else:
                    describe_segments(
                        bx, table_row, peer_row, descriptors, shared_length, Runs, True, mapped, table_row, False
                    )
            else:
                describe_segments(bx, table_row, table_row, descriptors, 0, Runs, False, mapped, table_row, False)
        else:
            describe_segments(bx, table_row, table_row, descriptors, 0, Runs, False, 0, table_row, False)

    @T.prim_func
    def descriptors(
        ITable: T.Tensor((batch, table_columns), T.int32),
        Len: T.Tensor((length_offset + batch,), T.int32),
        Runs: T.Tensor((batch + vote_rows, descriptor_width), T.int32),
    ):
        with T.Kernel(batch) as bx:
            publish_descriptors(bx, ITable, Len, Runs)

    @T.prim_func
    def main(
        Value: T.Tensor((batch, 16384), T.bfloat16),
        Weight: T.Tensor((128,), T.float32),
        Bias: T.Tensor((128,), T.float32),
        Ape: T.Tensor((4, 128), T.float32),
        Len: T.Tensor((length_offset + batch,), T.int32),
        TailTable: T.StridedTensor((batch,), (tail_id_stride,), T.int32),
        Tail: T.StridedTensor((tail_count, 4, 256), (tail_stride, 256, 1), T.float32),
        ITable: T.Tensor((batch, table_columns), T.int32),
        K: T.Tensor((pages, 34, 128), T.float8_e4m3fn),
        KS: T.Tensor((pages, 1088), T.float32),
        Runs: T.Tensor((batch + vote_rows, descriptor_width), T.int32),
    ):
        partial = T.Tensor((batch, parts, 256), T.float32, Value.data)
        with T.Kernel(batch) as bx:
            if not split_descriptors:
                publish_descriptors(bx, ITable, Len, Runs)
            raw = T.alloc_shared((256,), T.float32)
            tile = T.alloc_shared((parts, 256), T.float32)
            T.copy(partial[bx, :, :], tile)
            with T.SimdVF():
                active = S.pset(32, "PAT_ALL")
                for v in range(4):
                    # Reduce partial projections in their original order before pooling the key.
                    if parts == 8:
                        acc0 = S.vld(tile[0, v * 64], "NORM")
                        acc1 = S.vadd(acc0, S.vld(tile[1, v * 64], "NORM"), active)
                        acc2 = S.vadd(acc1, S.vld(tile[2, v * 64], "NORM"), active)
                        acc3 = S.vadd(acc2, S.vld(tile[3, v * 64], "NORM"), active)
                        acc4 = S.vadd(acc3, S.vld(tile[4, v * 64], "NORM"), active)
                        acc5 = S.vadd(acc4, S.vld(tile[5, v * 64], "NORM"), active)
                        acc6 = S.vadd(acc5, S.vld(tile[6, v * 64], "NORM"), active)
                        acc7 = S.vadd(acc6, S.vld(tile[7, v * 64], "NORM"), active)
                        S.vsts(raw[v * 64], acc7, active, "NORM_B32")
                    else:
                        acc0 = S.vld(tile[0, v * 64], "NORM")
                        acc1 = S.vadd(acc0, S.vld(tile[1, v * 64], "NORM"), active)
                        acc2 = S.vadd(acc1, S.vld(tile[2, v * 64], "NORM"), active)
                        acc3 = S.vadd(acc2, S.vld(tile[3, v * 64], "NORM"), active)
                        acc4 = S.vadd(acc3, S.vld(tile[4, v * 64], "NORM"), active)
                        acc5 = S.vadd(acc4, S.vld(tile[5, v * 64], "NORM"), active)
                        acc6 = S.vadd(acc5, S.vld(tile[6, v * 64], "NORM"), active)
                        acc7 = S.vadd(acc6, S.vld(tile[7, v * 64], "NORM"), active)
                        acc8 = S.vadd(acc7, S.vld(tile[8, v * 64], "NORM"), active)
                        acc9 = S.vadd(acc8, S.vld(tile[9, v * 64], "NORM"), active)
                        acc10 = S.vadd(acc9, S.vld(tile[10, v * 64], "NORM"), active)
                        acc11 = S.vadd(acc10, S.vld(tile[11, v * 64], "NORM"), active)
                        acc12 = S.vadd(acc11, S.vld(tile[12, v * 64], "NORM"), active)
                        acc13 = S.vadd(acc12, S.vld(tile[13, v * 64], "NORM"), active)
                        acc14 = S.vadd(acc13, S.vld(tile[14, v * 64], "NORM"), active)
                        acc15 = S.vadd(acc14, S.vld(tile[15, v * 64], "NORM"), active)
                        acc16 = S.vadd(acc15, S.vld(tile[16, v * 64], "NORM"), active)
                        acc17 = S.vadd(acc16, S.vld(tile[17, v * 64], "NORM"), active)
                        acc18 = S.vadd(acc17, S.vld(tile[18, v * 64], "NORM"), active)
                        acc19 = S.vadd(acc18, S.vld(tile[19, v * 64], "NORM"), active)
                        acc20 = S.vadd(acc19, S.vld(tile[20, v * 64], "NORM"), active)
                        acc21 = S.vadd(acc20, S.vld(tile[21, v * 64], "NORM"), active)
                        acc22 = S.vadd(acc21, S.vld(tile[22, v * 64], "NORM"), active)
                        acc23 = S.vadd(acc22, S.vld(tile[23, v * 64], "NORM"), active)
                        acc24 = S.vadd(acc23, S.vld(tile[24, v * 64], "NORM"), active)
                        acc25 = S.vadd(acc24, S.vld(tile[25, v * 64], "NORM"), active)
                        acc26 = S.vadd(acc25, S.vld(tile[26, v * 64], "NORM"), active)
                        acc27 = S.vadd(acc26, S.vld(tile[27, v * 64], "NORM"), active)
                        S.vsts(raw[v * 64], acc27, active, "NORM_B32")
            length = Len[length_offset + bx]
            page = TailTable[bx]
            pos = length - 1
            if length > 0 and page >= 0:
                state = T.alloc_shared((256,), T.float32)
                gamma = T.alloc_shared((128,), T.float32)
                beta = T.alloc_shared((128,), T.float32)
                if precise:
                    T.copy(Weight[:], gamma)
                    T.copy(Bias[:], beta)
                    with T.SimdVF():
                        mask = S.pset(32, "PAT_ALL")
                        left = S.vcvt(
                            S.vcvt(S.vld(raw[0], "NORM"), "bfloat16", mask, round=raw_round, part=0),
                            "float32",
                            mask,
                            part=0,
                        )
                        right = S.vcvt(
                            S.vcvt(S.vld(raw[64], "NORM"), "bfloat16", mask, round=raw_round, part=0),
                            "float32",
                            mask,
                            part=0,
                        )
                        m = S.vdupv(
                            S.vcadd(
                                S.vadd(
                                    S.vmuls(left, T.float32(1.0 / 128), mask),
                                    S.vmuls(right, T.float32(1.0 / 128), mask),
                                    mask,
                                ),
                                mask,
                            ),
                            mask,
                        )
                        left = S.vsub(left, m, mask)
                        right = S.vsub(right, m, mask)
                        sq0 = S.vmuls(S.vmul(left, left, mask), T.float32(1.0 / 128), mask)
                        sq1 = S.vmuls(S.vmul(right, right, mask), T.float32(1.0 / 128), mask)
                        variance = S.vadds(S.vdupv(S.vcadd(S.vadd(sq0, sq1, mask), mask), mask), T.float32(eps), mask)
                        if mode in ("sqrt", "both", "both_fma"):
                            std = S.vsqrt(variance, mask, precision="ftz_true")
                        else:
                            std = S.vsqrt(variance, mask)
                        if mode in ("div", "both", "both_fma"):
                            z0 = S.vdiv(left, std, mask, precision="ftz_true")
                            z1 = S.vdiv(right, std, mask, precision="ftz_true")
                        else:
                            z0 = S.vdiv(left, std, mask)
                            z1 = S.vdiv(right, std, mask)
                        if mode in ("fma", "both_fma"):
                            v0 = S.alloc_var("float32")
                            v1 = S.alloc_var("float32")
                            v0 = S.vld(beta[0], "NORM")
                            v1 = S.vld(beta[64], "NORM")
                            S.vmula(v0, z0, S.vld(gamma[0], "NORM"), mask)
                            S.vmula(v1, z1, S.vld(gamma[64], "NORM"), mask)
                        else:
                            v0 = S.vadd(S.vmul(z0, S.vld(gamma[0], "NORM"), mask), S.vld(beta[0], "NORM"), mask)
                            v1 = S.vadd(S.vmul(z1, S.vld(gamma[64], "NORM"), mask), S.vld(beta[64], "NORM"), mask)
                        S.vsts(state[0], v0, mask, "NORM_B32")
                        S.vsts(state[64], v1, mask, "NORM_B32")
                        for j in range(2):
                            g = S.vcvt(S.vld(raw[128 + j * 64], "NORM"), "bfloat16", mask, round=raw_round, part=0)
                            S.vsts(state[128 + j * 64], S.vcvt(g, "float32", mask, part=0), mask, "NORM_B32")
                else:
                    with T.SimtVF(threads=128):
                        x = T.alloc_fragment((1, 128), T.float32)
                        squares = T.alloc_fragment((1, 128), T.float32)
                        mean = T.alloc_fragment((1,), T.float32)
                        delta = T.alloc_fragment((1,), T.float32)
                        for c in T.Parallel(128):
                            x[0, c] = T.cast(T.cast(raw[c], T.bfloat16), T.float32)
                        T.reduce_sum(x, mean, dim=1)
                        for c in T.Parallel(128):
                            diff = x[0, c] - mean[0] * (1.0 / 128)
                            squares[0, c] = diff * diff
                        T.reduce_sum(squares, delta, dim=1)
                        for c in T.Parallel(128):
                            state[c] = (x[0, c] - mean[0] * (1.0 / 128)) * T.rsqrt(
                                delta[0] * (1.0 / 128) + eps
                            ) * Weight[c] + Bias[c]
                            state[c + dim] = T.cast(T.cast(raw[c + dim], T.bfloat16), T.float32)
                if length % 4 == 0:
                    old = T.alloc_shared((3, 256), T.float32)
                    bf = T.alloc_shared((128,), T.bfloat16)
                    packed = T.alloc_shared((128,), T.float8_e4m3fn)
                    scale = T.alloc_shared((8,), T.float32)
                    for j in T.serial(3):
                        T.copy(Tail[page, (pos - 3 + j) % capacity, :], old[j, :])
                    if precise_pool:
                        ape = T.alloc_shared((4, 128), T.float32)
                        T.copy(Ape[:, :], ape)
                        with T.SimdVF():
                            mask = S.pset(32, "PAT_ALL")
                            for half in range(2):
                                start = half * 64
                                g0 = S.vadd(S.vld(old[0, 128 + start], "NORM"), S.vld(ape[0, start], "NORM"), mask)
                                g1 = S.vadd(S.vld(old[1, 128 + start], "NORM"), S.vld(ape[1, start], "NORM"), mask)
                                g2 = S.vadd(S.vld(old[2, 128 + start], "NORM"), S.vld(ape[2, start], "NORM"), mask)
                                g3 = S.vadd(S.vld(state[128 + start], "NORM"), S.vld(ape[3, start], "NORM"), mask)
                                mx = S.vmax(S.vmax(g0, g1, mask), S.vmax(g2, g3, mask), mask)
                                e0 = S.vexpdif(g0, mx, mask)
                                e1 = S.vexpdif(g1, mx, mask)
                                e2 = S.vexpdif(g2, mx, mask)
                                e3 = S.vexpdif(g3, mx, mask)
                                den = S.vadd(S.vadd(e0, e2, mask), S.vadd(e1, e3, mask), mask)
                                p0 = S.vcvt(
                                    S.vcvt(S.vdiv(e0, den, mask), "bfloat16", mask, round="ROUND_A", part=0),
                                    "float32",
                                    mask,
                                    part=0,
                                )
                                p1 = S.vcvt(
                                    S.vcvt(S.vdiv(e1, den, mask), "bfloat16", mask, round="ROUND_A", part=0),
                                    "float32",
                                    mask,
                                    part=0,
                                )
                                p2 = S.vcvt(
                                    S.vcvt(S.vdiv(e2, den, mask), "bfloat16", mask, round="ROUND_A", part=0),
                                    "float32",
                                    mask,
                                    part=0,
                                )
                                p3 = S.vcvt(
                                    S.vcvt(S.vdiv(e3, den, mask), "bfloat16", mask, round="ROUND_A", part=0),
                                    "float32",
                                    mask,
                                    part=0,
                                )
                                value = S.vmul(S.vld(old[0, start], "NORM"), p0, mask)
                                value = S.vadd(value, S.vmul(S.vld(old[1, start], "NORM"), p1, mask), mask)
                                value = S.vadd(value, S.vmul(S.vld(old[2, start], "NORM"), p2, mask), mask)
                                value = S.vadd(value, S.vmul(S.vld(state[start], "NORM"), p3, mask), mask)
                                S.vsts(
                                    bf[start], S.vcvt(value, "bfloat16", mask, round="ROUND_A", part=0), mask, "PK_B32"
                                )
                    else:
                        with T.SimtVF(threads=128):
                            for c in T.Parallel(dim):
                                m = T.alloc_var(T.float32, init=-1.0e30)
                                for j in T.unroll(3):
                                    m = T.max(m, old[j, dim + c] + Ape[j, c])
                                m = T.max(m, state[dim + c] + Ape[3, c])
                                num = T.alloc_var(T.float32, init=0.0)
                                den = T.alloc_var(T.float32, init=0.0)
                                for j in T.unroll(3):
                                    z = T.exp(old[j, dim + c] + Ape[j, c] - m)
                                    den += z
                                zlast = T.exp(state[dim + c] + Ape[3, c] - m)
                                den += zlast
                                for j in T.unroll(3):
                                    prob = T.cast(T.exp(old[j, dim + c] + Ape[j, c] - m) / den, T.bfloat16)
                                    num += old[j, c] * T.cast(prob, T.float32)
                                last_weight = T.cast(zlast / den, T.bfloat16)
                                bf[c] = T.cast(num + state[c] * T.cast(last_weight, T.float32), T.bfloat16)
                    if precise_quant:
                        with T.SimdVF():
                            mask = S.pset(32, "PAT_ALL")
                            scalar = S.pset(32, "PAT_VL1")
                            a = S.vcvt(S.vld(bf[0], "UNPK_B16"), "float32", mask, part=0)
                            b = S.vcvt(S.vld(bf[64], "UNPK_B16"), "float32", mask, part=0)
                            peak = S.vdupv(S.vcmax(S.vmax(S.vabs(a, mask), S.vabs(b, mask), mask), mask), mask)
                            unit = S.vdup(T.float32(1.0), "float32", mask)
                            if kind == "quant_div":
                                scaled = S.vdiv(peak, S.vdup(T.float32(448.0), "float32", mask), mask)
                            elif kind == "quant_ieee":
                                scaled = S.vdiv(
                                    peak, S.vdup(T.float32(448.0), "float32", mask), mask, precision="ftz_true"
                                )
                            else:
                                scaled = S.vmuls(peak, T.float32(1.0 / 448.0), mask)
                            sf = S.vsel(unit, scaled, S.vcmps(peak, T.float32(0.0), mask, op="eq"))
                            S.vsts(scale[0], sf, scalar, "ONEPT_B32")
                            S.vsts(
                                packed[0],
                                S.vcvt(S.vdiv(a, sf, mask, precision="ftz_true"), "float8_e4m3fn", mask),
                                mask,
                                "PK4_B32",
                            )
                            S.vsts(
                                packed[64],
                                S.vcvt(S.vdiv(b, sf, mask, precision="ftz_true"), "float8_e4m3fn", mask),
                                mask,
                                "PK4_B32",
                            )
                    else:
                        with T.SimtVF(threads=128):
                            x = T.alloc_fragment((1, 128), T.float32)
                            mx = T.alloc_fragment((1,), T.float32)
                            for c in T.Parallel(128):
                                x[0, c] = T.abs(T.cast(bf[c], T.float32))
                            T.reduce_max(x, mx, dim=1)
                            sf = T.if_then_else(mx[0] > 0.0, mx[0] / 448.0, 1.0)
                            for c in T.Parallel(128):
                                packed[c] = T.cast(T.cast(bf[c], T.float32) / sf, T.float8_e4m3fn)
                            for c in T.Parallel(1):
                                scale[c] = sf
                    slot = pos // 4
                    kp = ITable[bx, slot // pool_page_rows]
                    if pages * 4352 > 1 << 31:
                        kp = T.cast(kp, T.int64)
                    T.copy(packed, K[kp, slot % pool_page_rows, :])
                    with T.SimtVF(threads=32):
                        for c in T.Parallel(1):
                            KS[kp, 1024 + slot % pool_page_rows] = scale[c]
                if length % 4 != 0:
                    T.copy(state, Tail[page, pos % capacity, :])

    return (main, descriptors) if split_descriptors else main
