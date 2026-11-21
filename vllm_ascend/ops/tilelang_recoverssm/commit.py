# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 RecoverSSM commit for BF16 checkpoint and dense 128x128 heads."""

import tilelang
from tilelang import language as T


def build(
    layers,
    batch,
    heads,
    pages,
    *,
    steps=4,
    tile=64,
    align=True,
    index_stride=1,
    state_pad=0,
    replay_pad=0,
    packed=False,
    wide=True,
):
    dim = 128
    rec = 257
    parts = dim // tile
    jobs = layers * batch * heads * parts
    if tile not in (64, 128) or jobs % 2 or (packed and tile != 128):
        raise ValueError("incompatible vector partition")
    state_elements = heads * dim * dim if wide else pages * (heads * dim * dim + state_pad)
    replay_elements = heads * steps * rec if wide else pages * (heads * steps * rec + replay_pad)
    if max(state_elements, replay_elements) >= (1 << 31):
        raise ValueError("page span exceeds signed 32-bit element indexing")
    # TileLang's AscendC intrinsic rebinds the two typed GlobalTensors to the
    # layer selected by this AIV. No assumptions about adjacent allocations.
    lanes = min(48, jobs) if tile == 128 else jobs
    wave_offset = f" + wave*{lanes}" if lanes != jobs else ""
    bind = (
        "const int indirect_layer = ((AscendC::GetBlockIdx()/2*2 + "
        f"AscendC::GetSubBlockIdx(){wave_offset})/{heads * parts})%{layers};\n"
        "state.SetGlobalBuffer((__gm__ bfloat16_t*)table.GetValue(indirect_layer));\n"
        f"replay.SetGlobalBuffer((__gm__ float*)table.GetValue({2 * layers}+indirect_layer));"
    )
    if wide:
        bind = (
            "const int indirect_layer = ((AscendC::GetBlockIdx()/2*2 + "
            f"AscendC::GetSubBlockIdx(){wave_offset})/{heads * parts})%{layers};\n"
            "auto checkpoint_base = (__gm__ bfloat16_t*)table.GetValue(indirect_layer);\n"
            f"const int64_t checkpoint_stride = table.GetValue({layers}+indirect_layer);\n"
            f"auto replay_base = (__gm__ float*)table.GetValue({2 * layers}+indirect_layer);\n"
            f"const int64_t replay_stride = table.GetValue({3 * layers}+indirect_layer);\n"
            "state.SetGlobalBuffer(checkpoint_base + int64_t(s)*checkpoint_stride);\n"
            "replay.SetGlobalBuffer(replay_base + int64_t(s)*replay_stride);"
        )
    bind_boundary = "state.SetGlobalBuffer(checkpoint_base + int64_t(b)*checkpoint_stride);"
    bind_final = "state.SetGlobalBuffer(checkpoint_base + int64_t(f)*checkpoint_stride);"
    # One 64-lane repeat reads one half of a key row. A zero repeat stride
    # reuses the same key lanes for each of the 128 consecutive value rows.
    record_pitch = 264
    scratch = steps * record_pitch if packed else dim
    key_offset = f"step*{record_pitch}+128" if packed else "0"
    correction = f"key[step*{record_pitch}]" if packed else "cort[0]"
    wait_replay = (
        "AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(7);\nAscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(7);\n"
    )
    head_cpp = f"(AscendC::GetBlockIdx()/2*2+AscendC::GetSubBlockIdx(){wave_offset})%{heads}"
    replay_page_offset = "0" if wide else f"s * table.GetValue({3 * layers}+indirect_layer)"
    fetch_records = (
        f"const int64_t rbase = {replay_page_offset} + ({head_cpp})*{steps * rec};\n"
        f"AscendC::DataCopyPad(key, replay[rbase], AscendC::DataCopyExtParams(n < {steps} ? n : {steps}, "
        f"{rec * 4}, 0, 0, 0), AscendC::DataCopyPadExtParams<float>(true, 0, 7, 0.0f));\n"
        + wait_replay
        + "AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(7);\n"
        "AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(7);"
    )
    scale_columns = (
        ("" if packed else wait_replay)
        + f"AscendC::Brcb(key[{scratch}], {correction}, {tile // 8}, AscendC::BrcbRepeatParams(1,8));\n"
        "AscendC::PipeBarrier<PIPE_V>();\n"
        f"AscendC::Mul(outer[0], key[{scratch}], key[{key_offset}], uint64_t(64), {tile}, "
        "AscendC::BinaryRepeatParams(1, 0, 1, 16, 1, 0));\n"
        f"AscendC::Mul(outer[64], key[{scratch}], key[{key_offset}+64], uint64_t(64), {tile}, "
        "AscendC::BinaryRepeatParams(1, 0, 1, 16, 1, 0));\n"
        "AscendC::PipeBarrier<PIPE_V>();\n"
        "AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(7);\n"
        "AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(7);"
    )

    @T.prim_func
    def commit(
        state: T.Tensor((state_elements,), "bfloat16"),
        replay: T.Tensor((replay_elements,), "float32"),
        table: T.Tensor((4 * layers,), "int64"),
        source: T.Tensor((1 + (batch - 1) * index_stride,), "int32"),
        counts: T.Tensor((batch,), "int32"),
        final: T.Tensor((batch,), "int32"),
        boundary: T.Tensor((batch,), "int32"),
        prefix: T.Tensor((batch,), "int32"),
    ):
        with T.Kernel(lanes // 2, is_npu=True) as (core, sub):  # noqa: SIM117 - TileLang parses nested scopes
            with T.Scope("V"):
                for wave in T.serial((jobs + lanes - 1) // lanes):
                    job = core * 2 + sub + wave * lanes
                    row = T.min(job // (parts * heads * layers), batch - 1)
                    layer = job // (parts * heads) % layers
                    head = job // parts % heads
                    lo = job % parts * tile
                    n = counts[row]
                    s = source[row * index_stride]
                    f = final[row]
                    b = boundary[row]
                    length = prefix[row]
                    h = T.alloc_ub((tile, dim), "float32")
                    half = T.alloc_ub((tile * dim,), "bfloat16")
                    key = T.alloc_ub((1, scratch + (tile * 8 if tile == 128 else 0)), "float32")
                    cort = T.alloc_ub((tile, 1), "float32")
                    outer = T.alloc_ub((tile, dim), "float32")
                    keys = T.alloc_ub((tile, dim), "float32")
                    record_scale = T.alloc_ub((8,), "float32")
                    if (job < jobs) & (s > 0) & (f > 0) & (n > 0):
                        T._src_code(bind)
                        state_stride = 0 if wide else T.cast(table[layers + layer], "int32")
                        rep_stride = 0 if wide else T.cast(table[3 * layers + layer], "int32")
                        hv_offset = head * dim * dim + lo * dim
                        start = s * state_stride + hv_offset
                        T.copy(state[start : start + tile * dim], half)
                        T.tile.cast(h, half, "CAST_NONE", tile * dim)
                        if packed:
                            T._src_code(fetch_records)
                        need = (b > 0) & ((b != f) | (length != n))
                        if align:
                            if need & ((length <= 0) | (length > T.min(n, steps))):
                                boff = b * state_stride + hv_offset
                                if wide:
                                    T._src_code(bind_boundary)
                                T.copy(half, state[boff : boff + tile * dim])
                        for step in T.serial(T.min(n, steps)):
                            base = s * rep_stride + (head * steps + step) * rec
                            if packed:
                                off = step * record_pitch + 2 * dim
                                T.copy(key[0, off : off + 8], record_scale)
                                T.tile.mul(h, h, record_scale[0])
                            else:
                                T.copy(replay[base + dim : base + 2 * dim], key[0, :dim])
                                T.copy(replay[base + lo : base + lo + tile], cort[:, 0])
                                T.tile.mul(h, h, replay[base + 2 * dim])
                            if tile == 128:
                                T._src_code(scale_columns)
                            else:
                                T.tile.broadcast(outer, cort)
                                T.tile.broadcast(keys, key)
                                T.tile.mul(outer, outer, keys)
                            T.tile.add(h, h, outer)
                            if align:
                                if need & (step + 1 == length):
                                    boff = b * state_stride + hv_offset
                                    if wide:
                                        T._src_code(bind_boundary)
                                    T.tile.cast(half, h, "CAST_RINT", tile * dim)
                                    T.copy(half, state[boff : boff + tile * dim])
                        dst = f * state_stride + hv_offset
                        if wide:
                            T._src_code(bind_final)
                        T.tile.cast(half, h, "CAST_RINT", tile * dim)
                        T.copy(half, state[dst : dst + tile * dim])
                        if lanes != jobs:
                            T._src_code(
                                "AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(7); "
                                "AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(7);"
                            )

    return tilelang.compile(
        commit,
        target="ascendc",
        platform="A3",
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )
