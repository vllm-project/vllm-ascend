# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSES/CANN-Open-Software-License-Agreement-Version-2.0.txt.

"""A3 TileLang V0 for CANN v9.2.1 RoPE64 INT8 QSFA.

The gather follows the sparse-block-size-one, PA_BSND/TND, BF16 path in
``GetRealS2Idx``, ``CopyInKv``, ``CopyOutMrgeResult``, and ``MergeKv`` in the
release's ``arch22/kv_quant_sparse_flash_attention_service_vector_mla.h``
(commit 8fbf1ffea0cf0c401407fdf202743a75e0a427b7). Huawei Technologies Co., Ltd.
copyright 2025, distributed under CANN Open Software License Agreement 2.0.
"""

from textwrap import indent

_GATHER_TEMPLATE = r"""
{
    using namespace AscendC;
    constexpr int topk = 2048;
    constexpr int tokens_per_page = 128;
    constexpr int pages_per_request = 1024;
    constexpr int queries_per_request = 6;
    constexpr int tokens_per_tile = 512;
    constexpr int rows_per_fetch = 32;
    constexpr int quant_channels = 512;
    constexpr int rope_channels = 64;
    constexpr int packed_bytes = 656;
    constexpr int packed_stride = 672;
    constexpr int nz_lanes = 16;
    constexpr int metadata_words = 128;
    constexpr int input_ping_pong_bytes = 32768;
    constexpr int output_nope_event = 4;
@OUTPUT_ROPE_EVENT@
    const int core_id = qsfa_gather_core;
    const int vector_id = qsfa_gather_lane;
    const int query_row = qsfa_gather_row;
    const int tile_index = qsfa_gather_tile;
    const int merge_slot = qsfa_gather_slot;
    const int tile_tokens = qsfa_gather_tokens;
    const int request = query_row / queries_per_request;
    if (request != qsfa_gather_last_request) {
        qsfa_gather_causal_base = kv_lengths.GetValue(request) - query_ends.GetValue(request);
        qsfa_gather_last_request = request;
    }
    const int causal_limit = qsfa_gather_causal_base + query_row + 1;
    const int pair_count = (tile_tokens + 1) / 2;
    const int first_half = (pair_count + 1) / 2 * 2;
    const int start = vector_id == 0 ? 0 : first_half;
    const int stop = vector_id == 0 && first_half < tile_tokens ? first_half : tile_tokens;
    const int64_t topk_base = static_cast<int64_t>(query_row) * topk + tile_index * tokens_per_tile;
    const int64_t page_base = request * pages_per_request;
    const int64_t decoded_base = static_cast<int64_t>(core_id * 4 + merge_slot) * tokens_per_tile *
                                 (quant_channels + rope_channels);

    auto raw = vec_in1.ReinterpretCast<int8_t>();
    auto fp16 = vec_tmp1.ReinterpretCast<half>();
@LATENT_ND_BUFFER@@OUTPUT_BUFFERS@    auto scales = vec_tmp2;
    auto work = vec_in2;
    DataCopyPadExtParams<int8_t> pad{true, 0, 16, 0};
    const DataCopyExtParams one_row{1, packed_bytes, 0, 0, 0};
    SetFlag<HardEvent::MTE3_MTE2>(0);
    SetFlag<HardEvent::MTE3_MTE2>(1);

@OUTPUT_SETUP@    int fetch = 0;
    int loaded = 0;
    int written = 0;
    bool needs_input = true;
    for (int token = start; token < stop; token += 2) {
        const int buffer = fetch & 1;
        if (needs_input) {
            WaitFlag<HardEvent::MTE3_MTE2>(buffer);
            needs_input = false;
        }
        auto source = raw[buffer * input_ping_pong_bytes];
        const int first = indices.GetValue(topk_base + token);
        if (first >= 0) {
            const int second = token + 1 < stop ? indices.GetValue(topk_base + token + 1) : -1;
            int64_t first_row = -1;
            int64_t second_row = -1;
            if (first < causal_limit) {
                const int page = page_table.GetValue(page_base + first / tokens_per_page);
                first_row = static_cast<int64_t>(page) * tokens_per_page + first % tokens_per_page;
            }
            if (second >= 0 && second < causal_limit) {
                const int page = page_table.GetValue(page_base + second / tokens_per_page);
                second_row = static_cast<int64_t>(page) * tokens_per_page + second % tokens_per_page;
            }
            if (first_row >= 0 || second_row >= 0) {
                const int64_t low = first_row < second_row ? first_row : second_row;
                const int64_t high = first_row < second_row ? second_row : first_row;
                const int64_t stride = (high - low - 1) * packed_bytes;
                if (low >= 0 && stride >= 0 && stride < 2147483647LL &&
                    first + 1 < causal_limit && second + 1 < causal_limit) {
                    const DataCopyExtParams pair{2, packed_bytes, static_cast<uint32_t>(stride), 0, 0};
                    DataCopyPad(source[(loaded - written) * packed_stride], packed[low * packed_bytes], pair, pad);
                    loaded += 2;
                } else {
                    if (first_row >= 0) {
                        DataCopyPad(source[(loaded - written) * packed_stride],
                                    packed[first_row * packed_bytes], one_row, pad);
                        ++loaded;
                    }
                    if (second_row >= 0) {
                        DataCopyPad(source[(loaded - written) * packed_stride],
                                    packed[second_row * packed_bytes], one_row, pad);
                        ++loaded;
                    }
                }
            }
        }
        const int rows = loaded - written;
        if (first < 0 || rows + 2 > rows_per_fetch || token + 2 >= stop) {
            if (rows > 0) {
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                const auto no_round = RoundMode::CAST_NONE;
                if (rows == 1) {
                    Cast(fp16, source, no_round, static_cast<uint64_t>(128), 4, {1, 1, 8, 4});
                } else {
                    for (int channel = 0; channel < quant_channels; channel += 128) {
                        Cast(fp16[channel], source[channel], no_round, static_cast<uint64_t>(128),
                             static_cast<uint8_t>(rows), {1, 1, 32, 21});
                    }
                }
                PipeBarrier<PIPE_V>();
                auto scale_input = source[640].ReinterpretCast<float>();
                if (rows == 1) {
                    Brcb(scales, scale_input, 1, {1, 4});
                } else {
                    auto raw_scales = scales[1024];
                    const DataCopyParams scale_copy{static_cast<uint16_t>(rows), 1, 20, 0};
                    DataCopy(raw_scales, scale_input, scale_copy);
                    PipeBarrier<PIPE_V>();
                    Brcb(scales, raw_scales, rows, {1, 4});
                }
                PipeBarrier<PIPE_V>();
@DEQUANTIZE_LATENT@@OUTPUT@
                written = loaded;
            }
            SetFlag<HardEvent::MTE3_MTE2>(buffer);
            ++fetch;
            needs_input = true;
            if (first < 0) {
                break;
            }
        }
    }
    if (start + loaded < stop) {
        const int buffer = fetch & 1;
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(buffer);
        auto zero = raw.ReinterpretCast<bfloat16_t>();
        Duplicate(zero, static_cast<bfloat16_t>(0.0f), quant_channels);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
@ZERO_FILL@
        SetFlag<HardEvent::MTE3_MTE2>(buffer);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(1);

@OUTPUT_DRAIN@    gather_valid.SetValue(merge_slot, loaded);
    SetFlag<HardEvent::S_MTE3>(1);
    WaitFlag<HardEvent::S_MTE3>(1);
    const DataCopyExtParams valid_copy{1, metadata_words * sizeof(int32_t), 0, 0, 0};
    const int64_t valid_base = (static_cast<int64_t>(core_id * 4 + merge_slot) * 2 + vector_id) * metadata_words;
    DataCopyPad(valid_gm[valid_base], gather_valid, valid_copy);
}
"""


_OUTPUT_BUFFERS = """\
    auto nope_nz = vec_out1.ReinterpretCast<bfloat16_t>();
    auto rope_nz = vec_out2.ReinterpretCast<bfloat16_t>();
"""

_OUTPUT_SINGLE = """

@PACK_LATENT@
                SetFlag<HardEvent::V_MTE3>(output_nope_event);
                WaitFlag<HardEvent::V_MTE3>(output_nope_event);
@STORE_LATENT@
                SetFlag<HardEvent::MTE3_V>(output_nope_event);

                WaitFlag<HardEvent::MTE3_V>(output_rope_event);
                Copy(rope_nz, source[quant_channels].ReinterpretCast<bfloat16_t>(),
                     rope_channels, static_cast<uint8_t>(rows), {static_cast<uint16_t>(@C0_PITCH@), 1, 1, 21});
                SetFlag<HardEvent::V_MTE3>(output_rope_event);
                WaitFlag<HardEvent::V_MTE3>(output_rope_event);
@STORE_ROPE@
                SetFlag<HardEvent::MTE3_V>(output_rope_event);"""

_OUTPUT_FUSED_BUFFERS = """\
    auto fused_output = vec_out1.ReinterpretCast<bfloat16_t>();
"""

_OUTPUT_PINGPONG_DRAIN = """\
    // Softmax reuses both halves of vec_out1 under the event for bank zero.
    // Retire all Gather stores without consuming either bank's credit.
    if (loaded != 0) {
        SetFlag<HardEvent::MTE3_V>(1);
        WaitFlag<HardEvent::MTE3_V>(1);
    }

"""


_STORE_LATENT = """\
                const int64_t block_base = decoded_base + (start + written) * nz_lanes;
                DataCopyExtParams nz_copy{32, static_cast<uint32_t>(rows * 32), 0,
                                          static_cast<uint32_t>((tokens_per_tile - rows) * 32), 0};
                DataCopyPad(kv_nz[block_base], nope_nz, nz_copy);"""

_STORE_ROPE = """\
                nz_copy.blockCount = rope_channels / nz_lanes;
                DataCopyPad(kv_nz[block_base + tokens_per_tile * quant_channels], rope_nz, nz_copy);"""

_ZERO_FILL = """\
        DataCopyExtParams zero_copy{32, 32, 0, (tokens_per_tile - 1) * 32, 0};
        for (int token = start + loaded; token < stop; ++token) {
            DataCopyPad(kv_nz[decoded_base + token * nz_lanes], zero, zero_copy);
        }
        zero_copy.blockCount = rope_channels / nz_lanes;
        for (int token = start + loaded; token < stop; ++token) {
            DataCopyPad(kv_nz[decoded_base + tokens_per_tile * quant_channels + token * nz_lanes], zero, zero_copy);
        }"""

_BLOCKED_STORE_LATENT = """\
                constexpr int panel_tokens = 128;
                constexpr int panel_elements = panel_tokens * (quant_channels + rope_channels);
                const int token = @STORE_TOKEN@;
                const int panel_row = token % panel_tokens;
                const int available = panel_tokens - panel_row;
                const int first_rows = @STORE_ROWS@ < available ? @STORE_ROWS@ : available;
                const int next_rows = @STORE_ROWS@ - first_rows;
                const int64_t panel_base = decoded_base + (token / panel_tokens) * panel_elements;
                const int64_t first_base = panel_base + panel_row * nz_lanes;
                const int64_t next_base = panel_base + panel_elements;
                // UB groups retain their original `@STORE_PITCH@` pitch when the GM copy splits.
                DataCopyExtParams first_copy{@STORE_GROUPS@, static_cast<uint32_t>(first_rows * 32),
                    static_cast<uint32_t>(@FIRST_SKIP@), static_cast<uint32_t>((panel_tokens - first_rows) * 32), 0};
                DataCopyExtParams next_copy{@STORE_GROUPS@, static_cast<uint32_t>(next_rows * 32),
                    static_cast<uint32_t>(@NEXT_SKIP@), static_cast<uint32_t>((panel_tokens - next_rows) * 32), 0};
                DataCopyPad(kv_nz[first_base], @STORE_BUFFER@, first_copy);
                if (next_rows) {
                    DataCopyPad(kv_nz[next_base], @STORE_BUFFER@[first_rows * nz_lanes], next_copy);
                }"""

_BLOCKED_STORE_ROPE = """\
                constexpr int rope_offset = panel_tokens * quant_channels;
                first_copy.blockCount = rope_channels / nz_lanes;
                DataCopyPad(kv_nz[first_base + rope_offset], rope_nz, first_copy);
                if (next_rows) {
                    next_copy.blockCount = rope_channels / nz_lanes;
                    DataCopyPad(kv_nz[next_base + rope_offset], rope_nz[first_rows * nz_lanes], next_copy);
                }"""

_BLOCKED_ZERO_FILL = """\
        constexpr int panel_tokens = 128;
        constexpr int panel_elements = panel_tokens * (quant_channels + rope_channels);
        const DataCopyExtParams latent_zero{32, 32, 0, (panel_tokens - 1) * 32, 0};
        const DataCopyExtParams rope_zero{rope_channels / nz_lanes, 32, 0, (panel_tokens - 1) * 32, 0};
        for (int token = start + loaded; token < stop; ++token) {
            const int64_t base = decoded_base + (token / panel_tokens) * panel_elements +
                (token % panel_tokens) * nz_lanes;
            DataCopyPad(kv_nz[base], zero, latent_zero);
            DataCopyPad(kv_nz[base + panel_tokens * quant_channels], zero, rope_zero);
        }"""


_DEQUANTIZE_LATENT = """\
                for (int row_start = 0; row_start < rows; row_start += 16) {
                    const int chunk_rows = rows - row_start < 16 ? rows - row_start : 16;
                    const uint32_t elements = chunk_rows * quant_channels;
                    Cast(work, fp16[row_start * quant_channels], no_round, elements);
                    PipeBarrier<PIPE_V>();
                    for (int sub = 0; sub < 128; sub += 64) {
                        Mul(work[sub], work[sub], scales[row_start * 32], 64, 4 * chunk_rows,
                            {1, 1, 0, 16, 16, 1});
                    }
                    PipeBarrier<PIPE_V>();
@CAST_BF16@
                    PipeBarrier<PIPE_V>();
                }"""

_CAST_BF16 = """\
                    Cast(nd_bf16[row_start * quant_channels], work, RoundMode::CAST_RINT, elements);"""

_DIRECT_NZ_CAST = """\
                    // Each repeat consumes one row. Its four BF16 blocks go to four C0 groups.
                    const UnaryRepeatParams nz_strides{static_cast<uint16_t>(@C0_PITCH@), 1, 1,
                                                       quant_channels * sizeof(float) / 32};
                    #pragma unroll
                    for (int channel = 0; channel < quant_channels; channel += 64) {
                        const int offset = (channel / nz_lanes * c0_pitch) * nz_lanes;
                        Cast(nope_nz[offset], work[channel], RoundMode::CAST_RINT, static_cast<uint64_t>(64),
                             static_cast<uint8_t>(chunk_rows), nz_strides);
                    }"""

_FUSED_OUTPUT_BANKS = """\
                    // The RoPE C0 groups follow the NoPE groups in both UB and GM.
                    auto bank = fused_output[output_bank * c0_pitch * (quant_channels + rope_channels)];
                    auto nope_nz = bank;
                    auto rope_nz = bank[c0_pitch * quant_channels];"""

_OUTPUT_PINGPONG = """\
                    // A single credit guards both NoPE and RoPE for this bank.
                    const int output_event = output_bank == 0 ? output_nope_event : output_rope_event;
@OUTPUT_BANKS@
                    WaitFlag<HardEvent::MTE3_V>(output_event);
@DIRECT_NZ_CAST@
                    Copy(rope_nz, source[row_start * packed_stride + quant_channels].ReinterpretCast<bfloat16_t>(),
                         rope_channels, static_cast<uint8_t>(chunk_rows), {c0_pitch, 1, 1, 21});
                    SetFlag<HardEvent::V_MTE3>(output_event);
                    WaitFlag<HardEvent::V_MTE3>(output_event);
@STORE_LATENT@
@STORE_ROPE@
                    SetFlag<HardEvent::MTE3_V>(output_event);
                    output_bank ^= 1;"""


_PACK_LATENT = """\
                WaitFlag<HardEvent::MTE3_V>(output_nope_event);
                for (int row_start = 0; row_start < rows; row_start += 8) {
                    const int chunk_rows = rows - row_start < 8 ? rows - row_start : 8;
                    Copy(nope_nz[row_start * nz_lanes], nd_bf16[row_start * quant_channels],
                         static_cast<uint64_t>(chunk_rows * nz_lanes), 32,
                         {1, 32, static_cast<uint16_t>(rows), 1});
                }"""


def _with_kv_stores(latent, rope, zero, *, output_fused_dma=False):
    if output_fused_dma:
        # Each bank holds 16 rows of NoPE and RoPE side by side in NZ layout.
        output_buffers = _OUTPUT_FUSED_BUFFERS
        output_setup = "    constexpr int c0_pitch = 16;\n    int output_bank = 0;\n"
        output_drain = _OUTPUT_PINGPONG_DRAIN
        output = ""
        cast_bf16 = (
            _OUTPUT_PINGPONG.replace("@OUTPUT_BANKS@", _FUSED_OUTPUT_BANKS)
            .replace("@DIRECT_NZ_CAST@", _DIRECT_NZ_CAST)
            .replace("@STORE_LATENT@", indent(latent, " " * 4))
            .replace("@STORE_ROPE@", "")
        )
        pitch = "c0_pitch"
        nd_buffer = ""
        store_token = "start + written + row_start"
        store_rows = "chunk_rows"
        first_skip, next_skip = "c0_pitch - first_rows", "c0_pitch - next_rows"
        store_groups, store_buffer = "(quant_channels + rope_channels) / nz_lanes", "bank"
    else:
        output_buffers = _OUTPUT_BUFFERS
        output_setup = output_drain = ""
        output = _OUTPUT_SINGLE.replace("@PACK_LATENT@", _PACK_LATENT)
        cast_bf16 = _CAST_BF16
        pitch = "rows"
        nd_buffer = "    auto nd_bf16 = vec_tmp1.ReinterpretCast<bfloat16_t>();\n"
        store_token, store_rows = "start + written", "rows"
        first_skip, next_skip = "next_rows", "first_rows"
        store_groups, store_buffer = "32", "nope_nz"
    return (
        _GATHER_TEMPLATE.replace("@OUTPUT_BUFFERS@", output_buffers)
        .replace("@OUTPUT_ROPE_EVENT@", "    constexpr int output_rope_event = 5;\n")
        .replace("@OUTPUT_SETUP@", output_setup)
        .replace("@OUTPUT_DRAIN@", output_drain)
        .replace("@OUTPUT@", output)
        .replace("@DEQUANTIZE_LATENT@", _DEQUANTIZE_LATENT.replace("@CAST_BF16@", cast_bf16))
        .replace("@STORE_LATENT@", latent)
        .replace("@STORE_ROPE@", rope)
        .replace("@ZERO_FILL@", zero)
        .replace("@LATENT_ND_BUFFER@", nd_buffer)
        .replace("@C0_PITCH@", pitch)
        .replace("@STORE_TOKEN@", store_token)
        .replace("@STORE_ROWS@", store_rows)
        .replace("@STORE_PITCH@", pitch)
        .replace("@FIRST_SKIP@", first_skip)
        .replace("@NEXT_SKIP@", next_skip)
        .replace("@STORE_GROUPS@", store_groups)
        .replace("@STORE_BUFFER@", store_buffer)
    )


_GATHER = _with_kv_stores(_STORE_LATENT, _STORE_ROPE, _ZERO_FILL)
_GATHER_BLOCKED_KV = _with_kv_stores(_BLOCKED_STORE_LATENT, _BLOCKED_STORE_ROPE, _BLOCKED_ZERO_FILL)
_GATHER_BLOCKED_OUTPUT_FUSED_DMA = _with_kv_stores(
    _BLOCKED_STORE_LATENT,
    _BLOCKED_STORE_ROPE,
    _BLOCKED_ZERO_FILL,
    output_fused_dma=True,
)


_STAGE_METADATA = """\
    // Gather/softmax valid counts use only words [0, 384) of this UB buffer.
    // Keep the request's page table until the AIV advances to another request.
    auto staged_pages = gather_valid[512];
    auto staged_indices = gather_valid[1536];
    constexpr int words_per_block = 32 / sizeof(int32_t);
    constexpr int index_capacity = tokens_per_tile / 2;
    const int index_count = stop - start;
    const bool use_bulk_indices = index_count > 0 && index_count <= index_capacity &&
                                  start % words_per_block == 0 && index_count % words_per_block == 0;
    if (request_changed) {
        DataCopy(staged_pages, page_table[page_base], pages_per_request);
    }
    if (use_bulk_indices) {
        DataCopy(staged_indices, indices[topk_base + start], index_count);
    }
    if (request_changed || use_bulk_indices) {
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
    }

"""


def _with_bulk_metadata(source):
    return (
        source.replace(
            "    if (request != qsfa_gather_last_request) {",
            "    const bool request_changed = request != qsfa_gather_last_request;\n    if (request_changed) {",
        )
        .replace("    auto raw =", _STAGE_METADATA + "    auto raw =")
        .replace(
            "indices.GetValue(topk_base + token);",
            "use_bulk_indices ? staged_indices.GetValue(token - start) : indices.GetValue(topk_base + token);",
        )
        .replace(
            "const int second = token + 1 < stop ? indices.GetValue(topk_base + token + 1) : -1;",
            "const int second = token + 1 >= stop ? -1 :\n"
            "                use_bulk_indices ? staged_indices.GetValue(token + 1 - start) "
            ": indices.GetValue(topk_base + token + 1);",
        )
        .replace(
            "page_table.GetValue(page_base + first / tokens_per_page)", "staged_pages.GetValue(first / tokens_per_page)"
        )
        .replace(
            "page_table.GetValue(page_base + second / tokens_per_page)",
            "staged_pages.GetValue(second / tokens_per_page)",
        )
    )


def make_gather_stage(T, *, bulk_metadata=False, blocked_kv=False, output_fused_dma=False):
    """Create the V0 macro for use within the caller's vector scope.

    Caller buffers use the macro parameter names, with ``kv_nz`` as the
    ``decoded`` argument. The caller owns the MTE3-to-V credits and publishes
    completion after transferring KV and ``valid_gm[core, slot, vector, :]``.
    Each metadata block stores the valid count at the word for its merge slot.
    Fused output DMA keeps NoPE and RoPE in ``vec_out1`` and omits ``vec_out2``.
    """
    gather_source = _GATHER_BLOCKED_KV if blocked_kv else _GATHER
    if output_fused_dma:
        gather_source = _GATHER_BLOCKED_OUTPUT_FUSED_DMA
    if bulk_metadata:
        gather_source = _with_bulk_metadata(gather_source)

    @T.macro
    def gather(
        packed,
        indices,
        page_table,
        query_ends,
        kv_lengths,
        decoded,
        core,
        vec,
        row,
        tile,
        kv_slot,
        valid_tokens,
        vec_in1,
        vec_in2,
        vec_out1,
        vec_out2,
        vec_tmp1,
        vec_tmp2,
        gather_valid,
        valid_gm,
        *,
        source=gather_source,
    ):
        qsfa_gather_core = T.alloc_var("int32", core)
        qsfa_gather_lane = T.alloc_var("int32", vec)
        qsfa_gather_row = T.alloc_var("int32", row)
        qsfa_gather_tile = T.alloc_var("int32", tile)
        qsfa_gather_slot = T.alloc_var("int32", kv_slot)
        qsfa_gather_tokens = T.alloc_var("int32", valid_tokens)
        if output_fused_dma:
            T._src_code(
                source,
                packed.access_ptr("r"),
                indices.access_ptr("r"),
                page_table.access_ptr("r"),
                query_ends.access_ptr("r"),
                kv_lengths.access_ptr("r"),
                decoded.access_ptr("w"),
                valid_gm.access_ptr("w"),
                vec_in1.access_ptr("rw"),
                vec_in2.access_ptr("rw"),
                vec_out1.access_ptr("rw"),
                vec_tmp1.access_ptr("rw"),
                vec_tmp2.access_ptr("rw"),
                gather_valid.access_ptr("rw"),
                qsfa_gather_core,
                qsfa_gather_lane,
                qsfa_gather_row,
                qsfa_gather_tile,
                qsfa_gather_slot,
                qsfa_gather_tokens,
            )
        else:
            T._src_code(
                source,
                packed.access_ptr("r"),
                indices.access_ptr("r"),
                page_table.access_ptr("r"),
                query_ends.access_ptr("r"),
                kv_lengths.access_ptr("r"),
                decoded.access_ptr("w"),
                valid_gm.access_ptr("w"),
                vec_in1.access_ptr("rw"),
                vec_in2.access_ptr("rw"),
                vec_out1.access_ptr("rw"),
                vec_out2.access_ptr("rw"),
                vec_tmp1.access_ptr("rw"),
                vec_tmp2.access_ptr("rw"),
                gather_valid.access_ptr("rw"),
                qsfa_gather_core,
                qsfa_gather_lane,
                qsfa_gather_row,
                qsfa_gather_tile,
                qsfa_gather_slot,
                qsfa_gather_tokens,
            )

    return gather
