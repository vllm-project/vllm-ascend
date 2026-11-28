# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSES/CANN-Open-Software-License-Agreement-Version-2.0.txt.

"""A3 QSFA Cube stages for the native TND V-template with 64 RoPE channels.

The bank layout and pipeline follow ComputeMm1 and ComputeMm2 in CANN 9.2.1's
arch22/kv_quant_sparse_flash_attention_service_cube_mla.h (ops-transformer
8fbf1ffea0cf0c401407fdf202743a75e0a427b7), distributed under the CANN Open
Software License Agreement, Version 2.0. TileLang owns the cross-core flags.
"""

from textwrap import indent

_INITIALIZE = """
constexpr uint32_t qsfa_heads = 64;
constexpr uint32_t qsfa_s2 = 512;
constexpr uint32_t qsfa_combined = 576;
constexpr uint32_t qsfa_c0 = 16;
constexpr uint32_t qsfa_n_panel = 128;
constexpr uint32_t qsfa_bank_l1 = qsfa_heads * qsfa_combined;
constexpr uint32_t qsfa_bank_ab = 32 * 1024 / sizeof(bfloat16_t);
constexpr uint32_t qsfa_bank_c = 64 * 1024 / sizeof(float);
constexpr uint32_t qsfa_ab_event_base = 3;
constexpr uint32_t qsfa_qp_events[4] = {2, 3, 4, 5};
constexpr uint32_t qsfa_kv_events[3] = {6, 7, 1};
uint32_t qsfa_qp_cursor = 0;
uint32_t qsfa_kv_cursor = 0;
uint32_t qsfa_ab_cursor = 0;
uint32_t qsfa_c_cursor = 0;
auto qsfa_l1 = ascend_l1.Get<bfloat16_t>();
auto qsfa_a = ascend_l0a.Get<bfloat16_t>();
auto qsfa_b = ascend_l0b.Get<bfloat16_t>();
auto qsfa_c = ascend_l0c.Get<float>();
AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(qsfa_ab_event_base);
AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(qsfa_ab_event_base + 1);
for (uint32_t qsfa_i = 0; qsfa_i < 4; ++qsfa_i) {
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(qsfa_qp_events[qsfa_i]);
}
for (uint32_t qsfa_i = 0; qsfa_i < 3; ++qsfa_i) {
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(qsfa_kv_events[qsfa_i]);
}
"""

_FINALIZE = """
AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(qsfa_ab_event_base);
AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(qsfa_ab_event_base + 1);
for (uint32_t qsfa_i = 0; qsfa_i < 4; ++qsfa_i) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(qsfa_qp_events[qsfa_i]);
}
for (uint32_t qsfa_i = 0; qsfa_i < 3; ++qsfa_i) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(qsfa_kv_events[qsfa_i]);
}
"""

_RESIDENT_QUERY_INIT = """
// Bank 2 is reserved for Q. Its former 32 KiB P slice uses bank 3's upper half.
constexpr uint32_t qsfa_query_l1_bank = 2;
constexpr uint32_t qsfa_query_l1_event = 0;
AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(qsfa_query_l1_event);
"""

_RESIDENT_QUERY_DRAIN = """
AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(qsfa_query_l1_event);
"""


def _load3d(dest, source, rows, columns, *, transpose):
    return f"""
    {{
        AscendC::LoadData3DParamsV2<bfloat16_t> load{{}};
        load.l1H = ({rows}) / qsfa_c0;
        load.l1W = qsfa_c0;
        load.padList[3] = 255;
        load.mExtension = {rows};
        load.kExtension = {columns};
        load.strideH = load.strideW = 1;
        load.filterH = load.filterW = 1;
        load.dilationFilterH = load.dilationFilterW = 1;
        load.enTranspose = {int(transpose)};
        load.channelSize = {columns};
        // AscendC's default config resets both the fmatrix and padding.
        AscendC::LoadData({dest}, {source}, load);
    }}
"""


def _matmul(n, k, initialize, finish):
    return f"""
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ab_event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ab_event);
        AscendC::MmadParams mad{{}};
        mad.m = qsfa_heads;
        mad.n = {n};
        mad.k = {k};
        mad.cmatrixInitVal = {initialize};
        mad.unitFlag = ({finish}) ? 0b11 : 0b10;
        AscendC::Mmad(accumulator, a_l0, b_l0, mad);
        if ((qsfa_heads / 16) * (mad.n / 16) < 10) {{
            AscendC::PipeBarrier<PIPE_M>();
        }}
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ab_event);
"""


def _fixpipe(destination, n):
    return f"""
        AscendC::FixpipeParamsV220 fix{{}};
        fix.mSize = qsfa_heads;
        fix.nSize = {n};
        fix.srcStride = qsfa_heads;
        fix.dstStride = qsfa_s2;
        fix.ndNum = 1;
        fix.unitFlag = 0b11;
        AscendC::Fixpipe({destination}, accumulator, fix);
"""


_AB_WAIT = """
        const uint32_t ab_bank = qsfa_ab_cursor++ % 2;
        const uint32_t ab_event = qsfa_ab_event_base + ab_bank;
        auto a_l0 = qsfa_a[ab_bank * qsfa_bank_ab];
        auto b_l0 = qsfa_b[ab_bank * qsfa_bank_ab];
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ab_event);
"""


_Q_LOAD = """
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(qp_event);
    AscendC::Nd2NzParams q_copy{};
    q_copy.ndNum = 1;
    q_copy.nValue = qsfa_heads;
    q_copy.dValue = q_copy.srcDValue = qsfa_combined;
    q_copy.dstNzC0Stride = qsfa_heads;
    q_copy.dstNzNStride = 1;
    AscendC::DataCopy(q_l1, query[qsfa_qk_query_offset], q_copy);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(qp_event);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(qp_event);
""".strip("\n")

_Q_PANEL_LOAD = """
    const uint32_t qp_bank = 2 * (qsfa_qp_cursor++ % 2);
    const uint32_t qp_event = qsfa_qp_events[qp_bank];
    auto q_l1 = qsfa_l1[qp_bank * qsfa_bank_l1];
@Q_LOAD@
""".strip("\n").replace("@Q_LOAD@", _Q_LOAD)

_Q_PANEL_RELEASE = "    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(qp_event);"

_Q_RESIDENT_LOAD = """
    // PV retains the original ping-pong cursor progression.
    ++qsfa_qp_cursor;
    const uint32_t qp_event = qsfa_query_l1_event;
    auto q_l1 = qsfa_l1[qsfa_query_l1_bank * qsfa_bank_l1];
    if (qsfa_qk_panel == 0) {
@Q_LOAD@
    }
""".strip("\n").replace("@Q_LOAD@", indent(_Q_LOAD, "    "))

_Q_RESIDENT_RELEASE = """
    if (qsfa_qk_panel == 3) {
@Q_RELEASE@
    }
""".strip("\n").replace("@Q_RELEASE@", indent(_Q_PANEL_RELEASE, "    "))

_QK_TEMPLATE = (
    """
{
    constexpr uint32_t qk_l1_k = 288;
    constexpr uint32_t qk_l0_k = 96;
@LOAD_QUERY_PANEL@
    for (uint32_t n0 = 0; n0 < qsfa_qk_tokens; n0 += qsfa_n_panel) {
        const uint32_t remaining = qsfa_qk_tokens - n0;
        const uint32_t n = remaining < qsfa_n_panel ? remaining : qsfa_n_panel;
        const uint32_t n_aligned = (n + 15) / 16 * 16;
        auto accumulator = qsfa_c[(qsfa_c_cursor++ % 2) * qsfa_bank_c];
        for (uint32_t k1 = 0; k1 < 2; ++k1) {
            const uint32_t kv_bank = qsfa_kv_cursor++ % 3;
            const uint32_t kv_event = qsfa_kv_events[kv_bank];
            auto k_l1 = qsfa_l1[(4 + kv_bank) * qsfa_bank_l1];
@LOAD_KV_PANEL@
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kv_event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kv_event);
            for (uint32_t k0 = 0; k0 < qk_l1_k; k0 += qk_l0_k) {
                @AB_WAIT@
                @LOAD_Q@
                AscendC::LoadData2DParams load_k{};
                load_k.repeatTimes = (n_aligned / qsfa_c0) * (qk_l0_k / qsfa_c0);
                load_k.srcStride = 1;
                AscendC::LoadData(b_l0, k_l1[k0 * n_aligned], load_k);
                @MATMUL@
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kv_event);
        }
        @FIXPIPE@
    }
@RELEASE_QUERY_PANEL@
}
""".replace("@AB_WAIT@", _AB_WAIT)
    .replace(
        "@LOAD_Q@",
        _load3d("a_l0", "q_l1[(k1 * qk_l1_k + k0) * qsfa_heads]", "qsfa_heads", "qk_l0_k", transpose=False),
    )
    .replace(
        "@MATMUL@",
        _matmul("n_aligned", "qk_l0_k", "k1 == 0 && k0 == 0", "k1 == 1 && k0 + qk_l0_k == qk_l1_k"),
    )
    .replace(
        "@FIXPIPE@",
        _fixpipe("scores[qsfa_qk_result_offset + n0]", "n_aligned"),
    )
)


_QK_KV_LOAD = """\
            const uint64_t k_offset = qsfa_qk_kv_offset + k1 * qk_l1_k * qsfa_s2 + n0 * qsfa_c0;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kv_event);
            AscendC::DataCopyParams k_copy{};
            k_copy.blockLen = n;
            k_copy.srcStride = qsfa_s2 - n;
            k_copy.dstStride = n_aligned - n;
            // The native second L1 panel has a latent tail and a RoPE segment.
            k_copy.blockCount = (k1 == 0 ? qk_l1_k : 224) / qsfa_c0;
            AscendC::DataCopy(k_l1, kv_nz[k_offset], k_copy);
            if (k1 == 1) {
                k_copy.blockCount = 64 / qsfa_c0;
                AscendC::DataCopy(k_l1[224 * n_aligned], kv_nz[k_offset + 224 * qsfa_s2], k_copy);
            }"""

_QK_BLOCKED_KV_LOAD = (
    _QK_KV_LOAD.replace(
        "k1 * qk_l1_k * qsfa_s2 + n0 * qsfa_c0",
        "n0 * qsfa_combined + k1 * qk_l1_k * qsfa_n_panel",
    )
    .replace("qsfa_s2 - n", "qsfa_n_panel - n")
    .replace("224 * qsfa_s2", "224 * qsfa_n_panel")
)

_QK = _QK_TEMPLATE.replace("@LOAD_KV_PANEL@", _QK_KV_LOAD)
_QK_BLOCKED_KV = _QK_TEMPLATE.replace("@LOAD_KV_PANEL@", _QK_BLOCKED_KV_LOAD)


_PV_TEMPLATE = (
    """
{
    constexpr uint32_t pv_l1_k = 256;
    constexpr uint32_t pv_l0_k = 128;
    const uint32_t qp_base = 2 * (qsfa_qp_cursor++ % 2);
    for (uint32_t n0 = 0; n0 < qsfa_s2; n0 += qsfa_n_panel) {
        auto accumulator = qsfa_c[(qsfa_c_cursor++ % 2) * qsfa_bank_c];
        for (uint32_t k1 = 0; k1 < qsfa_pv_tokens; k1 += pv_l1_k) {
            const uint32_t remaining = qsfa_pv_tokens - k1;
            const uint32_t panel_k = remaining < pv_l1_k ? remaining : pv_l1_k;
            const uint32_t kv_bank = qsfa_kv_cursor++ % 3;
            const uint32_t kv_event = qsfa_kv_events[kv_bank];
            auto v_l1 = qsfa_l1[(4 + kv_bank) * qsfa_bank_l1];
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kv_event);
            for (uint32_t k0 = 0; k0 < panel_k; k0 += pv_l0_k) {
                const uint32_t k_left = panel_k - k0;
                const uint32_t k = k_left < pv_l0_k ? k_left : pv_l0_k;
                const uint32_t k_aligned = (k + 15) / 16 * 16;
@LOAD_VALUE_PANEL@
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kv_event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kv_event);
            const uint32_t qp_bank = qp_base + k1 / pv_l1_k;
            const uint32_t qp_event = qsfa_qp_events[qp_bank];
@SELECT_P_BANK@
            if (n0 == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(qp_event);
                AscendC::Nd2NzParams p_copy{};
                p_copy.ndNum = 1;
                p_copy.nValue = qsfa_heads;
                p_copy.dValue = panel_k;
                p_copy.srcDValue = qsfa_s2;
                p_copy.dstNzC0Stride = qsfa_heads;
                p_copy.dstNzNStride = 1;
                AscendC::DataCopy(p_l1, probs[qsfa_pv_result_offset + k1], p_copy);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(qp_event);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(qp_event);
            }
            for (uint32_t k0 = 0; k0 < panel_k; k0 += pv_l0_k) {
                const uint32_t k_left = panel_k - k0;
                const uint32_t k = k_left < pv_l0_k ? k_left : pv_l0_k;
                const uint32_t k_aligned = (k + 15) / 16 * 16;
                @AB_WAIT@
                @LOAD_V@
                @LOAD_P@
                @MATMUL@
            }
            if (n0 + qsfa_n_panel == qsfa_s2) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(qp_event);
            }
            if (k1 + panel_k == qsfa_pv_tokens) {
                @FIXPIPE@
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kv_event);
        }
    }
}
""".replace("@AB_WAIT@", _AB_WAIT)
    .replace(
        "@LOAD_V@",
        _load3d("b_l0", "v_l1[k0 * qsfa_n_panel]", "k_aligned", "qsfa_n_panel", transpose=True),
    )
    .replace(
        "@LOAD_P@",
        _load3d("a_l0", "p_l1[k0 * qsfa_heads]", "qsfa_heads", "k_aligned", transpose=False),
    )
    .replace(
        "@MATMUL@",
        _matmul("qsfa_n_panel", "k", "k1 == 0 && k0 == 0", "k1 + k0 + k == qsfa_pv_tokens"),
    )
    .replace(
        "@FIXPIPE@",
        _fixpipe("partial[qsfa_pv_result_offset + n0]", "qsfa_n_panel"),
    )
)


_PV_VALUE_LOAD = """\
                AscendC::DataCopyParams v_copy{};
                v_copy.blockCount = qsfa_n_panel / qsfa_c0;
                v_copy.blockLen = k;
                v_copy.srcStride = qsfa_s2 - k;
                v_copy.dstStride = k_aligned - k;
                const uint64_t v_offset = qsfa_pv_kv_offset + n0 * qsfa_s2 + (k1 + k0) * qsfa_c0;
                AscendC::DataCopy(v_l1[k0 * qsfa_n_panel], kv_nz[v_offset], v_copy);"""

_PV_BLOCKED_VALUE_LOAD = _PV_VALUE_LOAD.replace(
    "n0 * qsfa_s2 + (k1 + k0) * qsfa_c0",
    "(k1 + k0) * qsfa_combined + n0 * qsfa_n_panel",
).replace("qsfa_s2 - k", "qsfa_n_panel - k")

_PV = _PV_TEMPLATE.replace("@LOAD_VALUE_PANEL@", _PV_VALUE_LOAD)
_PV_BLOCKED_KV = _PV_TEMPLATE.replace("@LOAD_VALUE_PANEL@", _PV_BLOCKED_VALUE_LOAD)

_P_NATIVE_BANK = "            auto p_l1 = qsfa_l1[qp_bank * qsfa_bank_l1];"

_P_RESIDENT_QUERY_BANK = """
            const uint32_t p_l1_offset = qp_bank == qsfa_query_l1_bank
                ? 3 * qsfa_bank_l1 + qsfa_heads * pv_l1_k
                : qp_bank * qsfa_bank_l1;
            auto p_l1 = qsfa_l1[p_l1_offset];
""".strip("\n")


def make_cube_lifecycle(T, *, resident_query=False):
    """Bracket all Cube stages once in the same explicit Cube scope."""
    init_source = _INITIALIZE + _RESIDENT_QUERY_INIT if resident_query else _INITIALIZE
    drain_source = _FINALIZE + _RESIDENT_QUERY_DRAIN if resident_query else _FINALIZE

    @T.macro
    def initialize():
        T._src_code(init_source)

    @T.macro
    def finalize():
        T._src_code(drain_source)

    return initialize, finalize


def make_cube_stages(T, *, resident_query=False, blocked_kv=False):
    """Return QK and PV macros for the fixed 64-head, 512-token exchange.

    The prim_func must name its GM Buffers ``query``, ``kv_nz``, ``scores``,
    ``probs``, and ``partial`` to match the AscendC hardware fragments. Expand
    each macro at a single source site inside the initialized Cube scope.
    Only active stages with 1–512 logical tokens should reach these macros.
    The kernel uses the preallocated TileLang L1/L0 banks for this Cube scope.
    Resident Q requires the matching lifecycle and four panels per query row,
    issued in KV slot order 0 through 3.
    """
    qk_source = (
        (_QK_BLOCKED_KV if blocked_kv else _QK)
        .replace(
            "@LOAD_QUERY_PANEL@",
            _Q_RESIDENT_LOAD if resident_query else _Q_PANEL_LOAD,
        )
        .replace(
            "@RELEASE_QUERY_PANEL@",
            _Q_RESIDENT_RELEASE if resident_query else _Q_PANEL_RELEASE,
        )
    )
    pv_source = (_PV_BLOCKED_KV if blocked_kv else _PV).replace(
        "@SELECT_P_BANK@",
        _P_RESIDENT_QUERY_BANK if resident_query else _P_NATIVE_BANK,
    )

    @T.macro
    def qk(query, kv_nz, scores, core, row, kv_slot, score_slot, valid_tokens):
        qsfa_qk_query_offset = T.alloc_var("int64", T.Cast("int64", row) * 64 * 576)
        qsfa_qk_kv_offset = T.alloc_var("int64", (T.Cast("int64", core) * 4 + kv_slot) * 512 * 576)
        qsfa_qk_result_offset = T.alloc_var("int64", (T.Cast("int64", core) * 2 + score_slot) * 128 * 512)
        qsfa_qk_tokens = T.alloc_var("uint32", T.Cast("uint32", valid_tokens))
        if resident_query:
            qsfa_qk_panel = T.alloc_var("uint32", T.Cast("uint32", kv_slot))
            T._src_code(
                qk_source,
                query.access_ptr("r"),
                kv_nz.access_ptr("r"),
                scores.access_ptr("w"),
                qsfa_qk_query_offset,
                qsfa_qk_kv_offset,
                qsfa_qk_result_offset,
                qsfa_qk_tokens,
                qsfa_qk_panel,
            )
        else:
            T._src_code(
                qk_source,
                query.access_ptr("r"),
                kv_nz.access_ptr("r"),
                scores.access_ptr("w"),
                qsfa_qk_query_offset,
                qsfa_qk_kv_offset,
                qsfa_qk_result_offset,
                qsfa_qk_tokens,
            )

    @T.macro
    def pv(kv_nz, probs, partial, core, kv_slot, score_slot, valid_tokens):
        qsfa_pv_kv_offset = T.alloc_var("int64", (T.Cast("int64", core) * 4 + kv_slot) * 512 * 576)
        qsfa_pv_result_offset = T.alloc_var("int64", (T.Cast("int64", core) * 2 + score_slot) * 128 * 512)
        qsfa_pv_tokens = T.alloc_var("uint32", T.Cast("uint32", valid_tokens))
        T._src_code(
            pv_source,
            kv_nz.access_ptr("r"),
            probs.access_ptr("r"),
            partial.access_ptr("w"),
            qsfa_pv_kv_offset,
            qsfa_pv_result_offset,
            qsfa_pv_tokens,
        )

    return qk, pv
