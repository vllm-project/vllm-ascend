# SPDX-License-Identifier: Apache-2.0
"""MTP5 QLI with two FP16 L1 slots and QK issued ahead of score MMAD."""

from functools import partial
from textwrap import indent

from qli_tilelang_v13 import (
    _COMPUTE_SCORE_PACKED,
    _LOAD_SCORE_B_PACKED,
    _QK_FIXPIPE_PACKED,
    _STORE_SCORE_PACKED,
    PAGE_SIZE,
    ROWS_PER_REQUEST,
)
from qli_tilelang_v17 import _KEY_EVENT_BASE, _key_panel_source
from qli_tilelang_v17 import build_tilelang_call as build_v17_call
from qli_tilelang_v18 import _PRUNED_TOPK

_PANELS_PER_TILE = 2048 // PAGE_SIZE
_K_PANEL_ELEMENTS = PAGE_SIZE * 128
_QK_L1_SLOT_ELEMENTS = ROWS_PER_REQUEST * 32 * PAGE_SIZE

# IDs are scoped to each producer and consumer pipe pair. IDs below 4
# remain available to the TileLang sync pass, as in V23.
_K_BANK_EVENT = 4
_SCORE_B_EVENT = 5
_QK_C_EVENT = 5
_SCORE_C_EVENT = 7
_QK_L1_EVENT_BASE = 4


def _ownership_flags(op, *, software_qk_events=True, setup=""):
    qk_event = f"    AscendC::{op}<AscendC::HardEvent::FIX_M>({_QK_C_EVENT});\n" if software_qk_events else ""
    setup_line = indent(setup.strip(), "    ") + "\n" if setup else ""
    return f"""
{{
{setup_line}    AscendC::{op}<AscendC::HardEvent::M_MTE1>({_K_BANK_EVENT});
    AscendC::{op}<AscendC::HardEvent::M_MTE1>({_SCORE_B_EVENT});
{qk_event}    AscendC::{op}<AscendC::HardEvent::FIX_M>({_SCORE_C_EVENT});
}}
"""


def _load_key(panel):
    return f"""
{{
    const uint32_t k_slot = ({panel}) & 1U;
    const uint32_t k_event = {_KEY_EVENT_BASE} + k_slot;
    WaitFlag<HardEvent::M_MTE1>({_K_BANK_EVENT});
    WaitFlag<HardEvent::MTE2_MTE1>(k_event);
    tl::ascend::copy_l1_to_l0b<int8_t, 128, 128, true>(
        key_b[0], key_l1[k_slot * {_K_PANEL_ELEMENTS}], 128, 128);
    SetFlag<HardEvent::MTE1_MTE2>(k_event);
    SetFlag<HardEvent::MTE1_M>({_K_BANK_EVENT});
}}
"""


_QK_MMAD = "tl::ascend::mma<int8_t, int32_t, 192, 128>(query_a[0], key_b[0], qk_l0c[0], true, 128);"


def _issue_qk(qk_mmad, *, software_qk_events):
    wait_free = f"WaitFlag<HardEvent::FIX_M>({_QK_C_EVENT});\n" if software_qk_events else ""
    post_ready = f"SetFlag<HardEvent::M_FIX>({_QK_C_EVENT});\n" if software_qk_events else ""
    return f"""
WaitFlag<HardEvent::MTE1_M>({_K_BANK_EVENT});
{wait_free}{qk_mmad.strip()}
SetFlag<HardEvent::M_MTE1>({_K_BANK_EVENT});
{post_ready}"""


def _with_qk_l1_slot(panel, body):
    return f"""
{{
    const uint32_t l1_slot = ({panel}) & 1U;
    const uint32_t l1_event = {_QK_L1_EVENT_BASE} + l1_slot;
    auto slot_l1 = qk_l1[l1_slot * {_QK_L1_SLOT_ELEMENTS}];
{indent(body.strip(), "    ")}
}}
"""


def _convert_qk(panel, qk_fixpipe, *, software_qk_events):
    wait_ready = f"WaitFlag<HardEvent::M_FIX>({_QK_C_EVENT});\n" if software_qk_events else ""
    post_free = f"SetFlag<HardEvent::FIX_M>({_QK_C_EVENT});\n" if software_qk_events else ""
    return _with_qk_l1_slot(
        panel,
        f"""
{wait_ready}{qk_fixpipe.replace("qk_l1", "slot_l1").strip()}
{post_free}SetFlag<HardEvent::FIX_MTE1>(l1_event);
""",
    )


# FixPipe writes [head, token] zN with successive head fractals contiguous.
# L0B expects nZ, so each token block advances through head fractals with a
# destination stride equal to the number of token blocks. Load2D transposes
# the elements inside each 16x16 fractal.
_LOAD_SCORE_B_2D = """
{
    constexpr uint32_t fractal_elements = 16U * 16U;
    constexpr uint32_t head_blocks = 192U / 16U;
    constexpr uint32_t token_blocks = 128U / 16U;
    AscendC::LoadData2DParams cfg;
    cfg.startIndex = 0;
    cfg.repeatTimes = head_blocks;
    cfg.srcStride = 1;
    cfg.dstGap = token_blocks - 1;
    cfg.sid = 0;
    cfg.ifTranspose = true;
    cfg.addrMode = 0;
    #pragma unroll
    for (uint32_t token_block = 0; token_block < token_blocks; ++token_block) {
        AscendC::LoadData(score_b[token_block * fractal_elements],
                          qk_l1[token_block * head_blocks * fractal_elements], cfg);
    }
}
"""


def _load_score(panel, source):
    return _with_qk_l1_slot(
        panel,
        f"""
WaitFlag<HardEvent::M_MTE1>({_SCORE_B_EVENT});
WaitFlag<HardEvent::FIX_MTE1>(l1_event);
{source.replace("qk_l1", "slot_l1").strip()}
SetFlag<HardEvent::MTE1_M>({_SCORE_B_EVENT});
""",
    )


_ISSUE_SCORE = f"""
WaitFlag<HardEvent::MTE1_M>({_SCORE_B_EVENT});
WaitFlag<HardEvent::FIX_M>({_SCORE_C_EVENT});
{_COMPUTE_SCORE_PACKED.strip()}
SetFlag<HardEvent::M_MTE1>({_SCORE_B_EVENT});
SetFlag<HardEvent::M_FIX>({_SCORE_C_EVENT});
"""


def _store_score(panel):
    return f"""
{{
    const uint32_t cube_panel = {panel};
    WaitFlag<HardEvent::M_FIX>({_SCORE_C_EVENT});
{indent(_STORE_SCORE_PACKED.strip(), "    ")}
    SetFlag<HardEvent::FIX_M>({_SCORE_C_EVENT});
}}
"""


def _panel_code(*blocks, spaces=4):
    return indent("\n".join(block.strip() for block in blocks), " " * spaces)


def _score_ring(
    qk_mmad, qk_fixpipe, *, software_qk_events=True, setup="", key_panel_source=_key_panel_source, score_load_2d=False
):
    issue_qk = _issue_qk(qk_mmad, software_qk_events=software_qk_events)
    convert_qk = partial(_convert_qk, qk_fixpipe=qk_fixpipe, software_qk_events=software_qk_events)
    load_score = partial(_load_score, source=_LOAD_SCORE_B_2D if score_load_2d else _LOAD_SCORE_B_PACKED)
    prologue = _panel_code(
        key_panel_source("0U"),
        key_panel_source("1U"),
        _load_key("0U"),
        issue_qk,
        convert_qk("0U"),
    )
    # FIX order is Convert(0), Convert(1), StoreScore(0), Convert(2), ... .
    # StoreScore(p-2) waits for Score(p-2), which waits for LoadScore(p-2).
    # This releases the converted L1 slot before Convert(p) overwrites it.
    steady = _panel_code(
        _load_key("score_panel + 1U"),
        key_panel_source("score_panel + 2U", f"score_panel + 2U < {_PANELS_PER_TILE}U"),
        issue_qk,
        load_score("score_panel"),
        _ISSUE_SCORE,
        convert_qk("score_panel + 1U"),
        _store_score("score_panel"),
        spaces=8,
    )
    epilogue = _panel_code(
        load_score(f"{_PANELS_PER_TILE - 1}U"),
        _ISSUE_SCORE,
        _store_score(f"{_PANELS_PER_TILE - 1}U"),
    )
    tile = f"""
{{
    using namespace AscendC;
{prologue}
    for (uint32_t score_panel = 0; score_panel + 1U < {_PANELS_PER_TILE}U; ++score_panel) {{
{steady}
    }}
{epilogue}
}}
"""
    init = _ownership_flags("SetFlag", software_qk_events=software_qk_events, setup=setup)
    drain = _ownership_flags("WaitFlag", software_qk_events=software_qk_events)
    return init, tile, drain


build_tilelang_call = partial(
    build_v17_call,
    topk_source=_PRUNED_TOPK,
    score_ring=_score_ring(_QK_MMAD, _QK_FIXPIPE_PACKED),
)
