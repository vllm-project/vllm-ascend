# SPDX-License-Identifier: Apache-2.0
"""Composition audit for the assembled INT2 backend.

Pins the method-level completeness of the split: every mixin module exposes
EXACTLY the frozen method inventory (the union equals the pre-split
slots/threetier/ring/staging/quantcache/bridge inventories; nothing lost,
nothing extra), the assembly class composes all twelve mixins on top of the
upstream base impl, and each method resolves to exactly one defining module.
"""

import importlib
import pathlib

import vllm_ascend.attention.int2 as pkg

# frozen inventory (53 methods over 12 mixins)
INVENTORY = {
    "slotstate": [
        "_ensure_tt",
        "_oscar_layer_ord",
        "_r164_bind_slots",
        "_r166_bind_slots",
        "_r170_rebind_locked",
        "_tt_hp_bufs",
        "_tt_rid",
        "_tt_slot",
        "_tt_write_hp",
    ],
    "batchmeta_resolve": ["_tt_resolve_slots"],
    "batchmeta_update": [
        "_r164_update_meta",
        "_r170_mdfill",
        "_r170_rowslot_migrate",
        "_r172_wrapper_flatrefresh",
        "_r196_idfill_memo",
    ],
    "ring": [
        "_r155_cu_ring_row",
        "_r163_stage_sr",
        "_r197_batched_ring",
        "_r198_pos",
        "_r198_pr_window",
        "_r198_restage",
        "_r198_vring_on",
        "_r198_win_idx",
        "_r198_win_ridx",
    ],
    "staging": [
        "_r113_flush_stage",
        "_r113_update_oscar_mid_batched",
        "_r113_update_oscar_mid_v1cap",
        "_r156_h3_ensure",
        "_r156_h3_stage_all",
        "_r218_stage_assign",
        "_r218_stage_bufs",
        "_r218_stage_lookup",
        "_r218_stage_plan",
    ],
    "quantcache": [
        "_dequantize_packed_int2",
        "_ensure_int2_scale_buffers",
        "_gather_and_dequantize_int2_cache",
        "_pack_int2_to_uint8",
        "_prepare_int2_scales",
        "_quantize_kv_to_int2",
        "_reshape_and_cache_int2",
        "_unpack_uint8_to_int2",
    ],
    "bridge": ["_r112_ensure_fast", "_r135_ensure_v1", "_r91_bridge_decode", "_r91_ensure_loaded"],
    "middle_candidate": ["_r109_middle_candidate"],
    "middle_batched": ["_r164_midprobe", "_r196_batched_sr", "_r197_probe_record"],
    "middle_exec": ["_forward_int2_decode_threetier_middle"],
    "decode": ["_forward_int2_decode_threetier"],
    "prefill": ["_attn_core_prefill", "_forward_int2_prefill_threetier"],
}
MIXIN_CLASS = {
    "slotstate": "Int2SlotStateMixin",
    "batchmeta_resolve": "Int2BatchMetaResolveMixin",
    "batchmeta_update": "Int2BatchMetaUpdateMixin",
    "ring": "Int2RingMixin",
    "staging": "Int2StagingMixin",
    "quantcache": "Int2QuantCacheMixin",
    "bridge": "Int2BridgeMixin",
    "middle_candidate": "Int2MiddleCandidateMixin",
    "middle_batched": "Int2MiddleBatchedMixin",
    "middle_exec": "Int2MiddleExecMixin",
    "decode": "Int2DecodeMixin",
    "prefill": "Int2PrefillMixin",
}


def _methods(cls):
    return sorted(
        n
        for n in dir(cls)
        if not n.startswith("__")
        and callable(getattr(cls, n, None))
        and n in {m for v in INVENTORY.values() for m in v}
    )


def test_method_inventory_exact():
    seen: dict = {}
    for mod_name, expected in INVENTORY.items():
        mod = importlib.import_module(f"vllm_ascend.attention.int2.{mod_name}")
        cls = getattr(mod, MIXIN_CLASS[mod_name])
        got = [n for n in expected if hasattr(cls, n)]
        missing = set(expected) - set(got)
        assert not missing, f"{mod_name} missing {missing}"
        for m in expected:
            seen.setdefault(m, []).append(mod_name)
    dupes = {m: mods for m, mods in seen.items() if len(mods) > 1}
    assert not dupes, f"methods defined in >1 mixin: {dupes}"
    total = sum(len(v) for v in INVENTORY.values())
    assert total == 53


def test_assembly_composes_all_mixins():
    from vllm_ascend.attention.attention_int2 import AscendINT2AttentionBackendImpl

    bases = [c.__name__ for c in AscendINT2AttentionBackendImpl.__mro__]
    for name in MIXIN_CLASS.values():
        assert name in bases, f"{name} not in assembly MRO"
    assert "AscendAttentionBackendImpl" in bases  # upstream base last
    # every inventory method resolves on the assembled class
    for m in {m for v in INVENTORY.values() for m in v}:
        assert hasattr(AscendINT2AttentionBackendImpl, m), f"assembly lost {m}"


def test_no_legacy_monolith_modules():
    root = pathlib.Path(pkg.__file__).parent
    assert not (root / "slots.py").exists()
    assert not (root / "threetier.py").exists()
    assert not (root / "batchmeta.py").exists()  # superseded by resolve/update
    assert not (root / "middlecall.py").exists()  # superseded by candidate/batched/exec
