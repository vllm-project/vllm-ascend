import torch

from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter


class _FakeLayer:
    prefix = "model.layers.0.mlp.gate_proj"

    def __init__(self):
        self.__dict__ = {}


def test_tid2eid_int32_cache_invalidates_on_identity_change():
    router = AscendFusedTopKRouter.__new__(AscendFusedTopKRouter)
    tid2eid = torch.arange(4, dtype=torch.int64)
    router.tid2eid = tid2eid
    router._tid2eid_int32_cache = None

    first = router._get_tid2eid_int32(tid2eid)
    assert first.dtype == torch.int32
    second = router._get_tid2eid_int32(tid2eid)
    assert second is first, "same source tensor must reuse the cached int32 view"

    # EPLB rebalance reassigns tid2eid; the cache must follow.
    new_tid2eid = torch.arange(4, dtype=torch.int64)
    router.tid2eid = new_tid2eid
    third = router._get_tid2eid_int32(new_tid2eid)
    assert third is not first
    assert third.dtype == torch.int32


def test_mega_moe_max_tokens_property_lazily_caches():
    from unittest.mock import patch

    import vllm_ascend.ops.fused_moe.moe_comm_method as mcm

    # Tests build this object via object.__new__, bypassing __init__.
    impl = mcm.FusedMC2CommImpl.__new__(mcm.FusedMC2CommImpl)
    with patch.object(mcm, "get_ascend_config") as mock_cfg:
        mock_cfg.return_value.mega_moe_max_tokens = 512
        assert impl.mega_moe_max_tokens == 512
        mock_cfg.return_value.mega_moe_max_tokens = 999
        assert impl.mega_moe_max_tokens == 512, "value must be cached after first read"


def test_tp_rank_cache_handles_falsy_zero():
    from vllm_ascend.quantization.method_adapters import AscendLinearMethod

    layer = _FakeLayer()
    tp_rank = AscendLinearMethod._resolve_tp_rank(layer)
    assert tp_rank == 0
    layer.__dict__["_ascend_tp_rank"] = tp_rank
    assert layer.__dict__.get("_ascend_tp_rank") == 0, "falsy 0 must still hit the cache"


def test_fused_mc2_placeholder_constants_are_shared_and_empty():
    from vllm_ascend.ops.fused_moe.routed_experts import _EMPTY_FP32_SCALES, _EMPTY_INT64_SCALES

    assert _EMPTY_INT64_SCALES[0].dtype == torch.int64
    assert _EMPTY_INT64_SCALES[0].numel() == 0
    assert _EMPTY_FP32_SCALES[0].dtype == torch.float32
    assert _EMPTY_FP32_SCALES[0].numel() == 0
