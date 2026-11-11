from types import SimpleNamespace

import pytest
from vllm.config import CUDAGraphMode

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend import utils
from vllm_ascend.ascend_forward_context import MoECommType


@pytest.mark.parametrize(
    ("small_batch", "decode", "prefill", "recompute", "uniform_graph", "expected"),
    [
        (MoECommType.MC2, MoECommType.MC2, MoECommType.MC2, False, True, True),
        (MoECommType.FUSED_MC2, MoECommType.MC2, MoECommType.MC2, False, True, False),
        (MoECommType.MC2, MoECommType.FUSED_MC2, MoECommType.MC2, False, True, False),
        (MoECommType.MC2, MoECommType.MC2, MoECommType.FUSED_MC2, True, True, False),
        (MoECommType.MC2, MoECommType.MC2, MoECommType.ALLTOALL, False, True, False),
        (MoECommType.MC2, MoECommType.MC2, MoECommType.ALLTOALL, True, True, True),
        (MoECommType.MC2, MoECommType.ALLTOALL, MoECommType.MC2, True, True, False),
        (MoECommType.MC2, MoECommType.MC2, MoECommType.MC2, False, False, False),
        (MoECommType.MC2, MoECommType.MC2, MoECommType.MC2, True, False, True),
    ],
)
def test_dp_sync_is_retained_when_any_batch_can_use_fused_mc2(
    monkeypatch, small_batch, decode, prefill, recompute, uniform_graph, expected
):
    ascend_config = SimpleNamespace(
        get_mc2_comm_alg=lambda: "default",
        scheduler_config=SimpleNamespace(recompute_scheduler_enable=recompute),
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL if uniform_graph else CUDAGraphMode.FULL_DECODE_ONLY
        ),
    )
    methods = {1: small_batch, 128: decode, 4096: prefill}
    monkeypatch.setattr(utils, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(utils, "is_mega_moe_supported", lambda: False)
    monkeypatch.setattr(utils, "is_moe_model", lambda _: True)
    monkeypatch.setattr(utils, "get_potential_max_tokens", lambda: 128)
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: False)
    monkeypatch.setattr(afc, "select_moe_comm_method", lambda tokens, _: methods[tokens])

    assert utils.should_skip_allreduce_across_dp_group(config) is expected
