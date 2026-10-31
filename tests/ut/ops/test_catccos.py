from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import catccos as catccos_module
from vllm_ascend.ops.fused_moe import moe_comm_method as moe_comm_module
from vllm_ascend.ops.fused_moe.catccos import (
    CATCCOS_SHMEM_BUFFER_BYTES,
    CATCCOS_TOKEN_ALIGNMENT,
    CatCCOSRuntime,
    CatCCOSRuntimeConfig,
    pad_catccos_inputs,
    validate_catccos_fused_input,
    validate_catccos_moe_config,
    validate_catccos_selection,
)
from vllm_ascend.ops.fused_moe.moe_comm_method import CatCCOSCommImpl
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEFusedExpertsInput,
    MoEQuantParams,
    MoERoutingParams,
    MoEWeights,
)
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType


def _make_moe_config(**overrides):
    values = {
        "ep_size": 8,
        "num_experts": 8,
        "num_local_experts": 1,
        "experts_per_token": 8,
        "hidden_dim": 256,
        "intermediate_size_per_partition": 256,
        "in_dtype": torch.bfloat16,
        "global_redundant_expert_num": 0,
        "has_bias": False,
        "is_lora_enabled": False,
        "swiglu_limit": 0.0,
        "activation": "silu",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_fused_input(m: int = 65, **overrides):
    values = {
        "hidden_states": torch.randn(m, 256, dtype=torch.bfloat16),
        "topk_weights": torch.randn(m, 8, dtype=torch.bfloat16),
        "topk_ids": torch.zeros(m, 8, dtype=torch.int64),
        "weights": MoEWeights(
            w1=torch.randn(1, 256, 512, dtype=torch.bfloat16),
            w2=torch.randn(1, 256, 256, dtype=torch.bfloat16),
        ),
        "routing": MoERoutingParams(
            expert_map=None,
            global_redundant_expert_num=0,
            mc2_mask=None,
            apply_router_weight_on_input=False,
        ),
        "quant": MoEQuantParams(),
    }
    values.update(overrides)
    return MoEFusedExpertsInput(**values)


def test_runtime_config_defaults_and_required_values(monkeypatch, tmp_path):
    library = tmp_path / "libcatccos_torch.so"
    library.touch()
    monkeypatch.setenv("VLLM_ASCEND_CATCCOS_LIBRARY_PATH", str(library))
    monkeypatch.setenv("VLLM_ASCEND_CATCCOS_STORE_ADDR", "tcp://127.0.0.1:29411")
    monkeypatch.delenv("VLLM_ASCEND_CATCCOS_LOCAL_MEM_SIZE", raising=False)

    config = CatCCOSRuntimeConfig.from_env()

    assert config.library_path == str(library)
    assert config.store_addr == "tcp://127.0.0.1:29411"
    assert config.local_mem_size == 1 << 30


def test_runtime_config_rejects_insufficient_symmetric_memory(monkeypatch, tmp_path):
    library = tmp_path / "libcatccos_torch.so"
    library.touch()
    monkeypatch.setenv("VLLM_ASCEND_CATCCOS_LIBRARY_PATH", str(library))
    monkeypatch.setenv("VLLM_ASCEND_CATCCOS_STORE_ADDR", "tcp://127.0.0.1:29411")
    monkeypatch.setenv(
        "VLLM_ASCEND_CATCCOS_LOCAL_MEM_SIZE",
        str(CATCCOS_SHMEM_BUFFER_BYTES - 1),
    )

    with pytest.raises(ValueError, match=str(CATCCOS_SHMEM_BUFFER_BYTES)):
        CatCCOSRuntimeConfig.from_env()


def test_runtime_propagates_init_failure(monkeypatch, tmp_path):
    library = tmp_path / "libcatccos_torch.so"
    library.touch()
    monkeypatch.setenv("VLLM_ASCEND_CATCCOS_LIBRARY_PATH", str(library))
    monkeypatch.setenv("VLLM_ASCEND_CATCCOS_STORE_ADDR", "tcp://127.0.0.1:29411")
    monkeypatch.setattr(catccos_module, "get_ascend_device_type", lambda: AscendDeviceType.A2)
    ep_group = SimpleNamespace(rank_in_group=3, world_size=8, barrier=MagicMock())
    monkeypatch.setattr(catccos_module, "get_ep_group", lambda: ep_group)
    ops = SimpleNamespace(
        load_library=MagicMock(),
        catccos=SimpleNamespace(
            dispatch_ffn_combine=MagicMock(),
            init=MagicMock(return_value=7),
        ),
    )
    monkeypatch.setattr(catccos_module.torch, "ops", ops)

    with pytest.raises(RuntimeError, match="initialization failed"):
        CatCCOSRuntime(_make_moe_config())

    ep_group.barrier.assert_called_once_with()


def test_runtime_initializes_before_first_dispatch(monkeypatch, tmp_path):
    library = tmp_path / "libcatccos_torch.so"
    library.write_bytes(b"catccos")
    monkeypatch.setenv("VLLM_ASCEND_CATCCOS_LIBRARY_PATH", str(library))
    monkeypatch.setenv("VLLM_ASCEND_CATCCOS_STORE_ADDR", "tcp://127.0.0.1:29411")
    monkeypatch.setattr(catccos_module, "get_ascend_device_type", lambda: AscendDeviceType.A2)
    ep_group = SimpleNamespace(rank_in_group=3, world_size=8, barrier=MagicMock())
    monkeypatch.setattr(catccos_module, "get_ep_group", lambda: ep_group)
    dispatch = MagicMock(return_value=torch.ones(64, 256, dtype=torch.bfloat16))
    init = MagicMock(return_value=0)
    ops = SimpleNamespace(
        load_library=MagicMock(),
        catccos=SimpleNamespace(dispatch_ffn_combine=dispatch, init=init),
    )
    monkeypatch.setattr(catccos_module.torch, "ops", ops)

    runtime = CatCCOSRuntime(_make_moe_config())

    init.assert_called_once_with(3, 8, 1 << 30, "tcp://127.0.0.1:29411")
    ep_group.barrier.assert_called_once_with()
    runtime.dispatch(
        torch.randn(64, 256, dtype=torch.bfloat16),
        torch.zeros(64, 8, dtype=torch.int64),
        torch.rand(64, 8, dtype=torch.bfloat16),
        torch.randn(1, 256, 512, dtype=torch.bfloat16),
        torch.randn(1, 256, 256, dtype=torch.bfloat16),
    )
    assert dispatch.call_args.args[1].dtype == torch.int32
    assert dispatch.call_args.args[2].dtype == torch.float32


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"in_dtype": torch.float16}, "BF16"),
        ({"ep_size": 1}, "expert parallelism"),
        ({"num_local_experts": 17, "num_experts": 136}, "at most 16"),
        ({"experts_per_token": 4}, "TopK=8"),
        ({"hidden_dim": 255}, "divisible by 256"),
        ({"intermediate_size_per_partition": 255}, "divisible by 512"),
        ({"global_redundant_expert_num": 1}, "redundant experts"),
        ({"has_bias": True}, "expert bias"),
        ({"is_lora_enabled": True}, "LoRA"),
        ({"swiglu_limit": 1.0}, "SwiGLU clamp"),
        ({"activation": "gelu"}, "SiLU"),
    ],
)
def test_moe_config_capability_gates(monkeypatch, override, message):
    monkeypatch.setattr(
        "vllm_ascend.ops.fused_moe.catccos.get_ascend_device_type",
        lambda: AscendDeviceType.A2,
    )

    with pytest.raises(ValueError, match=message):
        validate_catccos_moe_config(_make_moe_config(**override))


def test_selection_rejects_four_rank_ep_for_128_experts(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.ops.fused_moe.catccos.get_ascend_device_type",
        lambda: AscendDeviceType.A2,
    )
    monkeypatch.setattr(
        "vllm_ascend.ops.fused_moe.catccos.get_ep_group",
        lambda: SimpleNamespace(world_size=4),
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            get_num_experts=lambda: 128,
            hf_text_config=SimpleNamespace(num_experts_per_tok=8),
        ),
        lora_config=None,
    )

    with pytest.raises(ValueError, match="at most 16"):
        validate_catccos_selection(config, False)


def test_selection_rejects_model_quantization(monkeypatch):
    monkeypatch.setattr(catccos_module, "get_ascend_device_type", lambda: AscendDeviceType.A2)
    monkeypatch.setattr(catccos_module, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            get_num_experts=lambda: 128,
            hf_text_config=SimpleNamespace(num_experts_per_tok=8),
            quantization="ascend",
        ),
        lora_config=None,
    )

    with pytest.raises(ValueError, match="unquantized BF16"):
        validate_catccos_selection(config, False)


@pytest.mark.parametrize(
    ("parallel_eplb", "ascend_eplb"),
    [
        (True, {}),
        (False, {"expert_map_path": "map.json"}),
        (False, {"expert_map_record_path": "record.json"}),
        (False, {"num_redundant_experts": 1}),
        (False, {"dynamic_eplb": True}),
    ],
)
def test_selection_rejects_eplb(monkeypatch, parallel_eplb, ascend_eplb):
    monkeypatch.setattr(catccos_module, "get_ascend_device_type", lambda: AscendDeviceType.A2)
    monkeypatch.setattr(catccos_module, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    eplb_values = {
        "dynamic_eplb": False,
        "expert_map_path": None,
        "expert_map_record_path": None,
        "num_redundant_experts": 0,
    }
    eplb_values.update(ascend_eplb)
    monkeypatch.setattr(
        catccos_module,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=False,
            eplb_config=SimpleNamespace(**eplb_values),
        ),
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            pipeline_parallel_size=1,
            enable_eplb=parallel_eplb,
        ),
        model_config=SimpleNamespace(
            get_num_experts=lambda: 128,
            hf_text_config=SimpleNamespace(num_experts_per_tok=8),
        ),
        lora_config=None,
    )

    with pytest.raises(ValueError, match="EPLB"):
        validate_catccos_selection(config, False)


def test_selection_rejects_mixed_shared_expert_placement(monkeypatch):
    monkeypatch.setattr(catccos_module, "get_ascend_device_type", lambda: AscendDeviceType.A2)
    monkeypatch.setattr(catccos_module, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(
        catccos_module,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=False,
            mix_placement=True,
            eplb_config=SimpleNamespace(
                dynamic_eplb=False,
                expert_map_path=None,
                expert_map_record_path=None,
                num_redundant_experts=0,
            ),
        ),
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            pipeline_parallel_size=1,
            enable_eplb=False,
        ),
        model_config=SimpleNamespace(
            get_num_experts=lambda: 128,
            hf_text_config=SimpleNamespace(num_experts_per_tok=8),
        ),
        lora_config=None,
    )

    with pytest.raises(ValueError, match="mixed shared-expert placement"):
        validate_catccos_selection(config, False)


def test_pad_catccos_inputs_uses_m64_and_zero_routing_rows():
    fused_input = _make_fused_input(m=65)

    hidden, ids, weights, original_tokens = pad_catccos_inputs(
        fused_input.hidden_states,
        fused_input.topk_ids,
        fused_input.topk_weights,
    )

    assert hidden.shape[0] == 2 * CATCCOS_TOKEN_ALIGNMENT
    assert original_tokens == 65
    assert torch.count_nonzero(hidden[65:]) == 0
    assert torch.count_nonzero(ids[65:]) == 0
    assert torch.count_nonzero(weights[65:]) == 0


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"dynamic_eplb": True}, "dynamic EPLB"),
        ({"swiglu_limit": 1.0}, "SwiGLU clamp"),
        ({"lora_context": object()}, "LoRA"),
        ({"activation": "gelu"}, "SiLU"),
    ],
)
def test_fused_input_capability_gates(override, message):
    with pytest.raises(ValueError, match=message):
        validate_catccos_fused_input(
            _make_fused_input(**override),
            _make_moe_config(),
        )


@pytest.mark.parametrize(
    ("modifier", "message"),
    [
        (
            lambda fused: replace(
                fused,
                routing=replace(fused.routing, log2phy=torch.arange(8)),
            ),
            "non-linear expert placement",
        ),
        (
            lambda fused: replace(
                fused,
                routing=replace(fused.routing, global_redundant_expert_num=1),
            ),
            "redundant experts",
        ),
        (
            lambda fused: replace(
                fused,
                routing=replace(fused.routing, apply_router_weight_on_input=True),
            ),
            "router weights",
        ),
        (
            lambda fused: replace(
                fused,
                weights=replace(fused.weights, w1_bias=torch.zeros(1)),
            ),
            "expert bias",
        ),
        (
            lambda fused: replace(
                fused,
                quant=MoEQuantParams(quant_type=QuantType.W8A8),
            ),
            "unquantized BF16",
        ),
        (
            lambda fused: replace(fused, need_trans=True),
            "pre-transposed ND",
        ),
        (
            lambda fused: replace(
                fused,
                weights=replace(fused.weights, w1_scale=torch.ones(1)),
            ),
            "quantization side tensors",
        ),
        (
            lambda fused: replace(
                fused,
                hidden_states=fused.hidden_states.to(torch.float16),
            ),
            "BF16 hidden states",
        ),
    ],
)
def test_fused_input_nested_capability_gates(modifier, message):
    with pytest.raises(ValueError, match=message):
        validate_catccos_fused_input(
            modifier(_make_fused_input(m=1)),
            _make_moe_config(),
        )


def test_comm_impl_dispatches_padded_inputs_and_slices_output(monkeypatch):
    comm = CatCCOSCommImpl.__new__(CatCCOSCommImpl)
    comm.moe_config = _make_moe_config()
    comm.runtime = MagicMock()
    comm.runtime.dispatch.side_effect = lambda hidden, *_: torch.ones_like(hidden)
    event = MagicMock()
    stream = MagicMock()
    stream.record_event.return_value = event
    monkeypatch.setattr(torch.npu, "current_stream", lambda: stream)

    result = comm.fused_experts(_make_fused_input(m=65))

    assert result.routed_out.shape == (65, 256)
    dispatch_args = comm.runtime.dispatch.call_args.args
    assert dispatch_args[0].shape == (128, 256)
    assert dispatch_args[1].shape == (128, 8)
    assert stream.record_event.call_count == 2


def test_comm_impl_propagates_dispatch_failure(monkeypatch):
    comm = CatCCOSCommImpl.__new__(CatCCOSCommImpl)
    comm.moe_config = _make_moe_config()
    comm.runtime = MagicMock()
    comm.runtime.dispatch.side_effect = RuntimeError("507015")
    stream = MagicMock()
    monkeypatch.setattr(torch.npu, "current_stream", lambda: stream)

    with pytest.raises(RuntimeError, match="507015"):
        comm.fused_experts(_make_fused_input(m=64))


def test_setup_reuses_single_catccos_registry_backend(monkeypatch):
    class FakeCatCCOSCommImpl:
        init_count = 0

        def __init__(self, _):
            type(self).init_count += 1
            self.validate_count = 0

        def validate_compatible(self, _):
            self.validate_count += 1

    moe_config = _make_moe_config()
    monkeypatch.setattr(moe_comm_module, "catccos_moe_enabled", lambda: True)
    monkeypatch.setattr(moe_comm_module, "CatCCOSCommImpl", FakeCatCCOSCommImpl)
    monkeypatch.setattr(moe_comm_module, "AlltoAllCommImpl", MagicMock())
    monkeypatch.setattr(moe_comm_module, "AllGatherCommImpl", MagicMock())
    monkeypatch.setattr(moe_comm_module, "MC2CommImpl", MagicMock())
    monkeypatch.setattr(moe_comm_module, "FusedMC2CommImpl", MagicMock())
    moe_comm_module._MoECommMethods.clear()
    try:
        moe_comm_module.setup_moe_comm_method(moe_config)
        backend = moe_comm_module.get_moe_comm_method(MoECommType.CATCCOS)
        moe_comm_module.setup_moe_comm_method(moe_config)

        assert FakeCatCCOSCommImpl.init_count == 1
        assert backend.validate_count == 1
    finally:
        moe_comm_module._MoECommMethods.clear()
