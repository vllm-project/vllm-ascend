from types import SimpleNamespace

from vllm.config import CUDAGraphMode

from vllm_ascend import ascend_forward_context as forward_context
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.utils import AscendDeviceType


def _make_vllm_config(
    *,
    model_type="gemma4",
    architectures=("Gemma4ForConditionalGeneration",),
    enforce_eager=False,
    cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    world_size=2,
):
    hf_config = SimpleNamespace(model_type=model_type, architectures=architectures)
    hf_text_config = SimpleNamespace(model_type=model_type, num_experts_per_tok=2)
    model_config = SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=hf_text_config,
        enforce_eager=enforce_eager,
    )
    parallel_config = SimpleNamespace(
        enable_expert_parallel=True,
        world_size_across_dp=world_size,
        pipeline_parallel_size=1,
    )
    compilation_config = SimpleNamespace(cudagraph_mode=cudagraph_mode)
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        compilation_config=compilation_config,
    )


def _patch_a5_moe(monkeypatch):
    monkeypatch.setattr(forward_context, "is_moe_model", lambda _: True)
    monkeypatch.setattr(forward_context, "get_ascend_device_type", lambda: AscendDeviceType.A5)
    monkeypatch.setattr(forward_context, "get_ep_group", lambda: SimpleNamespace(world_size=2))
    monkeypatch.setattr(forward_context, "get_mc2_tokens_capacity", lambda: 8)


def test_a5_gemma4_graph_moe_uses_allgather(monkeypatch):
    _patch_a5_moe(monkeypatch)

    moe_comm_type = forward_context.select_moe_comm_method(1, _make_vllm_config())

    assert moe_comm_type == MoECommType.ALLGATHER


def test_a5_gemma4_eager_moe_uses_allgather(monkeypatch):
    _patch_a5_moe(monkeypatch)

    moe_comm_type = forward_context.select_moe_comm_method(
        1,
        _make_vllm_config(enforce_eager=True),
    )

    assert moe_comm_type == MoECommType.ALLGATHER


def test_a5_gemma4_profile_moe_keeps_default_mc2_selection(monkeypatch):
    _patch_a5_moe(monkeypatch)

    moe_comm_type = forward_context.select_moe_comm_method(
        1,
        _make_vllm_config(),
        in_profile_run=True,
    )

    assert moe_comm_type == MoECommType.MC2


def test_a5_non_gemma4_graph_moe_keeps_default_mc2_selection(monkeypatch):
    _patch_a5_moe(monkeypatch)

    moe_comm_type = forward_context.select_moe_comm_method(
        1,
        _make_vllm_config(model_type="qwen3_moe", architectures=("Qwen3MoeForCausalLM",)),
    )

    assert moe_comm_type == MoECommType.MC2
