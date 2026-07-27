from types import SimpleNamespace

import torch
from vllm.compilation.passes.inductor_pass import get_pass_context, pass_context
from vllm.config.utils import Range

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.compilation.compiler_interface import _RangeBoundFusionPass
from vllm_ascend.compilation.passes import allreduce_rmsnorm_fusion_pass
from vllm_ascend.compilation.passes.allreduce_rmsnorm_fusion_pass import (
    MatmulAllReduceAddRMSNormPass,
    _is_supported_910c_down_proj_match,
)


def test_range_bound_fusion_pass_uses_bound_range_and_restores_context():
    seen_ranges = []

    def manager(graph, example_inputs=None, compiler_config=None):
        seen_ranges.append(get_pass_context().compile_range)
        return graph

    graph = object()
    bound_pass = _RangeBoundFusionPass(manager, Range(1, 2))

    with pass_context(Range(3, 2048)):
        assert bound_pass(graph) is graph
        assert get_pass_context().compile_range == Range(3, 2048)

    assert seen_ranges == [Range(1, 2)]


def _match_tensor(shape, dtype=torch.bfloat16):
    return SimpleNamespace(meta={"val": torch.empty(shape, dtype=dtype, device="meta")})


def _make_910c_match(k=12800, dtype=torch.bfloat16):
    return SimpleNamespace(
        kwargs={
            "x": _match_tensor((1, k), dtype),
            "weight": _match_tensor((5120, k), dtype),
            "residual": _match_tensor((1, 5120), dtype),
            "rms_norm_weight": _match_tensor((5120,), dtype),
        }
    )


def test_910c_extra_check_only_accepts_bf16_down_proj():
    assert _is_supported_910c_down_proj_match(_make_910c_match())
    assert not _is_supported_910c_down_proj_match(_make_910c_match(k=4096))
    assert not _is_supported_910c_down_proj_match(_make_910c_match(dtype=torch.float16))


def test_910c_pass_only_runs_for_supported_decode_range():
    fusion_pass = object.__new__(MatmulAllReduceAddRMSNormPass)
    fusion_pass.use_910c_op = True

    assert fusion_pass.is_applicable_for_range(Range(1, 2))
    assert not fusion_pass.is_applicable_for_range(Range(3, 2048))


def test_910c_config_adds_decode_compile_range(monkeypatch):
    compilation_config = SimpleNamespace(compile_ranges_endpoints=[2048], compile_sizes=[])
    vllm_config = SimpleNamespace(compilation_config=compilation_config, additional_config={})
    ascend_config = object.__new__(AscendConfig)
    ascend_config.vllm_config = vllm_config
    ascend_config.ascend_compilation_config = SimpleNamespace(enable_npugraph_ex=True, fuse_allreduce_rms=True)
    monkeypatch.setattr(allreduce_rmsnorm_fusion_pass, "should_use_910c_op", lambda _: True)

    ascend_config.update_compile_ranges_split_points()

    assert compilation_config.compile_ranges_endpoints == [2, 2048]
