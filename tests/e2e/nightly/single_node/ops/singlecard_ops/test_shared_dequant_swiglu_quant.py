# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Run legacy and official kernels in separate processes, never one registry."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

TOKENS = [1, 2, 7, 8, 9, 64]
WIDTHS = [256, 384, 4096, 6144]
LIMITS = [1.0, 2.0, 10.0, 100.0]


@pytest.fixture(scope="module")
def isolated_outputs(tmp_path_factory):
    from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

    if not get_current_hardware_profile().supports(HardwareCapability.SHARED_SWIGLU_ONE_GROUP):
        pytest.skip("The official shared one-group adapter is validated only on A3.")
    legacy_value = os.environ.get("VLLM_ASCEND_LEGACY_ROOT")
    if not legacy_value:
        pytest.skip("Set VLLM_ASCEND_LEGACY_ROOT to an A3-built pre-replacement checkout.")
    root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
    legacy = Path(legacy_value).resolve()
    output = tmp_path_factory.mktemp("shared-swiglu")
    helper = root / "vllm_ascend/ops/fused_moe/shared_dequant_swiglu_quant.py"
    legacy_opp = legacy / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
    assert legacy_opp.is_dir(), f"Missing legacy OPP: {legacy_opp}"
    results = {}
    for side, checkout in [("legacy", legacy), ("official", root), ("conflict", root)]:
        libraries = list((checkout / "vllm_ascend").glob("vllm_ascend_C*.so"))
        assert len(libraries) == 1, f"Expected one rebuilt binding in {checkout}: {libraries}"
        env = os.environ.copy()
        env.pop("ASCEND_CUSTOM_OPP_PATH", None)
        if side != "official":
            env["ASCEND_CUSTOM_OPP_PATH"] = str(legacy_opp)
        destination = output / f"{side}.pt"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--isolated",
                side,
                str(libraries[0]),
                str(helper),
                str(destination),
            ],
            env=env,
            check=True,
            timeout=300,
        )
        results[side] = torch.load(destination, map_location="cpu", weights_only=True)
    return results


@pytest.mark.parametrize("num_tokens", TOKENS)
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("use_cache", [False, True])
def test_shared_clipped_swiglu_legacy_parity(isolated_outputs, num_tokens, width, limit, use_cache):
    key = f"{num_tokens}:{width}:{limit}"
    expected = isolated_outputs["legacy"][key]
    actual = isolated_outputs["official"][f"{key}:{use_cache}"]
    for stage in ["initial", "updated", "graph"]:
        for output, golden in zip(actual[stage], expected["updated" if stage == "graph" else stage]):
            assert torch.equal(output, golden)


def test_shared_clipped_swiglu_rejects_legacy_override(isolated_outputs):
    assert isolated_outputs["conflict"]["rejected"]


@torch.inference_mode()
def _run_isolated(side, library, helper, destination):
    import torch_npu

    torch_npu.npu.config.allow_internal_format = True
    # Loading the binding directly avoids package bootstrap re-adding custom OPP.
    torch.ops.load_library(library)
    spec = importlib.util.spec_from_file_location("shared_swiglu", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if side == "conflict":
        x = torch.ones((1, 256), dtype=torch.int32, device="npu")
        with pytest.raises(RuntimeError, match="cannot coexist with a custom DequantSwigluQuant override"):
            module.shared_dequant_swiglu_quant(x, torch.ones(256, device="npu"), torch.ones(1, device="npu"), 10.0)
        torch.save({"rejected": True}, destination)
        return
    results = {}
    # All fixtures are regenerated identically in the two fresh processes.
    for tokens in TOKENS:
        for width in WIDTHS:
            for limit in LIMITS:
                kinds = [False] if side == "legacy" else [False, True]
                for use_cache in kinds:
                    rng = torch.Generator().manual_seed(20261021)
                    x = torch.randint(-10000, 10000, (tokens, width), dtype=torch.int32, generator=rng).npu()
                    w = (torch.rand(width, generator=rng) * 0.5 + 0.01).npu()
                    act = (torch.rand(tokens, generator=rng) * 0.5 + 0.01).npu()
                    cache = module.SharedSwiGLUCache() if use_cache else None

                    def call(x=x, w=w, act=act, limit=limit, cache=cache):
                        if side == "legacy":
                            return torch.ops._C_ascend.npu_dequant_swiglu_quant(
                                x=x,
                                weight_scale=w,
                                activation_scale=act,
                                group_index=None,
                                activate_left=True,
                                quant_mode=1,
                                swiglu_mode=1,
                                clamp_limit=limit,
                                glu_alpha=1.0,
                                glu_bias=0.0,
                            )
                        return module.shared_dequant_swiglu_quant(x, w, act, limit, cache=cache)

                    initial = tuple(t.cpu() for t in call())
                    if side != "legacy":
                        graph = torch.npu.NPUGraph()
                        with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
                            captured = call()
                    x.neg_()
                    act.mul_(0.5)
                    result = {"initial": initial, "updated": tuple(t.cpu() for t in call())}
                    key = f"{tokens}:{width}:{limit}"
                    if side != "legacy":
                        graph.replay()
                        torch.npu.synchronize()
                        result["graph"] = tuple(t.cpu() for t in captured)
                        key += f":{use_cache}"
                        del graph, captured
                    results[key] = result
    torch.save(results, destination)


if __name__ == "__main__":
    if len(sys.argv) != 6 or sys.argv[1] != "--isolated":
        raise SystemExit("Run with pytest or --isolated SIDE LIBRARY HELPER OUTPUT")
    _run_isolated(*sys.argv[2:])
