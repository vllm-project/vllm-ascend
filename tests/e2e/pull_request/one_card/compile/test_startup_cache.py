# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Fresh-process validation of fusion caching on one NPU."""

import json
import os
import subprocess
import sys
from pathlib import Path


def run_worker(mode, root):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), mode, str(root)],
        env=dict(os.environ, OMP_NUM_THREADS="1", TRITON_CACHE_DIR=str(root / "triton")),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((root / (mode + ".json")).read_text())


def test_real_fusion_cache_across_processes(tmp_path):
    cold = run_worker("miss", tmp_path)
    warm = run_worker("hit", tmp_path)
    changed = run_worker("changed", tmp_path)
    assert cold["compilations"] == changed["compilations"] == 1
    assert warm["compilations"] == 0
    assert cold["saved"] > 0 and warm["hits"] > 0
    assert cold["fingerprint"] == warm["fingerprint"] != changed["fingerprint"]


def fusion_worker(mode, root):
    import shutil
    from types import SimpleNamespace

    import torch
    import torch_npu  # noqa: F401
    from torch._dynamo.utils import counters
    from vllm.compilation.passes.inductor_pass import pass_context
    from vllm.config import VllmConfig
    from vllm.config.utils import Range

    import vllm_ascend.ops.register_custom_ops  # noqa: F401
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.compilation import compiler_interface as interface
    from vllm_ascend.compilation.graph_fusion_pass_manager import GraphFusionPassManager
    from vllm_ascend.compilation.passes.muls_add_pass import MulsAddFusionPass
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    from vllm_ascend.utils import COMPILATION_PASS_KEY

    init_device_properties_triton()
    config = VllmConfig()
    config.additional_config = {"ascend_compilation_config": {"enable_npugraph_ex": False}}
    init_ascend_config(config)
    config.model_config = SimpleNamespace(
        dtype=torch.float32, hf_text_config=SimpleNamespace(routed_scaling_factor=0.5)
    )
    # Simulate a changed pass installation without touching installed sources.
    if mode == "changed":
        source = Path(interface.__file__).parent
        copied = root / "modified" / "compilation"
        for relative in interface.FUSION_CACHE_SOURCE_FILES:
            target = copied / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
        path = copied / "passes/muls_add_pass.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "passes/muls_add_pass.py", path)
        path.write_text(path.read_text() + "\n# changed pass implementation\n")
        interface.__file__ = str(copied / "compiler_interface.py")

    compiler = interface.AscendCompiler()
    fingerprint = compiler.compute_hash(config)
    compiler.initialize_cache(str(root / fingerprint), disable_cache=False)
    manager = GraphFusionPassManager()
    fusion = MulsAddFusionPass(config)
    manager.add(fusion)
    compilations = 0

    def transform(graph):
        nonlocal compilations
        assert mode != "hit", "Warm process must deserialize without rerunning fusion"
        compilations += 1
        with pass_context(Range(1, 8)):
            graph = manager(graph)
        assert fusion.matched_count > 0
        assert any(node.target == torch.ops.vllm.muls_add.default for node in graph.graph.nodes)
        return graph

    def backend(graph, inputs):
        return compiler.compile(graph, inputs, {COMPILATION_PASS_KEY: transform}, Range(1, 8))[0]

    def model(x, y):
        output = x * 0.5 + y
        x.add_(1)
        return output, x, x.view(-1)

    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=True)
    torch.manual_seed(445)
    with torch.no_grad():
        for rows in (2, 5):
            x = torch.randn(rows, 2048, device="npu")
            y = torch.randn_like(x)
            expected, mutated, view = model(x.clone(), y)
            actual, actual_mutated, actual_view = compiled(x, y)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(actual_mutated, mutated)
            torch.testing.assert_close(actual_view, view)
            torch.testing.assert_close(x, mutated)
            assert actual_mutated.data_ptr() == actual_view.data_ptr() == x.data_ptr()
    return {
        "compilations": compilations,
        "fingerprint": fingerprint,
        "hits": counters["aot_autograd"]["autograd_cache_hit"],
        "saved": counters["aot_autograd"]["autograd_cache_saved"],
    }


if __name__ == "__main__":
    worker_mode, directory = sys.argv[1:]
    output_dir = Path(directory)
    result = fusion_worker(worker_mode, output_dir)
    (output_dir / (worker_mode + ".json")).write_text(json.dumps(result))
