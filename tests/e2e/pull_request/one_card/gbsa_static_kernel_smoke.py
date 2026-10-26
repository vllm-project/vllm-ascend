# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Run an isolated bundled-GBSA static-kernel blacklist smoke in a fresh process."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-len", type=int, choices=[1, 4], default=1)
    parser.add_argument("--super-kernel", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)
    bootstrap_custom_op_env(include_vendor_lib=True)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)

    import npugraph_ex as nge
    from npugraph_ex.configs.compiler_config import _process_kwargs_options

    from tests.e2e.pull_request.one_card.test_generic_block_sparse_attention import _make_case, _run_case
    from vllm_ascend.compilation.compiler_interface import _configure_backend

    if dist.is_initialized():
        raise RuntimeError("Run the GBSA static-kernel smoke in a fresh process")
    config = nge.CompilerConfig()
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(local_world_size=1, data_parallel_size_local=1),
        speculative_config=SimpleNamespace(num_speculative_tokens=args.query_len - 1),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 2, 4, 8]),
    )
    _configure_backend(
        config,
        SimpleNamespace(enable_static_kernel=True, enable_super_kernel=args.super_kernel),
        vllm_config,
        _process_kwargs_options,
    )
    # This isolated backend owns capture; vLLM normally supplies an outer graph.
    config.force_eager = False
    config.disable_static_kernel_compile_cache = True
    inputs, reference = _make_case(torch.bfloat16, (args.query_len, args.query_len), (257, 383), 1)
    torch.testing.assert_close(_run_case(inputs, args.query_len, 1).float().cpu(), reference, atol=1e-2, rtol=2e-2)

    def attention(*tensors):
        # Keep a vector add eligible for static compilation while GBSA is excluded.
        return (_run_case(tensors, args.query_len, 1) + 1,)

    dist.init_process_group("gloo", init_method=f"file://{output_dir / 'gloo-init'}", rank=0, world_size=1)
    try:
        compiled = torch.compile(
            attention, backend=nge.get_npu_backend(compiler_config=config), fullgraph=True, dynamic=False
        )
        max_errors = []
        for step in range(6):
            if step >= 4:
                inputs[0].mul_(0.75)
            expected = attention(*inputs)[0]
            actual = compiled(*inputs)[0]
            torch.npu.synchronize()
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=2e-2)
            max_errors.append((actual.float() - expected.float()).abs().max().item())
        excluded = sorted(
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*.json")
            if "opcompile_blacklist" in str(path) and "GenericBlockSparseAttention" in path.name
        )
        assert excluded, "No runtime GBSA JSON was observed in the static-kernel blacklist directory"
        result = {
            "status": "passed",
            "query_len": args.query_len,
            "static_kernel": True,
            "force_eager": False,
            "super_kernel": args.super_kernel,
            "max_abs_errors": max_errors,
            "gbsa_blacklisted_runtime_jsons": excluded,
            "scope": "isolated GBSA plus vector add; full model startup requires separate validation",
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
