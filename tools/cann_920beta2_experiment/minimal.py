"""无需模型权重，比较推理模式、最小编译图及 ACL 初始化模式。"""

import json
import sys
from pathlib import Path


def main():
    import torch
    import torch_npu
    from diagnostic_hooks import install_trace

    mode = sys.argv[1]
    evidence = Path(sys.argv[2])
    install_trace(evidence / f"minimal-{mode}-diagnostics")
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(20260907)
    print(torch.__version__, torch_npu.__version__, mode, flush=True)

    def forward(x1, x2, gamma, smooth):
        return torch.ops.npu.npu_add_rms_norm_dynamic_quant(
            x1,
            x2,
            gamma,
            smooth_scale1=None if mode == "compile-no-smooth" else smooth,
            epsilon=1e-6,
            output_mask=[True, False],
        )

    if mode != "eager-inference":
        import npugraph_ex as nge

        try:
            from npugraph_ex.configs.compiler_config import _process_kwargs_options
        except ImportError:
            from npugraph_ex.configs.npugraphex_config import _process_kwargs_options
        config = nge.CompilerConfig()
        _process_kwargs_options(
            config,
            {"options": {"force_eager": True, "inplace_pass": False, "clone_input": False, "clone_output": False}},
        )
        forward = torch.compile(
            forward, backend=nge.get_npu_backend(compiler_config=config), dynamic=True, fullgraph=True
        )
    with torch.inference_mode():
        for rows in (4096, 48, 1):
            x1 = torch.randn(rows, 7168, dtype=torch.bfloat16, device="npu")
            x2 = torch.randn_like(x1)
            gamma = torch.ones(7168, dtype=torch.bfloat16, device="npu")
            smooth = torch.ones_like(gamma)
            print(json.dumps({"stage": "before", "rows": rows}), flush=True)
            result = forward(x1, x2, gamma, smooth)
            torch.npu.synchronize()
            print(json.dumps({"stage": "after", "rows": rows, "outputs": [list(t.shape) for t in result]}), flush=True)


if __name__ == "__main__":
    main()
