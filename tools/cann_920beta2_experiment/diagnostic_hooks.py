"""实验用调用记录：保留最后一次真实 V2 调用的元数据，不读取张量值。"""

import faulthandler
import functools
import json
import os
import threading
import time
import traceback
from pathlib import Path


def install_trace(directory: Path) -> None:
    import torch
    from torch._subclasses.fake_tensor import FakeTensor

    directory.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    fault_log = (directory / f"python-fault-{pid}.log").open("w")

    def describe(value):
        if isinstance(value, torch.Tensor):
            return {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "offset": value.storage_offset(),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "requires_grad": value.requires_grad,
                "contiguous": value.is_contiguous(),
            }
        if isinstance(value, (tuple, list)):
            return [describe(item) for item in value]
        if isinstance(value, dict):
            return {key: describe(item) for key, item in value.items()}
        return value

    def wrap(original):
        @functools.wraps(original)
        def traced(operator, *args, **kwargs):
            if not str(operator).startswith("npu.npu_add_rms_norm_dynamic_quant"):
                return original(operator, *args, **kwargs)
            if any(isinstance(arg, FakeTensor) for arg in args):
                return original(operator, *args, **kwargs)
            # Re-enable after the backend's initialization of native signal handlers.
            faulthandler.enable(file=fault_log, all_threads=True)
            record = {
                "pid": pid,
                "thread": threading.get_ident(),
                "op": str(operator),
                "args": describe(args),
                "kwargs": describe(kwargs),
                "grad_enabled": torch.is_grad_enabled(),
                "inference_mode": torch.is_inference_mode_enabled(),
                "stack": traceback.format_stack(limit=40),
            }
            (directory / f"last-call-{pid}.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
            maps = Path("/proc/self/maps").read_text()
            (directory / f"loaded-libraries-{pid}.txt").write_text(
                "\n".join(line for line in maps.splitlines() if ".so" in line) + "\n"
            )
            result = original(operator, *args, **kwargs)
            (directory / f"last-success-{pid}.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
            return result

        return traced

    torch._ops.OpOverload.__call__ = wrap(torch._ops.OpOverload.__call__)
    torch._ops.OpOverloadPacket.__call__ = wrap(torch._ops.OpOverloadPacket.__call__)


def trace_fusion_config(directory: Path) -> None:
    from vllm_ascend.compilation.graph_fusion_pass_manager import GraphFusionPassManager

    original = GraphFusionPassManager.configure

    @functools.wraps(original)
    def configure(manager, config):
        result = original(manager, config)
        requested = config.additional_config.get("ascend_compilation_config", {}).get("fuse_norm_quant", True)
        actual = manager.ascend_compilation_config.fuse_norm_quant
        passes = [type(item).__name__ for item in manager.passes]
        record = {"requested_fuse_norm_quant": requested, "actual_fuse_norm_quant": actual, "passes": passes}
        (directory / f"fusion-config-{os.getpid()}.json").write_text(json.dumps(record, indent=2) + "\n")
        print("ARDQ_DIAGNOSTIC_CONFIG " + json.dumps(record), flush=True)
        assert actual == requested
        assert requested or "AddRMSNormQuantFusionPass" not in passes
        return result

    GraphFusionPassManager.configure = configure
    original_call = GraphFusionPassManager.__call__

    @functools.wraps(original_call)
    def apply_passes(manager, graph):
        result = original_call(manager, graph)
        (directory / f"graph-{os.getpid()}-{time.time_ns()}.py").write_text(graph.code)
        return result

    GraphFusionPassManager.__call__ = apply_passes
