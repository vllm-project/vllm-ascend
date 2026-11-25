# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field

from tilelang import tvm
from tilelang.engine.lower import (
    device_codegen,
    device_codegen_without_compile,
    extrac_params,
    host_codegen,
    lower_to_host_device_ir,
)
from tilelang.engine.param import CompiledArtifact
from tilelang.jit.kernel import JITKernel
from tvm import tirx

from ..mtp.native_output import lower_native_weight_copy
from .joint_phase_tir import emit_stage, grid_barrier, rewrite_host


@dataclass
class Stage:
    program: object
    engine: str
    buffers: dict = field(default_factory=dict)
    guard: object = None
    native_weight: bool = False
    independent: bool = False


def cube(program, *, guard=None, nz=False, independent=False, **buffers):
    return Stage(program, "CUBE", buffers, guard, native_weight=nz, independent=independent)


def vector(program, *, guard=None, **buffers):
    return Stage(program, "VECTOR", buffers, guard)


def mixed(program, *, guard=None, **buffers):
    return Stage(program, "mixed", buffers, guard)


def phase(*stages, barrier=None):
    return stages, barrier


def compile_fused(abi, phases, *, name="main_kernel", views=(), controls=None):
    """Fuse each phase on one persistent grid. All workers rendezvous between phases."""
    workers = 28

    class Fused(JITKernel):
        def _compile_artifact(self, func, pass_configs, compile_metadata):
            ctx = self.backend_context
            with self.target, tvm.transform.PassContext(opt_level=3, config=pass_configs):
                host, device, *_ = lower_to_host_device_ir(func, ctx)
                carrier = next(iter(device.functions.values()))
                pointers = {var.name: var for var in carrier.params}
                for base, view in views:
                    ptr = pointers[base] if base in pointers else pointers[view]
                    pointers[base] = pointers[view] = ptr
                worker = tirx.Var("worker", "int32")
                grid = tirx.IterVar(
                    tvm.ir.Range.from_min_extent(0, workers), worker, tirx.IterVar.ThreadIndex, "blockIdx.x"
                )
                statements = controls(pointers) if controls else []
                memory = 0
                for i, (stages, barrier_guard) in enumerate(phases):
                    for spec in stages:
                        _, mod, *_ = lower_to_host_device_ir(spec.program, ctx)
                        fn = next(iter(mod.functions.values()))
                        if spec.native_weight:
                            fn = lower_native_weight_copy(fn)
                        args = {**pointers, **{k: pointers[v] for k, v in spec.buffers.items()}}
                        extent = int(fn.attrs["thread_extent"]["blockIdx.x"])
                        body = emit_stage(
                            fn, args, worker, workers, extent, spec.engine, independent_tiles=spec.independent
                        )
                        statements.append(tirx.IfThenElse(spec.guard, body, None) if spec.guard is not None else body)
                        memory = max(memory, int(fn.attrs.get("dyn_shared_memory_buf", 0)))
                    if i + 1 < len(phases):
                        barrier = grid_barrier(pair_at_entry=False)
                        statements.append(
                            tirx.IfThenElse(barrier_guard, barrier, None) if barrier_guard is not None else barrier
                        )
                body = tirx.AttrStmt(grid, "thread_extent", workers, tirx.SeqStmt(statements))
                extent = int(carrier.attrs["thread_extent"]["blockIdx.x"])
                fn = carrier.with_body(body).with_attr("thread_extent", {"blockIdx.x": tirx.IntImm("int32", workers)})
                fn = fn.with_attr("dyn_shared_memory_buf", tirx.IntImm("int64", memory)).with_attr(
                    "global_symbol", name
                )
                dev = tvm.IRModule({name: fn})
                host = rewrite_host(
                    host, extent, workers, int(carrier.attrs["dyn_shared_memory_buf"]), memory, kernel_name=name
                )
                code = device_codegen_without_compile(dev, ctx).inspect_source()
                assert code.count('extern "C" __global__') == 1
                ready = host_codegen(host, ctx)
                ready.import_module(device_codegen(dev, ctx))
                return CompiledArtifact(
                    ready,
                    dev,
                    extrac_params(func),
                    code,
                    rt_mod=ready,
                    target=self.target,
                    target_host=self.target_host,
                )

    return Fused(func=abi, target="ascend", execution_backend="tvm_ffi")
