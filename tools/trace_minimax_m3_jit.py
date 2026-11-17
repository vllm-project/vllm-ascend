# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only JIT tracing; install in every model worker with sitecustomize.

Call install(output_dir, phase_file) from a task-local sitecustomize.py before
model imports. Put its directory and this repository on PYTHONPATH. Set the phase
file to startup, warmup, capture or serving as the workload advances. Use a fresh
TRITON_CACHE_DIR for cold runs. Never use traced runs for clean latency results.
"""

import functools
import hashlib
import json
import os
import threading
import time
from pathlib import Path


def install(output_dir: str, phase_file: str) -> None:
    import triton.compiler.compiler as compiler
    from triton.runtime.jit import JITFunction

    if getattr(JITFunction, "_minimax_trace_installed", False):
        return
    if not hasattr(JITFunction, "_do_compile") or not hasattr(compiler.ASTSource, "make_ir"):
        raise RuntimeError("Inspect this Triton version before installing MiniMax JIT trace hooks")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    phase_path = Path(phase_file)
    lock = threading.Lock()
    seen = set()

    def selected(kernel):
        module = getattr(getattr(kernel, "fn", None), "__module__", "")
        return module.startswith("vllm_ascend.") and any(
            part in module for part in ("indexer_dcp", "minimax_qknorm_rope_cache", "msa_m3_index_postprocess")
        )

    def emit(event, kernel, **data):
        phase = phase_path.read_text().strip() if phase_path.exists() else "startup"
        record = dict(
            time_ns=time.time_ns(),
            pid=os.getpid(),
            phase=phase,
            event=event,
            kernel=kernel.fn.__module__ + "." + kernel.fn.__name__,
            **data,
        )
        with lock, (directory / f"jit-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(record, default=str, sort_keys=True) + "\n")

    def describe(value):
        if hasattr(value, "shape") and hasattr(value, "stride") and hasattr(value, "dtype"):
            return dict(
                shape=list(value.shape),
                stride=list(value.stride()),
                dtype=str(value.dtype),
                alignment_mod_16=value.data_ptr() % 16,
            )
        return value if isinstance(value, (int, float, bool, str, type(None))) else str(value)

    original_run = JITFunction.run
    original_resolve = JITFunction._do_compile
    original_make_ir = compiler.ASTSource.make_ir

    @functools.wraps(original_run)
    def run(kernel, *args, **kwargs):
        if selected(kernel):
            parameters = {name: describe(value) for name, value in zip(kernel.arg_names, args)}
            parameters.update({name: describe(kwargs[name]) for name in kernel.arg_names if name in kwargs})
            options = {name: describe(value) for name, value in kwargs.items() if name not in kernel.arg_names}
            phase = phase_path.read_text().strip() if phase_path.exists() else "startup"
            key = (os.getpid(), phase, kernel.fn.__module__, kernel.fn.__name__, json.dumps(parameters, sort_keys=True))
            if key not in seen:
                seen.add(key)
                emit(
                    "launch_metadata",
                    kernel,
                    parameters=parameters,
                    options=options,
                    constexprs=kernel.constexprs,
                    do_not_specialize=kernel.do_not_specialize,
                )
        return original_run(kernel, *args, **kwargs)

    @functools.wraps(original_resolve)
    def resolve(kernel, *args, **kwargs):
        if not selected(kernel):
            return original_resolve(kernel, *args, **kwargs)
        key = args[0] if args else kwargs.get("key")
        digest = hashlib.sha256(repr(key).encode()).hexdigest()
        emit("jit_resolution_begin", kernel, key=digest)
        result = original_resolve(kernel, *args, **kwargs)
        emit("jit_resolution_complete", kernel, key=digest)
        return result

    @functools.wraps(original_make_ir)
    def make_ir(source, *args, **kwargs):
        if not selected(source.fn):
            return original_make_ir(source, *args, **kwargs)
        emit("frontend_compile_begin", source.fn)
        result = original_make_ir(source, *args, **kwargs)
        emit("frontend_compile_complete", source.fn)
        return result

    JITFunction.run = run
    JITFunction._do_compile = resolve
    compiler.ASTSource.make_ir = make_ir
    JITFunction._minimax_trace_installed = True
