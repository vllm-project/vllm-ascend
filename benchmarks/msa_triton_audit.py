# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit, reversible diagnostics for M3 Triton shape/compilation evidence.

Install one instance in each model worker before warmup. Mark inference after
capture and close after the accuracy workload. Never enable in a clean perf run.
Only tensor metadata is inspected; no values, casts or synchronizations occur.
TorchDynamo tracing bypasses observation; compiled replay can bypass Python.
"""

import hashlib
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path


class TritonAudit:
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"triton-{os.getpid()}-{time.time_ns()}.jsonl"
        self.file = self.path.open("x", buffering=1)
        self.phase = "warmup"
        self.lock = threading.RLock()
        self.local = threading.local()
        self.counts = Counter()
        self.shapes = set()
        self.binaries = set()
        self.restorations = []
        self.installed = False

    def emit(self, event, **data):
        with self.lock:
            self.file.write(
                json.dumps(dict(event=event, pid=os.getpid(), time=time.time(), phase=self.phase, **data), default=str)
                + "\n"
            )

    def mark_phase(self, phase):
        self.phase = phase
        self.emit("phase")

    @staticmethod
    def describe(value):
        if hasattr(value, "shape") and callable(getattr(value, "stride", None)):
            return dict(
                shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype), device=str(value.device)
            )
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return dict(type=type(value).__name__)

    @staticmethod
    def relevant(name):
        return "msa_m3_index_postprocess" in name or "msa_m3_triton_a5" in name

    def replace(self, owner, name, replacement):
        old = getattr(owner, name)
        self.restorations.append((owner, name, old, replacement))
        setattr(owner, name, replacement)
        return old

    def install(self, jit_type=None, hook_owner=None, pre_name=None, post_name=None):
        if self.installed:
            raise RuntimeError("Audit is already installed")
        if jit_type is None:
            from triton.runtime.jit import JITFunction

            jit_type = JITFunction
        if hook_owner is None:
            try:
                from triton import knobs
            except ImportError:
                knobs = None
            if knobs is not None and hasattr(knobs.runtime, "jit_cache_hook"):
                hook_owner, pre_name, post_name = knobs.runtime, "jit_cache_hook", "jit_post_compile_hook"
            elif hasattr(jit_type, "cache_hook"):
                hook_owner, pre_name = jit_type, "cache_hook"
                post_name = "compiled_hook" if hasattr(jit_type, "compiled_hook") else None
            else:
                raise RuntimeError("Installed Triton has no supported JIT cache hook; do not report zero compiles")
        if not pre_name or not hasattr(hook_owner, pre_name):
            raise RuntimeError("Missing compile/cache-miss hook")
        # Resolve this worker-local guard once; standalone CPU diagnostics can
        # still install synthetic JIT hooks without requiring Torch.
        try:
            from torch.compiler import is_compiling
        except ImportError:

            def is_compiling():
                return False

        original_run = jit_type.run
        audit = self

        def run(jit, *args, **kwargs):
            # Keep metadata inspection, locking and IO out of fullgraph traces.
            if is_compiling():
                return original_run(jit, *args, **kwargs)
            function = jit.fn
            name = function.__module__ + "." + function.__name__
            if not audit.relevant(name):
                return original_run(jit, *args, **kwargs)
            bound = dict(zip(jit.arg_names, args))
            bound.update(kwargs)
            signature = {key: audit.describe(value) for key, value in bound.items()}
            key = hashlib.sha256(json.dumps(signature, sort_keys=True, default=str).encode()).hexdigest()
            with audit.lock:
                audit.counts[(audit.phase, name, key)] += 1
                if (name, key) not in audit.shapes:
                    audit.shapes.add((name, key))
                    audit.emit("shape", kernel=name, signature_key=key, arguments=signature)
            previous = getattr(audit.local, "kernel", None)
            audit.local.kernel = name
            try:
                result = original_run(jit, *args, **kwargs)
                binary_hash = getattr(result, "hash", None)
                if isinstance(binary_hash, str):
                    with audit.lock:
                        if (name, binary_hash) not in audit.binaries:
                            audit.binaries.add((name, binary_hash))
                            audit.emit("binary_seen", kernel=name, binary_hash=binary_hash)
                return result
            finally:
                audit.local.kernel = previous

        self.replace(jit_type, "run", run)

        def install_hook(attribute, event):
            previous = getattr(hook_owner, attribute)

            def hook(*args, **kwargs):
                # Preserve existing instrumentation, including its skip return.
                result = previous(*args, **kwargs) if previous is not None else None
                info = kwargs.get("fn")
                kernel = getattr(audit.local, "kernel", None) or str(info)
                if audit.relevant(kernel):
                    audit.emit(
                        event,
                        kernel=kernel,
                        key=str(kwargs.get("key")),
                        already_compiled=kwargs.get("already_compiled"),
                        external_hook_return=result if isinstance(result, (bool, type(None))) else str(type(result)),
                    )
                return result

            audit.replace(hook_owner, attribute, hook)

        install_hook(pre_name, "jit_cache_miss")
        if post_name and hasattr(hook_owner, post_name):
            install_hook(post_name, "compile_or_load_completed")
        self.installed = True
        self.emit(
            "installed",
            hook_api=pre_name,
            post_hook_api=post_name,
            note="JIT misses/loads are not automatically backend code generation; use a fresh private cold cache",
        )
        return self

    def close(self):
        if self.file.closed:
            return
        conflicts = []
        for owner, name, old, replacement in reversed(self.restorations):
            if getattr(owner, name) is replacement:
                setattr(owner, name, old)
            else:
                conflicts.append(name)  # Do not overwrite somebody else's hook.
        self.emit(
            "summary",
            unique_shapes=len(self.shapes),
            unique_binaries=len(self.binaries),
            calls=[dict(phase=p, kernel=n, signature_key=s, count=c) for (p, n, s), c in self.counts.items()],
            restoration_conflicts=conflicts,
        )
        self.file.close()
        self.installed = False

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.close()
