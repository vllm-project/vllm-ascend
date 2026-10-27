# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from functools import wraps
from types import FunctionType
from typing import Any

from torch_npu._inductor.npu_triton_heuristics import TritonCompileResultNpu, triton_version_uses_attrs_dict


def _bind_launcher_constants(make_launcher):
    # Reuse torch_npu's existing compact-ABI branch without changing its module
    # global (which is also used by concurrent/full-signature compilations).
    compact_make_launcher = FunctionType(
        make_launcher.__code__,
        {**make_launcher.__globals__, "triton_version_uses_attrs_dict": lambda: False},
        make_launcher.__name__,
        make_launcher.__defaults__,
        make_launcher.__closure__,
    )

    @wraps(make_launcher)
    def make_launcher_with_constants(self):
        if not triton_version_uses_attrs_dict():
            return make_launcher(self)

        fn = self.kernel.src.fn
        constexpr_names = {fn.arg_names[i] for i in fn.constexprs}
        # NPULauncher builds its C argument parser from src.signature. Inductor
        # may omit all constexprs there even with Triton's attrs-dict API.
        if constexpr_names and constexpr_names.isdisjoint(self.kernel.src.signature):
            return compact_make_launcher(self)

        launcher = make_launcher(self)
        parameters = inspect.signature(launcher).parameters
        constants = {
            fn.arg_names[i]: self.compile_meta["constants"][fn.arg_names[i]]
            for i in fn.constexprs
            if fn.arg_names[i] in self.compile_meta["constants"] and fn.arg_names[i] in parameters
        }
        if not constants:
            return launcher  # Also a no-op once torch_npu binds constexprs itself.

        # Keep the original NPU grid and launch ABI, supplying only omitted
        # compile-time parameters. Generate once, not by binding args per launch.
        def_args = [name for name in parameters if name not in constants]
        call_args = [f"_constants[{name!r}]" if name in constants else name for name in parameters]
        scope: dict[str, Any] = {"_launcher": launcher, "_constants": constants}
        exec(f"def launcher({', '.join(def_args)}):\n    return _launcher({', '.join(call_args)})", scope)
        bound_launcher = scope["launcher"]
        bound_launcher.__dict__.update(launcher.__dict__)
        return bound_launcher

    return make_launcher_with_constants


# Remove when supported torch_npu launchers handle constexprs according to the
# compiled source signature rather than inferring the launch ABI from attrs.
TritonCompileResultNpu.make_launcher = _bind_launcher_constants(TritonCompileResultNpu.make_launcher)
