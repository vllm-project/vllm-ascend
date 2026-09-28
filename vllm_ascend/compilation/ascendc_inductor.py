# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Activate inductor_npu_ext (AscendC fusion) for torch._inductor on NPU.
#
# Two execution styles are supported:
#   * enable_ascendc_inductor_backend(): stock inductor pipeline with the
#     device "npu" codegen overridden to inductor_npu_ext NPUScheduling, which
#     emits and JIT-builds fused AscendC kernels directly.
#   * enable_inductor_fxrt_fx_wrapper(): inductor still performs lowering and
#     AscendC fusion, but the wrapper codegen is replaced by fxrt's
#     FxrtFxWrapper. It re-emits the lowered program as a host torch.fx graph
#     (fused regions survive as compiled-kernel HOP nodes) and routes that graph
#     to the fxrt runtime for execution.

from vllm.logger import init_logger

logger = init_logger(__name__)

_ASCENDC_ACTIVATED = False
_INDUCTOR_FXRT_ACTIVATED = False


def _install_aclgraph_update_plan_shim() -> None:
    """Provide a no-op ``torch_npu._inductor._aclgraph_update_plan``.

    inductor_npu_ext imports the aclgraph static-capture update-plan helpers
    from a newer torch_npu. They are only exercised when inductor cudagraphs
    are enabled (``config.triton.cudagraphs`` / ``cudagraph_trees``); the
    prefill STOCK_TORCH_COMPILE path runs with cudagraph_mode NONE, so a
    no-op shim is functionally equivalent on this torch_npu version.
    """
    import importlib
    import sys
    import types

    try:
        import torch_npu._inductor._aclgraph_update_plan  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    inductor_pkg = importlib.import_module("torch_npu._inductor")
    shim = types.ModuleType("torch_npu._inductor._aclgraph_update_plan")

    def _append_for_codegen_node(wrapper, node):
        return None

    def _emit_for_wrapper(wrapper, result, is_graph_partition_subgraph):
        return None

    shim.append_inductor_aclgraph_update_plan_for_codegen_node = _append_for_codegen_node
    shim.emit_inductor_aclgraph_update_plan_for_wrapper = _emit_for_wrapper
    shim.ACLGRAPH_UPDATE_PLAN_GLOBAL = "__aclgraph_update_plan__"

    sys.modules["torch_npu._inductor._aclgraph_update_plan"] = shim
    setattr(inductor_pkg, "_aclgraph_update_plan", shim)
    logger.info(
        "torch_npu._inductor._aclgraph_update_plan unavailable in this "
        "torch_npu version; installed no-op shim (aclgraph cudagraphs disabled)."
    )


def enable_ascendc_inductor_backend() -> None:
    """Override the inductor "npu" backend with inductor_npu_ext AscendC.

    Idempotent. Must run in each worker process before torch.compile lowers an
    FX graph to inductor.
    """
    global _ASCENDC_ACTIVATED
    if _ASCENDC_ACTIVATED:
        return

    import torch
    from torch._inductor.codegen import common as inductor_common

    # 1. Load torch_npu's inductor integration first. This registers the
    #    Triton-on-NPU backend plus the NPU lowering/fallback/decomposition
    #    patches that the AscendC extension builds on.
    try:
        from torch_npu.utils._dynamo import register_inductor_npu

        register_inductor_npu()
    except Exception:  # noqa: BLE001
        import torch_npu._inductor  # noqa: F401

    # 2. Compatibility shim for newer-torch_npu-only module.
    _install_aclgraph_update_plan_shim()

    # 3. Importing inductor_npu_ext re-registers device "npu" with the
    #    AscendC NPUScheduling / NpuWrapperCodeGen (last registration wins).
    import inductor_npu_ext  # noqa: F401

    device_codegen = inductor_common.device_codegens.get("npu")
    scheduling_name = getattr(device_codegen.scheduling, "__name__", None)
    logger.info(
        "inductor AscendC backend active; device 'npu' scheduling = %s",
        scheduling_name,
    )

    _ASCENDC_ACTIVATED = True


def _keep_npu_out_of_gpu_types_without_triton() -> None:
    """Drop the "npu" inductor GPU type when no usable Triton is present.

    Loading torch_npu's inductor integration appends "npu" to inductor's
    GPU_TYPES, after which the scheduler demands a working Triton for every npu
    graph. The AscendC fusion backend generates no Triton code, so without a
    usable Triton the device must stay out of that list.
    """
    try:
        from torch._inductor import utils as inductor_utils
        from torch.utils._triton import has_triton

        if not has_triton() and "npu" in inductor_utils.GPU_TYPES:
            inductor_utils.GPU_TYPES.remove("npu")
    except Exception:  # noqa: BLE001
        logger.debug("GPU_TYPES/triton probe failed", exc_info=True)


def _patch_void_extern_unbacked_symbol_defs() -> None:
    """Tolerate void/NoneLayout extern returns carrying unbacked-symbol defs.

    A functionalized in-place custom op (e.g. ``torch.ops.vllm.dsa_forward``,
    which returns ``None`` and mutates an ``output`` buffer) is lowered by
    inductor to a :class:`FallbackKernel` whose own result is an unused
    ``NoneLayout`` buffer ("bufN"). Inductor still appends an
    ``UnbackedSymbolDefsLine`` keyed by that result name even when the kernel
    binds no unbacked symbol. The stock ``FxConverter`` dereferences
    ``buffer_to_node[output_name]`` before checking whether any binding needs a
    node, raising ``KeyError: 'bufN'`` because no FX node was created for a void
    return. Skip the line when the root buffer is absent and no pending symbol
    needs it; keep the original behavior (and a clear error) otherwise.
    """
    from fxrt.torch.fx_wrapper import CompiledKernelFxConverter

    patch_attr = "_vllm_ascend_void_extern_patch"
    if getattr(CompiledKernelFxConverter, patch_attr, False):
        return

    original = CompiledKernelFxConverter._generate_unbacked_symbol_defs

    def _generate_unbacked_symbol_defs(self, line) -> None:  # type: ignore[no-untyped-def]
        if line.output_name not in self.buffer_to_node:
            pending = [
                symbol
                for symbol in (line.unbacked_bindings or {})
                if symbol.name not in self.buffer_to_node
            ]
            if not pending:
                return
            raise NotImplementedError(
                "fx_wrapper cannot bind unbacked symbols "
                f"{[str(s) for s in pending]} from void extern output "
                f"'{line.output_name}': no host FX node exists for this "
                "NoneLayout return value."
            )
        return original(self, line)

    CompiledKernelFxConverter._generate_unbacked_symbol_defs = (
        _generate_unbacked_symbol_defs
    )
    setattr(CompiledKernelFxConverter, patch_attr, True)


def _patch_extern_kernel_unflattened_kwargs() -> None:
    """Preserve reconstructed keyword args in fx_wrapper extern conversion.

    ``ExternKernel.unflatten_args`` returns ``(args, kwargs)`` rebuilt from the
    original operator call's pytree spec. Both stock ``FxConverter`` and fxrt's
    override discard the returned mapping (``args, _ = ...``) and only forward
    ``kernel.kwargs``. Ordinary fallback kernels put every constant in
    ``kernel.kwargs`` or positionally, so this usually goes unnoticed -- but an
    explicit allocation kept as an extern fallback, e.g.
    ``aten.empty.memory_format`` (the output buffer allocated inside the
    compiled region for a functionalized in-place custom op such as
    ``vllm.dsa_forward``), carries ``dtype``/``device``/``pin_memory`` in the
    reconstructed kwargs. Dropping them makes the host FX node default to a
    CPU float32 tensor; that fake device then propagates into downstream ops
    ("Unhandled FakeTensor Device Propagation ... found two different devices
    cpu, npu:0"). Merge the reconstructed kwargs back, letting ``kernel.kwargs``
    (and the explicit ``out=`` buffer) take precedence.
    """
    from fxrt.torch.fx_wrapper import CompiledKernelFxConverter
    from torch._inductor import ir

    patch_attr = "_vllm_ascend_unflatten_kwargs_patch"
    if getattr(CompiledKernelFxConverter, patch_attr, False):
        return

    def _generate_extern_kernel_common(self, kernel, out_ir_node) -> None:  # type: ignore[no-untyped-def]
        assert ir.is_node_sequence(kernel.inputs)
        tensor_nodes = tuple(self._generate_buffer(arg) for arg in kernel.inputs)
        reconstructed_kwargs = {}
        if hasattr(kernel, "unflatten_args"):
            args, reconstructed_kwargs = kernel.unflatten_args(
                tensor_nodes, kernel.constant_args
            )
        else:
            args = tensor_nodes + tuple(kernel.constant_args)
        args = self._lift_sym_args(args)

        def _as_node(value):  # type: ignore[no-untyped-def]
            return (
                self._generate_buffer(value)
                if isinstance(value, ir.IRNode)
                else value
            )

        kwargs = {key: _as_node(value) for key, value in kernel.kwargs.items()}
        for key, value in reconstructed_kwargs.items():
            kwargs.setdefault(key, _as_node(value))
        kwargs = self._lift_sym_args(kwargs)

        result_buffer = None
        if isinstance(kernel, ir.ExternKernelOut):
            kwargs["out"] = self.buffer_to_node[out_ir_node.codegen_reference()]
        elif isinstance(kernel.layout, (ir.Layout, ir.MultiOutputLayout)):
            result_buffer = kernel.get_name()
        elif isinstance(kernel.layout, ir.NoneLayout):
            pass
        else:
            raise NotImplementedError(
                f"Unrecognized output layout: {kernel.layout}"
            )

        fx_node = self.gm.graph.call_function(
            kernel.op_overload,
            args=args,
            kwargs=kwargs,
        )

        if result_buffer:
            fx_node.name = result_buffer
            self.buffer_to_node[result_buffer] = fx_node

    CompiledKernelFxConverter._generate_extern_kernel_common = (
        _generate_extern_kernel_common
    )
    setattr(CompiledKernelFxConverter, patch_attr, True)


def enable_inductor_fxrt_fx_wrapper() -> None:
    """Route inductor output back to fxrt through the fx_wrapper codegen.

    Inductor keeps doing decomposition/fusion with the inductor_npu_ext AscendC
    scheduling backend; the fxrt FxrtFxWrapper then remaps the lowered program
    to a host FX graph and executes it with the fxrt runtime. Idempotent; must
    run in each worker process before torch.compile lowers an FX graph.
    """
    global _INDUCTOR_FXRT_ACTIVATED
    if _INDUCTOR_FXRT_ACTIVATED:
        return

    from torch._inductor.codegen import common as inductor_common

    # torch_npu lazily re-registers its Triton-on-NPU codegen the first time an
    # inductor backend is built, which would clobber the AscendC scheduling
    # backend installed below; turn that lazy registration off first.
    try:
        from torch_npu.utils._dynamo import disable_register_inductor_npu

        disable_register_inductor_npu()
    except Exception:  # noqa: BLE001
        logger.debug("could not disable torch_npu lazy inductor registration", exc_info=True)

    # Compatibility shim (also pulls torch_npu._inductor in for its lowering
    # patches) and the AscendC fusion backend (last registration owns "npu").
    _install_aclgraph_update_plan_shim()
    import inductor_npu_ext  # noqa: F401

    _keep_npu_out_of_gpu_types_without_triton()

    # Replace only the fx_wrapper codegen slot; NPUScheduling is retained.
    from fxrt.torch import fx_wrapper

    installed = fx_wrapper.install_torch_npu_fx_wrapper()
    if not installed:
        raise RuntimeError("Failed to install the fxrt FxrtFxWrapper for npu")

    # Stock FxConverter mishandles void/NoneLayout extern returns (functionalized
    # in-place custom ops such as vllm.dsa_forward) that still carry an
    # unbacked-symbol defs line; guard that before graphs are lowered.
    _patch_void_extern_unbacked_symbol_defs()
    # ...and it drops the kwargs that unflatten_args reconstructs for extern
    # calls (notably aten.empty.memory_format's dtype/device), which would put
    # those tensors on CPU in the host FX graph.
    _patch_extern_kernel_unflattened_kwargs()

    # The fx_wrapper codegen is selected for a graph only when this is set.
    import torch._inductor.config as inductor_config

    inductor_config.fx_wrapper = True
    inductor_config.size_asserts = False
    inductor_config.alignment_asserts = False
    # inductor_npu_ext JIT-builds each fused AscendC kernel in inductor's async
    # compile pool. Its default ("subprocess"/fork) workers inherit an already
    # initialized NPU context, which torch_npu rejects ("Cannot re-initialize
    # NPU in forked subprocess"). Fresh "spawn" workers import torch_npu cleanly
    # and only drive the offline AscendC compiler (the device properties they
    # need are passed in from the parent, not queried in the child).
    inductor_config.worker_start_method = "spawn"

    device_codegen = inductor_common.device_codegens.get("npu")
    logger.info(
        "inductor->fx_wrapper->fxrt active; scheduling = %s, fx_wrapper = %s",
        getattr(device_codegen.scheduling, "__name__", None),
        getattr(device_codegen.fx_wrapper_codegen, "__name__", None),
    )

    _INDUCTOR_FXRT_ACTIVATED = True
