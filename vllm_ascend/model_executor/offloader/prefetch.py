"""Ascend prefetch-based CPU offloading with NZ-format static buffers."""

from collections.abc import Generator
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch
import torch.nn as nn
import torch_npu
from vllm.logger import logger
from vllm.model_executor.offloader.prefetch import (
    ParamInfo as VllmParamInfo,
)
from vllm.model_executor.offloader.prefetch import (
    PrefetchOffloader,
    StaticBufferPool,
)
from vllm.model_executor.offloader.prefetch import (
    _ModuleOffloader as VllmModuleOffloader,
)

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

# mark
_ASCEND_PREFETCH_NZ_WEIGHT_ATTR = "_vllm_ascend_prefetch_offload_nz_weight"


def mark_prefetch_offload_nz_weight(param: nn.Parameter) -> None:
    """Mark a parameter whose prefetch static buffer must use NZ format."""
    setattr(param, _ASCEND_PREFETCH_NZ_WEIGHT_ATTR, True)


def _is_using_nz_weight(param: nn.Parameter) -> bool:
    """if its current NPU storage format is FRACTAL_NZ."""
    if param.data.device.type != "npu":
        return False

    try:
        npu_format = int(torch_npu.get_npu_format(param.data))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False

    return npu_format == ACL_FORMAT_FRACTAL_NZ


def _is_prefetch_offload_nz_weight(param: nn.Parameter) -> bool:
    return bool(getattr(param, _ASCEND_PREFETCH_NZ_WEIGHT_ATTR, False))


@dataclass
class ParamInfo(VllmParamInfo):
    """Ascend parameter metadata with static-buffer format requirements."""

    use_nz_buffer: bool = False


def _format_static_buffers_for_nz(
    module_offloaders: list["_ModuleOffloader"],
    buffer_pool: StaticBufferPool,
    param_infos: list[ParamInfo],
) -> None:
    """Cast static buffers to NZ format and re-bind params that require it.

    npu_format_cast returns a NEW tensor, so the NZ buffers stored in the pool
    are not the ones bound by PrefetchOffloader.post_init(): params still point
    at the pre-cast ND buffers, and NZ-only w8a8 kernels
    (e.g. npu_grouped_matmul_swiglu_quant) would read ND storage as NZ.
    Re-bind NZ-marked params to the cast buffers so the subsequent initial
    prefetches fill NZ storage before the first forward.
    """
    key_to_use_nz: dict[tuple[str, tuple[int, ...], tuple[int, ...], torch.dtype], bool] = {}

    for info in param_infos:
        if info.key in key_to_use_nz and key_to_use_nz[info.key] != info.use_nz_buffer:
            raise ValueError(
                "Conflicting NZ buffer requirements for prefetch static buffer "
                f"key {info.key}: both NZ and non-NZ parameters use this key."
            )
        key_to_use_nz[info.key] = info.use_nz_buffer
        if info.use_nz_buffer:
            logger.info("%s enables NZ buffer", info.name)

    for key, use_nz_buffer in key_to_use_nz.items():
        if not use_nz_buffer:
            continue
        buffer_pool._buffers[key] = [
            torch_npu.npu_format_cast(buffer, ACL_FORMAT_FRACTAL_NZ) for buffer in buffer_pool._buffers[key]
        ]

    for offloader in module_offloaders:
        for name, param_offloader in offloader._param_offloaders.items():
            if not _is_prefetch_offload_nz_weight(param_offloader._param):
                continue
            cpu_storage = param_offloader._cpu_storage
            nz_buffer = buffer_pool.get_buffer(
                name=name,
                shape=tuple(cpu_storage.shape),
                stride=tuple(cpu_storage.stride()),
                dtype=cpu_storage.dtype,
                slot_idx=offloader._buffer_slot_idx,
            )
            param_offloader._gpu_buffer = nz_buffer
            param_offloader._param.data = nz_buffer


def _raise_if_nz_prefetch_incompatible_with_graph(param_infos: list[ParamInfo]) -> None:
    """Fail fast when NZ static buffers would meet ACL graph capture.

    The prefetch H2D copy from ND pinned CPU storage into a FRACTAL_NZ
    static buffer requires an ND->NZ format conversion. The current
    CANN/torch_npu stack does not support running aclop operators during
    NPU graph capture, and on these versions the conversion is only
    available as an aclop implementation, so this combination is
    unsupported for now. Without this check the combo aborts EngineCore
    at capture time with an opaque 107025 capture_end error.
    """
    nz_param_count = sum(1 for info in param_infos if info.use_nz_buffer)
    if nz_param_count == 0:
        return

    # Lazy import: vllm.config pulls in platform plugins, which import us.
    from vllm.config import get_current_vllm_config

    if get_current_vllm_config().model_config.enforce_eager:
        return

    raise RuntimeError(
        f"CPU weight offload is incompatible with ACL graph capture for "
        f"{nz_param_count} offloaded parameter(s) stored in FRACTAL_NZ: the "
        f"prefetch copy (ND pinned CPU storage -> NZ static buffer) requires "
        f"an ND->NZ conversion, which is aclop-only on the current "
        f"CANN/torch_npu versions; running aclop operators during NPU graph "
        f"capture is not supported yet by these libraries. Use "
        f"enforce_eager=True, or disable NZ weights "
        f"(additional_config={{'weight_nz_mode': 0}})."
    )


class AscendPrefetchOffloader(PrefetchOffloader):
    """Ascend prefetch offloader that reuses vLLM behavior with NZ buffers."""

    def __init__(
        self,
        group_size: int,
        num_in_group: int,
        prefetch_step: int,
        offload_params: set[str] | None = None,
        mode: str = "cpu",
    ):
        super().__init__(
            group_size=group_size,
            num_in_group=num_in_group,
            prefetch_step=prefetch_step,
            offload_params=offload_params,
            mode=mode,
        )
        self.module_offloaders: list[_ModuleOffloader] = []

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        import vllm.model_executor.offloader.prefetch as vllm_prefetch

        original_module_offloader = vllm_prefetch._ModuleOffloader
        vllm_prefetch._ModuleOffloader = _ModuleOffloader
        try:
            return super().wrap_modules(modules_generator)
        finally:
            vllm_prefetch._ModuleOffloader = original_module_offloader

    def post_init(self):
        super().post_init()

        device: torch.device | None = None
        param_infos: list[ParamInfo] = []
        for offloader in self.module_offloaders:
            param_infos.extend(offloader.get_param_infos())
            if device is None:
                device = offloader.device
        if device is None:
            # No modules to offload
            return

        _raise_if_nz_prefetch_incompatible_with_graph(param_infos)
        _format_static_buffers_for_nz(self.module_offloaders, self.buffer_pool, param_infos)

        for i in range(min(self.prefetch_step, len(self.module_offloaders))):
            self.module_offloaders[i].start_onload_to_static()


class _ModuleOffloader(VllmModuleOffloader):
    """vLLM module offloader with Ascend NZ-format detection."""

    def __init__(
        self,
        mode: str,
        module: nn.Module,
        copy_stream: torch.cuda.Stream,
        whitelist_param_names: list[str],
        layer_idx: int,
    ):
        super().__init__(
            mode=mode,
            module=module,
            copy_stream=copy_stream,
            whitelist_param_names=whitelist_param_names,
            layer_idx=layer_idx,
        )
        self._wrap_process_weights_for_format_detection()

    def _capture_static_buffer_formats_from_npu_params(self) -> None:
        """
        This function should be invoked during the `process_weights_after_loading`.
        And during `process_weights_after_loading`, the weights would be loaded to NPU
        for quantization and transformation of NPU formats.
        So the format of weights could be figured out during the `process_weights_after_loading`.
        This function is for checking the format of the weights,
        and set a mark to param if its weights is in NZ format.
        """
        for param_offloader in self._param_offloaders.values():
            param = param_offloader._param
            if _is_using_nz_weight(param):
                mark_prefetch_offload_nz_weight(param)

    def _wrap_process_weights_for_format_detection(self) -> None:
        """maybe_trans_nz was called only in quantization scenario"""
        wrapped_quant_methods: set[int] = set()
        for submodule in self.module.modules():
            quant_method = getattr(submodule, "quant_method", None)
            if quant_method is not None and id(quant_method) not in wrapped_quant_methods:
                process_weights = getattr(quant_method, "process_weights_after_loading", None)
                if callable(process_weights):
                    quant_method.process_weights_after_loading = self._wrap_process_weights(process_weights)
                    wrapped_quant_methods.add(id(quant_method))

    def _wrap_process_weights(self, process_weights: Any) -> Any:
        @wraps(process_weights)
        def wrapped_process_weights(*args: Any, **kwargs: Any) -> Any:
            result = process_weights(*args, **kwargs)
            self._capture_static_buffer_formats_from_npu_params()
            return result

        return wrapped_process_weights

    def get_param_infos(self) -> list[ParamInfo]:
        infos = []
        for name, offloader in self._param_offloaders.items():
            cpu_storage = offloader._cpu_storage
            assert cpu_storage is not None, "CPU storage not initialized"
            infos.append(
                ParamInfo(
                    name=name,
                    shape=tuple(cpu_storage.shape),
                    stride=tuple(cpu_storage.stride()),
                    dtype=cpu_storage.dtype,
                    # NOTE(wangjin) different from base
                    use_nz_buffer=_is_prefetch_offload_nz_weight(offloader._param),
                )
            )
        return infos
