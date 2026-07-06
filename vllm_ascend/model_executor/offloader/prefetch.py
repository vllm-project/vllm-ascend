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
    _CpuParamOffloader,
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
    buffer_pool: StaticBufferPool,
    param_infos: list[ParamInfo],
) -> None:
    """Cast static buffers to NZ format for keys whose parameters require it."""
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
        """Allocate Ascend NZ-aware static buffers and start initial prefetches."""
        for offloader in self.module_offloaders:
            offloader.sync_cpu_storage()

        param_infos: list[ParamInfo] = []
        device: torch.device | None = None

        for offloader in self.module_offloaders:
            param_infos.extend(offloader.get_param_infos())
            if device is None:
                device = offloader.device

        if device is None:
            return

        self.buffer_pool = StaticBufferPool(
            param_infos=param_infos,
            slot_capacity=self.prefetch_step,
            device=device,
        )
        # NOTE(wangjin) different from the base, but it is important
        _format_static_buffers_for_nz(self.buffer_pool, param_infos)

        for idx, offloader in enumerate(self.module_offloaders):
            slot_idx = idx % self.prefetch_step
            offloader.assign_buffer_slot(self.buffer_pool, slot_idx)

        for offloader in self.module_offloaders:
            offloader.post_init()
            self.total_offloaded_bytes += offloader.offloaded_bytes

        logger.info_once(
            f"[AscendPrefetchOffloader] Initialized {len(self.module_offloaders)} modules. "
            f"Total NPU memory saved: {self.total_offloaded_bytes / 1e9:.4f} GB, "
            f"Static buffer pool: {self.buffer_pool.total_bytes / 1e9:.4f} GB "
            f"(group_size={self.group_size}, num_in_group={self.num_in_group}, "
            f"prefetch_step={self.prefetch_step}, mode={self.mode})"
        )

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
        self.mode = mode
        self.module = module
        self.device = next(module.parameters()).device
        self.copy_stream = copy_stream
        self.layer_idx = layer_idx
        self.offloaded_bytes = 0

        self._copy_done_event = torch.cuda.Event()
        self._event_valid_for_eager = False
        self._prefetch_in_capture = False

        assert self.device != torch.device("cpu"), (
            "Module parameters should not already be on CPU (offloader handles CPU placement)"
        )

        self._buffer_pool: StaticBufferPool | None = None
        self._buffer_slot_idx: int = 0

        param_dict = dict(self.module.named_parameters())
        assert all(name in param_dict for name in whitelist_param_names), (
            f"Whitelist params {whitelist_param_names} not found in module params {list(param_dict.keys())}"
        )

        if mode != "cpu":
            raise ValueError(f"Unknown offload mode: {mode}")
        self._param_offloaders = {
            name: _CpuParamOffloader(module=module, param_name=name) for name in whitelist_param_names
        }
        # NOTE(wangjin) different from base
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
