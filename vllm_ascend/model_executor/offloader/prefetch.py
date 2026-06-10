# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU port of PrefetchOffloader — uses torch.npu.* APIs including is_current_stream_capturing."""

from collections.abc import Generator

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import vllm.model_executor.offloader.prefetch_ops  # noqa: F401
from vllm.logger import logger
from vllm.model_executor.offloader.base import BaseOffloader, should_pin_memory
from vllm.model_executor.offloader.prefetch import (
    ParamInfo,
    StaticBufferPool,
    _BaseParamOffloader,
)


class NPUPrefetchOffloader(BaseOffloader):
    """NPU version of PrefetchOffloader — replaces torch.cuda.* with torch.npu.*."""

    def __init__(
        self,
        group_size: int,
        num_in_group: int,
        prefetch_step: int,
        offload_params: set[str] | None = None,
        mode: str = "cpu",
    ):
        self.group_size = group_size
        self.num_in_group = num_in_group
        self.prefetch_step = prefetch_step
        self.offload_params = offload_params or set()
        self.mode = mode
        self.copy_stream = torch.npu.Stream()
        self.module_offloaders: list[_NPUModuleOffloader] = []
        self.buffer_pool: StaticBufferPool | None = None
        self.total_offloaded_bytes = 0

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        assert len(self.module_offloaders) == 0

        all_modules = []
        offload_modules = []

        for module_index, module in enumerate(modules_generator):
            all_modules.append(module)
            if module_index % self.group_size >= self.group_size - self.num_in_group:
                if self.offload_params:
                    whitelist = [
                        name
                        for name, _ in module.named_parameters()
                        if any(f".{p}." in f".{name}." for p in self.offload_params)
                    ]
                else:
                    whitelist = [name for name, _ in module.named_parameters()]

                if not whitelist:
                    continue

                offload_modules.append(module)
                self.module_offloaders.append(
                    _NPUModuleOffloader(
                        mode=self.mode,
                        module=module,
                        copy_stream=self.copy_stream,
                        whitelist_param_names=whitelist,
                        layer_idx=len(self.module_offloaders),
                    )
                )

        for index, module in enumerate(offload_modules):
            self._hook_module_forward(index, module)

        return all_modules

    def _hook_module_forward(self, index: int, module: nn.Module):
        original_forward = module.forward

        def forward(*args, **kwargs):
            module.forward = original_forward
            input_tensor = args[0] if args else kwargs.get("hidden_states")
            torch.ops.vllm.wait_prefetch(input_tensor, index)
            output = original_forward(*args, **kwargs)
            next_index = (index + self.prefetch_step) % len(self.module_offloaders)
            if isinstance(output, tuple):
                torch.ops.vllm.start_prefetch(output[0], next_index)
            else:
                torch.ops.vllm.start_prefetch(output, next_index)
            module.forward = forward
            return output

        module.forward = forward

    def _wait_for_layer(self, layer_idx: int):
        offloader = self.module_offloaders[layer_idx]
        if torch.npu.is_current_stream_capturing():
            if not offloader._prefetch_in_capture:
                return
            torch.npu.current_stream().wait_event(offloader._copy_done_event)
            offloader._prefetch_in_capture = False
        else:
            if offloader._event_valid_for_eager:
                torch.npu.current_stream().wait_event(offloader._copy_done_event)
            else:
                torch.npu.current_stream().wait_stream(self.copy_stream)

    def _start_prefetch(self, layer_idx: int):
        self.module_offloaders[layer_idx].start_onload_to_static()

    def sync_prev_onload(self):
        torch.npu.current_stream().wait_stream(self.copy_stream)

    def join_after_forward(self):
        for offloader in self.module_offloaders:
            if offloader._prefetch_in_capture:
                torch.npu.current_stream().wait_event(offloader._copy_done_event)
                offloader._prefetch_in_capture = False

    def post_init(self):
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

        for idx, offloader in enumerate(self.module_offloaders):
            slot_idx = idx % self.prefetch_step
            offloader.assign_buffer_slot(self.buffer_pool, slot_idx)

        for offloader in self.module_offloaders:
            offloader.post_init()
            self.total_offloaded_bytes += offloader.offloaded_bytes

        logger.info_once(
            f"[NPUPrefetchOffloader] Initialized {len(self.module_offloaders)} modules. "
            f"Total NPU memory saved: {self.total_offloaded_bytes / 1e9:.4f} GB, "
            f"Static buffer pool: {self.buffer_pool.total_bytes / 1e9:.4f} GB "
            f"(group_size={self.group_size}, num_in_group={self.num_in_group}, "
            f"prefetch_step={self.prefetch_step})"
        )

        for i in range(min(self.prefetch_step, len(self.module_offloaders))):
            self.module_offloaders[i].start_onload_to_static()


class _NPUModuleOffloader:
    """NPU version of _ModuleOffloader: all torch.cuda.* → torch.npu.*."""

    def __init__(
        self,
        mode: str,
        module: nn.Module,
        copy_stream: torch.npu.Stream,
        whitelist_param_names: list[str],
        layer_idx: int,
    ):
        self.mode = mode
        self.module = module
        self.device = next(module.parameters()).device
        self.copy_stream = copy_stream
        self.layer_idx = layer_idx
        self.offloaded_bytes = 0

        self._copy_done_event = torch.npu.Event()
        self._event_valid_for_eager = False
        self._prefetch_in_capture = False

        assert self.device != torch.device("cpu")

        self._buffer_pool: StaticBufferPool | None = None
        self._buffer_slot_idx: int = 0

        param_dict = dict(self.module.named_parameters())
        assert all(name in param_dict for name in whitelist_param_names)

        self._param_offloaders = {
            name: _BaseParamOffloader.create(mode, module=module, param_name=name) for name in whitelist_param_names
        }

    def post_init(self):
        for param_offloader in self._param_offloaders.values():
            param_offloader.post_init()
            self.offloaded_bytes += param_offloader.offloaded_bytes

    def sync_cpu_storage(self):
        for param_offloader in self._param_offloaders.values():
            param_offloader.sync_cpu_storage()

        deleted = [
            name for name, offloader in self._param_offloaders.items() if getattr(offloader, "_param_deleted", False)
        ]
        for name in deleted:
            del self._param_offloaders[name]

    def get_param_infos(self) -> list[ParamInfo]:
        infos = []
        for name, offloader in self._param_offloaders.items():
            cpu_storage = offloader._cpu_storage
            assert cpu_storage is not None
            infos.append(
                ParamInfo(
                    name=name,
                    shape=tuple(cpu_storage.shape),
                    stride=tuple(cpu_storage.stride()),
                    dtype=cpu_storage.dtype,
                )
            )
        return infos

    def assign_buffer_slot(self, pool: StaticBufferPool, slot_idx: int):
        self._buffer_pool = pool
        self._buffer_slot_idx = slot_idx
        for name, offloader in self._param_offloaders.items():
            cpu_storage = offloader._cpu_storage
            assert cpu_storage is not None
            buffer = pool.get_buffer(
                name=name,
                shape=tuple(cpu_storage.shape),
                stride=tuple(cpu_storage.stride()),
                dtype=cpu_storage.dtype,
                slot_idx=slot_idx,
            )
            offloader.assign_static_buffer(buffer)

    def start_onload_to_static(self):
        assert self._buffer_pool is not None

        self._prefetch_in_capture = torch.npu.is_current_stream_capturing()

        fork_event = torch.npu.Event()
        torch.npu.current_stream().record_event(fork_event)
        self.copy_stream.wait_event(fork_event)

        with torch.npu.stream(self.copy_stream):
            for name, offloader in self._param_offloaders.items():
                cpu_storage = offloader._cpu_storage
                gpu_buffer = offloader._gpu_buffer
                assert cpu_storage is not None
                assert gpu_buffer is not None
                assert not should_pin_memory() or cpu_storage.is_pinned(), f"CPU storage for {name} is not pinned!"
                gpu_buffer.copy_(cpu_storage, non_blocking=True)

        self._copy_done_event.record(self.copy_stream)
        self._event_valid_for_eager = not torch.npu.is_current_stream_capturing()
