# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Worker-local bindings for Ascend sleep resources.

This adapter owns orchestration, but borrows model, allocator, transfer engine
and cleanup manager. It does not replace their allocation or HCCL algorithms.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import torch
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger
from vllm.utils.mem_constants import GiB_bytes

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device_allocator.camem import CaMemAllocator
from vllm_ascend.device_allocator.sleep_signals import preserve_worker_signal_handlers
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker


class AscendSleepRuntime:
    def __init__(self, worker: NPUWorker):
        self._worker_ref = weakref.ref(worker)

    @property
    def worker(self) -> NPUWorker:
        worker = self._worker_ref()
        if worker is None:
            raise RuntimeError("Sleep worker has been released")
        return worker

    @staticmethod
    def agree(stage: str, error: Exception | None) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            group = get_world_group().cpu_group
            results: list = [None] * torch.distributed.get_world_size(group)
            # CPU group survives the HCCL extra-cleanup path. Never use a
            # destroyed device communicator to coordinate wake failures.
            torch.distributed.all_gather_object(results, (stage, error is None), group=group)
            if any(name != stage or not success for name, success in results):
                raise RuntimeError(f"Ascend sleep stage {stage!r} failed on one or more ranks") from error
        if error is not None:
            raise error

    def synchronize(self) -> None:
        for handle in self.worker._pp_send_work:
            handle.wait()
        self.worker._pp_send_work = []
        self.worker.synchronize_device()

    def clear_backup(self) -> None:
        self.worker._sleep_saved_buffers = {}
        self.worker._sleep_saved_parameters = {}

    def backup(self, level: int) -> None:
        self.clear_backup()
        if level == 2:
            model = self.worker.model_runner.model
            self.worker._save_sleep_parameters(model)
            self.worker._sleep_saved_buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}

    def restore(self) -> None:
        worker = self.worker
        if worker._sleep_saved_parameters:
            worker._restore_sleep_parameters(worker.model_runner.model)
        if worker._sleep_saved_buffers:
            for name, buffer in worker.model_runner.model.named_buffers():
                if name in worker._sleep_saved_buffers:
                    buffer.data.copy_(worker._sleep_saved_buffers[name].data)
            worker._sleep_saved_buffers = {}

    @staticmethod
    def _cleanup_enabled() -> bool:
        config = get_ascend_config().rl_config
        return config.enabled and config.sleep_mode_extra_cleanup

    def cleanup(self) -> None:
        if self._cleanup_enabled():
            self.worker.sleep_wakeup_manager.sleep()

    def rebuild(self, tags: list[str] | None) -> None:
        with preserve_worker_signal_handlers():
            if self._cleanup_enabled():
                self.worker.sleep_wakeup_manager.wakeup(tags)

    @staticmethod
    def allocation_tags() -> set[str]:
        return {data.tag for data in CaMemAllocator.get_instance().pointer_to_data.values()}

    @staticmethod
    def unregister() -> None:
        global_te.unregister_buffer()

    @staticmethod
    def reregister() -> None:
        global_te.reregister_buffer()

    @staticmethod
    def sleep_allocator(level: int) -> None:
        CaMemAllocator.get_instance().sleep(offload_tags=("weights",) if level == 1 else ())

    @staticmethod
    def wake_allocator(tags: list[str] | None) -> None:
        CaMemAllocator.get_instance().wake_up(tags=tags)

    def validate_wake(self) -> None:
        self.worker._check_nz_disabled()

    def suspend_legacy(self, level: int) -> None:
        # Preserve the original default ordering and memory diagnostic.
        self.worker.synchronize_device()
        before = torch.npu.mem_get_info()[0]
        self.backup(level)
        self.cleanup()
        self.worker.synchronize_device()
        self.unregister()
        self.sleep_allocator(level)
        free, total = torch.npu.mem_get_info()
        freed = free - before
        assert freed >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode (level=%s) freed %.2f GiB memory, %.2f GiB memory is still in use.",
            level,
            freed / GiB_bytes,
            (total - free) / GiB_bytes,
        )

    def resume_legacy(self, tags: list[str] | None) -> None:
        self.validate_wake()
        self.wake_allocator(tags)
        if tags is None or "kv_cache" in tags:
            self.reregister()
        if tags is None or "weights" in tags:
            self.restore()
        self.rebuild(tags)
        self.worker.synchronize_device()
