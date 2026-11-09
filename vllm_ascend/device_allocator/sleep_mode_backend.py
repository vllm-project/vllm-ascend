# SPDX-License-Identifier: Apache-2.0
"""CaMem mechanism for the upstream sleep-mode backend contract.

Worker policy (saved tensors, connector registration and RL extra cleanup)
remains in NPUWorker until all level/tag contracts can be inherited safely.
"""

from vllm.device_allocator.sleep_mode_backend import SleepModeBackend

from vllm_ascend.device_allocator.camem import CaMemAllocator
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te


class CaMemSleepBackend(SleepModeBackend):
    def suspend(self, level: int = 1) -> None:
        if level not in (1, 2):
            raise ValueError("Ascend sleep level must be 1 or 2.")
        allocator = CaMemAllocator.get_instance()
        # A partial unmap must never leave the backend advertised as running.
        self._state = "SUSPENDED"
        global_te.unregister_buffer()
        allocator.sleep(offload_tags=("weights",) if level == 1 else ())

    def resume(self, tags: list[str] | None = None) -> None:
        allocator = CaMemAllocator.get_instance()
        allocator.wake_up(tags=tags)
        if tags is None or "kv_cache" in tags:
            global_te.reregister_buffer()
        self._state = "SUSPENDED" if any(data.is_asleep for data in allocator.pointer_to_data.values()) else "RUNNING"

    def discard(self, tags: tuple[str, ...]) -> None:
        allocator = CaMemAllocator.get_instance()
        if "kv_cache" in tags:
            global_te.unregister_buffer()
        self._state = "SUSPENDED"
        allocator.discard(tags)

    @classmethod
    def preserves_communicators(cls) -> bool:
        return True

    @classmethod
    def preserves_compiled_artifacts(cls) -> bool:
        return True
