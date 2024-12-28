from typing import Optional

from vllm.config import VllmConfig
from vllm.platforms import Platform, PlatformEnum


class NPUPlatform(Platform):
    _enum = PlatformEnum.UNSPECIFIED
    device_name = "npu"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_device_name(cls) -> str:
        return "npu"

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        parallel_config.worker_cls = \
            "vllm_ascend_plugin.worker.NPUWorker"
