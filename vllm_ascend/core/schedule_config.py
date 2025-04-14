from dataclasses import dataclass, asdict
from typing import Union, Type

from vllm.config import SchedulerConfig


@dataclass
class AscendSchedulerConfig(SchedulerConfig):
    enable_chunked_prefill: bool = False
    policy: str = "fcfs"
    num_scheduler_steps: int = 1
    scheduler_cls: Union[str, Type[object]] = "vllm_ascend.core.scheduler.AscendScheduler"


    @classmethod
    def initialize_from_config(cls, vllm_scheduler_config: SchedulerConfig, ascend_scheduler_config: dict):
        scheduler_config = asdict(vllm_scheduler_config)
        # Override default values into original SchedulerConfig
        scheduler_config["enable_chunked_prefill"] = False
        scheduler_config["policy"] = "fcfs"
        scheduler_config["num_scheduler_steps"] = 1
        scheduler_config["scheduler_cls"] = "vllm_ascend.core.scheduler.AscendScheduler"
        # Override params in original SchedulerConfig with params in additional_config.ascend_scheduler_config
        for k, v in ascend_scheduler_config.items():
            scheduler_config[k] = v
        # The "chunked_prefill_enabled" param of vllm's SchedulerConfig can't be initialized.
        scheduler_config.pop("chunked_prefill_enabled")
        return cls(**scheduler_config)


    def __post_init__(self) -> None:
        self.chunked_prefill_enabled = self.enable_chunked_prefill
        if self.policy != "fcfs":
            raise NotImplementedError(f"currently AscendScheduler only supports fcfs policy, got {self.policy}")
        if self.is_multimodal_model:
            raise NotImplementedError(f"currently AscendScheduler only supports LLM modles.")
        if self.num_scheduler_steps >t1:
            raise NotImplementedError(f"currently AscendScheduler doesn't support multi-step.")
        if self.send_delta_data:
            raise NotImplementedError(f"currently AscendScheduler doesn't support send_delta_data.")
        if self.delay_factor > 0:
            raise NotImplementedError(f"currently AscendScheduler doesn't support scheduler_delay_factor.")