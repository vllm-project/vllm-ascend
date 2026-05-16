import os

from vllm.tasks import SupportedTask


def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
    tasks = list[SupportedTask]()

    if self.model_config.runner_type == "generate":
        if int(os.getenv("INFER_STATUS", "0")) > 0:
            tasks.extend(self.get_supported_generation_tasks())
        else:
            tasks.extend(["generate"])
    if self.model_config.runner_type == "pooling":
        tasks.extend(self.get_supported_pooling_tasks())
    return tuple(tasks)
