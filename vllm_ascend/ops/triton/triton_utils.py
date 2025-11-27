from typing import Any, Dict

import torch
from vllm.triton_utils import HAS_TRITON, triton

NUM_AICORE = -1
NUM_VECTORCORE = -1


def init_device_properties_triton():
    global NUM_AICORE, NUM_VECTORCORE
    if NUM_AICORE == -1 and HAS_TRITON:
        device_properties: Dict[str, Any] = (
            triton.runtime.driver.active.utils.get_device_properties(
                torch.npu.current_device()))
        NUM_AICORE = device_properties.get("num_aicore", -1)
        NUM_VECTORCORE = device_properties.get("num_vectorcore", -1)
        assert NUM_AICORE > 0 and NUM_VECTORCORE > 0, "Failed to detect device properties."
