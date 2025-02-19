import vllm
from vllm.executor.ray_utils import RayWorkerWrapper
import torch_npu # noqa: F401

class NPURayWorkerWrapper(RayWorkerWrapper):
    """Importing torch_npu in other Ray processes through an empty class and a monkey patch.
    """
    pass

vllm.executor.ray_utils.RayWorkerWrapper = NPURayWorkerWrapper
