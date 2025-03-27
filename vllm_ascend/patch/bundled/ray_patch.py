import torch_npu  # noqa: F401
import vllm
from vllm.executor.ray_utils import RayWorkerWrapper

if RayWorkerWrapper is not None:

    class NPURayWorkerWrapper(RayWorkerWrapper):
        """Importing torch_npu in other Ray processes through an empty class and 
        a monkey patch.

        When Ray performs a remote call, it serializes the Task or Actor and passes 
        it to the Worker process, where it is deserialized and executed. 
    
        If no patch is applied, the default code of the RayWorkerWrapper provided 
        by vLLM is used, which does not import torch_npu, causing an error in the 
        Worker process.
        See https://github.com/vllm-project/vllm-ascend/pull/92.
        """

        pass

    vllm.executor.ray_utils.RayWorkerWrapper = NPURayWorkerWrapper