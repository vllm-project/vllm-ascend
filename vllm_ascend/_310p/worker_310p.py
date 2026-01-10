import torch
import torch_npu
from vllm.logger import logger

from vllm_ascend.worker.worker import NPUWorker
from vllm_ascend.utils import is_310p
from vllm_ascend._310p.modelrunner_310p import NPUModelRunner310


class NPUWorker310(NPUWorker):
    def init_device(self):
        self.device = self._init_device()

        torch_npu.npu.set_compile_mode(jit_compile=False)

        from vllm_ascend.worker.worker import init_workspace_manager
        init_workspace_manager(self.device, num_ubatches=1)

        self.model_runner = NPUModelRunner310(self.vllm_config, self.device)

    def _warm_up_atb(self):
        logger.info("Skip warm-up atb ops for 310P device")
