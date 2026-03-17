import gc
import time

import torch
import torch.nn as nn
from torch.nn import Module
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_ascend.model_loader.rfork.rfork_worker import RForkWorker

logger = init_logger(__name__)


@register_model_loader("rfork")
class RForkModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        raise NotImplementedError

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        raise NotImplementedError

    def _get_fallback_load_format(self) -> str:
        fallback = getattr(self.load_config, "rfork_fallback_load_format", None)
        if fallback is None:
            fallback = "auto"
            setattr(self.load_config, "rfork_fallback_load_format", fallback)
        if str(fallback) == "rfork":
            logger.warning(
                "rfork_fallback_load_format is rfork, force fallback to auto "
                "to avoid recursive rfork retry loop."
            )
            fallback = "auto"
            setattr(self.load_config, "rfork_fallback_load_format", fallback)
        return str(fallback)

    def _build_rfork_worker(self, vllm_config: VllmConfig) -> RForkWorker:
        kv_transfer_config = vllm_config.kv_transfer_config
        disaggregation_mode = (
            "kv_both" if kv_transfer_config is None else str(kv_transfer_config.kv_role)
        )
        device = torch.device(vllm_config.device_config.device)
        gpu_id = (
            device.index
            if device.index is not None
            else int(torch.npu.current_device())
        )
        return RForkWorker(
            disaggregation_mode=disaggregation_mode,
            node_rank=vllm_config.parallel_config.node_rank,
            tp_rank=get_tensor_model_parallel_rank(),
            gpu_id=gpu_id,
            dtype=str(vllm_config.model_config.dtype),
            is_draft_model=False,
        )

    def _get_or_create_rfork_worker(self, vllm_config: VllmConfig) -> RForkWorker:
        rfork_worker = getattr(self.load_config, "rfork_worker", None)
        if rfork_worker is None:
            rfork_worker = self._build_rfork_worker(vllm_config)
            setattr(self.load_config, "rfork_worker", rfork_worker)
            logger.info("RFork worker initialized, load_format=rfork")
        return rfork_worker

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> Module | None:
        device_config = vllm_config.device_config
        load_config = self.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)

        with set_default_torch_dtype(model_config.dtype):
            need_del = False
            rfork_worker = None
            try:
                rfork_worker = self._get_or_create_rfork_worker(vllm_config)
                if not rfork_worker.is_seed_available():
                    raise RuntimeError("seed is not available.")

                with target_device:
                    model = initialize_model(
                        vllm_config=vllm_config,
                        model_config=model_config,
                        prefix=prefix,
                    )
                    need_del = True

                weight_load_start_time = time.time()
                if not rfork_worker.pre_transfer(model):
                    raise RuntimeError("pre_transfer failed.")
                if not rfork_worker.transfer(model):
                    raise RuntimeError("transfer failed.")
                if not rfork_worker.post_transfer():
                    raise RuntimeError("post_transfer failed.")
                logger.info(
                    "Loading weights took %.2f seconds",
                    time.time() - weight_load_start_time,
                )

                rfork_worker.set_transfer_result(True)
                rfork_worker.start_seed_service(model)
                process_weights_after_loading(model, model_config, target_device)

                return model.eval()
            except Exception as e:
                logger.warning(
                    "RFork transfer failed, cleaning up and falling back: %s",
                    e,
                )

                if rfork_worker is not None:
                    rfork_worker.post_transfer()
                    rfork_worker.set_transfer_result(False)

                if need_del:
                    del model
                    gc.collect()
                    torch.npu.empty_cache()
                    for _ in range(3):
                        gc.collect()
                        torch.npu.empty_cache()

                fallback = self._get_fallback_load_format()
                self.load_config.load_format = fallback
                logger.info("fall back into %s to load model", fallback)

                from vllm.model_executor.model_loader import get_model

                return get_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )
