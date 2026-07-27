# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from datetime import timedelta
from typing import TYPE_CHECKING

import torch
from torch.distributed.distributed_c10d import _set_pg_timeout
from vllm.config import set_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    stateless_destroy_torch_distributed_process_group,
    stateless_init_torch_distributed_process_group,
)
from vllm.logger import init_logger
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest
from vllm.v1.serial_utils import run_method

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


class WorkerSentinel:
    """Per-worker sentinel for fault tolerance.

    Handles commands dispatched from EngineCoreSentinel via collective_rpc,
    including device restart and DP group re-initialization on retry.
    """

    def __init__(self, worker: "Worker", device: torch.device):
        self.worker = worker
        self.device = device
        self.dp_rank = worker.parallel_config.data_parallel_rank
        self.dp_size = worker.parallel_config.data_parallel_size
        self.data_parallel_master_ip = worker.parallel_config.data_parallel_master_ip
        self.set_dp_gloo_timeout()

    def set_dp_gloo_timeout(self) -> None:
        timeout = timedelta(seconds=self.worker.vllm_config.parallel_config.cpu_distributed_timeout_seconds)
        dp_cpu_group = get_dp_group()
        _set_pg_timeout(timeout=timeout, group=dp_cpu_group.cpu_group)

    def handle_command(self, ft_request: FaultToleranceRequest):
        """Dispatch an FT command by instruction name."""
        with set_current_vllm_config(self.worker.vllm_config):
            return run_method(self, ft_request.instruction, (ft_request,), {})

    def query_mask(self, ft_request: FaultToleranceRequest) -> dict:
        """Query mask for fault tolerance.

        Ascend's all2all operator does not currently support querying mask.
        Returns all-zero mask for now; will be updated once the operator
        supports it.
        """
        return {"mask": get_ep_group().world_size * [0]}

    def retry(self, ft_request: FaultToleranceRequest):
        self._clean_worker_state()
        torch.accelerator.synchronize()
        params = ft_request.params
        if self.dp_size > 1:
            old_cpu_group = get_dp_group().cpu_group
            stateless_destroy_torch_distributed_process_group(old_cpu_group)
            world_size = self.worker.parallel_config.world_size
            port = params["new_stateless_dp_group_ports"][self.worker.rank % world_size]
            get_dp_group().cpu_group = stateless_init_torch_distributed_process_group(
                self.data_parallel_master_ip,
                port,
                self.dp_rank,
                self.dp_size,
                backend="gloo",
            )

    def _clean_worker_state(self):
        import torch_npu

        from vllm_ascend.platform import NPUPlatform

        NPUPlatform.set_device(self.device)
        torch_npu.npu.stop_device(self.device.index)
        torch_npu.npu.restart_device(self.device.index)
        torch_npu.distributed.reinit_process_group(None, False)
        torch.npu.synchronize()
        model_runner = self.worker.model_runner
        model_runner.execute_model_state = None
        if self.worker.use_v2_model_runner:
            for req_id in list(model_runner.req_states.req_id_to_index):
                model_runner._remove_request(req_id)
        else:
            model_runner.kv_connector_output = None
            input_batch = model_runner.input_batch
            cached_req_ids = list(input_batch.req_id_to_index)
            for req_id in cached_req_ids:
                model_runner.requests.pop(req_id, None)
                model_runner.num_prompt_logprobs.pop(req_id, None)
                input_batch.remove_request(req_id)
            input_batch.condense()
            input_batch.refresh_metadata()
            input_batch.req_prompt_embeds.clear()
            model_runner.async_output_copy_stream = torch.cuda.Stream()
            model_runner.prepare_inputs_event = torch.Event()
