# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm eager attention before DP workers can take different execution paths."""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

import torch
from vllm.config import set_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import override_mrv2_in_profile_run
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import is_deepseek_v41

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker


@torch.inference_mode()
def warmup_deepseek_v41_eager(worker: NPUWorker) -> None:
    """Exercise prefill and idle attention, synchronizing before expert collectives.

    Memory profiling skips attention, and eager workers do not capture graphs.
    Consequently the first idle DP step can compile a packaged attention variant
    while peers are already waiting inside an expert collective. Run both paths
    before serving, including the draft model, without depending on private DSL
    compiler APIs. Temporary MoE hooks keep compilation skew out of device-side
    collective timeouts; a barrier only at the end of the pass would be too late.
    """
    config = worker.vllm_config
    parallel = config.parallel_config
    if (
        not config.model_config.enforce_eager
        or parallel.data_parallel_size <= 1
        or not parallel.enable_expert_parallel
        or not is_deepseek_v41(config.model_config.hf_config)
        or not get_current_hardware_profile().supports(HardwareCapability.DSV41_PACKED_CACHE)
    ):
        return

    # Lazy import: only workers running this model need its implementations.
    from vllm_ascend.models.deepseek_v41.model import DeepseekV41MoE

    runner = worker.model_runner
    models = [runner.model]
    speculator = getattr(runner, "speculator", None)
    if speculator is not None and getattr(speculator, "model", None) is not None:
        models.append(speculator.model)
    expert_layers = {layer for model in models for layer in model.modules() if isinstance(layer, DeepseekV41MoE)}
    ep_group = get_ep_group()

    def wait_for_attention(_module, _args) -> None:
        # QLI compilation is synchronous on the host; its execution is async.
        # Complete both before releasing any rank into the following MoE.
        torch.npu.synchronize()
        ep_group.barrier()  # GroupCoordinator uses the CPU process group.

    logger.info("Warming DeepSeek V4.1 eager prefill and idle attention on all EP ranks.")
    with ExitStack() as hooks, set_current_vllm_config(config):
        for layer in expert_layers:
            hooks.callback(layer.register_forward_pre_hook(wait_for_attention).remove)
        ep_group.barrier()
        # The large pass and the one-request idle pass can select different QLI
        # variants. Match execute_dummy_batch's query length and state guard.
        # Synthetic inputs can route the entire large batch to a few experts.
        # Balance routing as in memory profiling, but still execute attention.
        with override_mrv2_in_profile_run(True):
            runner._dummy_run(
                config.scheduler_config.max_num_batched_tokens,
                skip_attn=False,
                skip_gdn_state_update=True,
            )
        runner._dummy_run(
            getattr(runner, "uniform_decode_query_len", 1),
            uniform_decode=True,
            skip_attn=False,
            skip_gdn_state_update=True,
        )
        torch.npu.synchronize()
        ep_group.barrier()
    logger.info("DeepSeek V4.1 eager attention warmup finished.")
