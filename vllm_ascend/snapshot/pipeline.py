# SPDX-License-Identifier: Apache-2.0
"""Snapshot-only lifecycle for pipeline-parallel work and buffers."""

import torch
from vllm.distributed.parallel_state import get_pp_group
from vllm.distributed.utils import get_distributed_timeout_or_none

from vllm_ascend.utils import create_hccl_pg_options


def prepare_pipeline_for_snapshot(worker) -> None:
    runner = worker.model_runner
    state = runner.req_states
    if state.num_reqs:
        raise RuntimeError("PP snapshot requires an idle runner with no active requests")
    # V2 non-last stages retain execute_model_state after dummy runs. That is
    # not an active request; it is discarded with the other restore-time state.

    # Complete native Work objects before checkpointing, never wait on restored Work.
    for handle in worker._pp_send_work:
        handle.wait()
    worker._pp_send_work.clear()
    group = get_pp_group()
    group._reap_completed_isends()
    for handles, _ in group._pending_isends:
        for handle in handles:
            handle.wait()
    torch.npu.synchronize()
    group._pending_isends.clear()


def destroy_pipeline_runtime(worker) -> None:
    worker._pp_send_work.clear()
    from vllm_ascend.patch.worker.patch_distributed import _abort_hccl_process_group

    handler = worker.model_runner.pp_handler
    # This sibling group is not owned by the global PP GroupCoordinator.
    _abort_hccl_process_group(handler.broadcast_group)
    torch.distributed.destroy_process_group(handler.broadcast_group)
    handler.broadcast_group = None
    handler.queue.clear()


@torch.inference_mode()
def restore_pipeline_runtime(runner) -> None:
    _refresh_pipeline_moe_groups(runner)
    runner.execute_model_state = None
    # Dummy runs read these buffers before any receive fills them. Preserve
    # storage and all auxiliary payload keys for graph capture and real receives.
    if runner.intermediate_tensors is not None:
        for name, tensor in runner.intermediate_tensors.tensors.items():
            # PP top-k payloads alias the model's index buffer, whose empty
            # sentinel is -1, not a valid cache index such as zero.
            tensor.fill_(-1 if name.startswith("pp_transport_topk_indices_") else 0)

    handler = runner.pp_handler
    group = get_pp_group()
    if torch.distributed.get_backend(group.device_group) == "hccl":
        handler.broadcast_group, _ = torch.distributed.new_subgroups_by_enumeration(
            group.group_ranks,
            timeout=get_distributed_timeout_or_none(),
            backend="hccl",
            pg_options=create_hccl_pg_options("pp_broadcast"),
            group_desc="pp_broadcast",
        )
        assert handler.broadcast_group is not None
    else:
        handler.broadcast_group = group.make_sibling_device_group(group_desc="pp_broadcast")
    handler.is_last_rank = group.is_last_rank
    handler.last_rank = group.last_rank
    handler.req_idx_gen_np.fill(0)
    handler.queue.clear()
    if not handler.is_last_rank:
        handler.queue.extend([None] * group.world_size)
    # Current V2 implements the draft broadcast in PPHandler itself; there
    # are no plugin-installed closures retaining the previous InputBatch.
    handler.sampled_tokens_broadcast = False


def _refresh_pipeline_moe_groups(runner) -> None:
    from vllm.distributed.parallel_state import get_dp_group, get_ep_group, get_tp_group

    from vllm_ascend.distributed.parallel_state import get_mc2_group
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    from vllm_ascend.snapshot.model_runner_lifecycle.module_lifecycle import get_drafter_model

    # The global comm-method registry retains only the last config of each
    # execution shape. Earlier layers' shared experts still own their configs.
    for model in (runner.get_model(), get_drafter_model(runner)):
        if model is None:
            continue
        for module in model.modules():
            if isinstance(module, AscendMoERunner):
                config = module.moe_config
                config.tp_group = get_tp_group()
                config.dp_group = get_dp_group()
                if config.ep_size > 1:
                    config.ep_group = get_ep_group()
                    config.mc2_group = get_mc2_group()
