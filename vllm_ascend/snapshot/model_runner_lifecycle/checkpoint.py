# SPDX-License-Identifier: Apache-2.0

import os

from vllm.distributed.parallel_state import get_pcp_group, get_pp_group, get_tp_group

from vllm_ascend.snapshot.model_runner_lifecycle.module_lifecycle import (
    dump_state_dict,
    get_drafter_model,
)


def model_checkpoint_path(runner, path: str, *, drafter: bool = False) -> str:
    parallel = runner.vllm_config.parallel_config
    tp_size = parallel.tensor_parallel_size
    model_name = runner.vllm_config.model_config.model.rstrip("/").rsplit("/", 1)[-1]
    directory = f"{model_name}_dp{runner.dp_size}_tp{tp_size}"
    rank_in_group = get_tp_group().rank_in_group
    rank = f"{runner.dp_rank}tp{rank_in_group}"
    if parallel.pipeline_parallel_size > 1:
        directory += f"_pp{parallel.pipeline_parallel_size}"
        rank += f"pp{get_pp_group().rank_in_group}"
    if parallel.prefill_context_parallel_size > 1:
        directory += f"_pcp{parallel.prefill_context_parallel_size}"
        rank += f"pcp{get_pcp_group().rank_in_group}"
    prefix = "model_ckpt_drafter" if drafter else "model_ckpt"
    return os.path.join(path, "snapshot_weight", directory, f"{prefix}.{rank}.pth")


def dump_model_runner(runner, path: str = "/mnt") -> None:
    model_path = model_checkpoint_path(runner, path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump_state_dict(runner.get_model(), model_path)
    # The drafter owns a separate model and derived tensors, so it cannot share
    # the target model's checkpoint file.
    drafter_model = get_drafter_model(runner)
    if drafter_model is not None:
        dump_state_dict(
            drafter_model,
            model_checkpoint_path(runner, path, drafter=True),
        )
