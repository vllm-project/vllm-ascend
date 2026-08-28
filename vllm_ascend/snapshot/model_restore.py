# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch.nn as nn
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger

from vllm_ascend.snapshot.model_state import dump_state_dict, restore_state_dict
from vllm_ascend.snapshot.tensor_state import (
    reset_model_runtime_tensor_state,
    reset_runtime_tensor_state,
    restore_derived_tensor_state,
    restore_global_tensor_state,
)
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer


def get_drafter_model(runner) -> nn.Module | None:
    """Return the speculative decoder's model when it owns one."""
    drafter = getattr(runner, "drafter", None)
    if drafter is None:
        return None
    model = None
    get_model = getattr(drafter, "get_model", None)
    if callable(get_model):
        try:
            model = get_model()
        except Exception:  # noqa: BLE001
            model = None
    if model is None:
        model = getattr(drafter, "model", None)
    return model if isinstance(model, nn.Module) else None


def dump_model_runner(runner, path: str = "/mnt") -> None:
    tp_size = runner.vllm_config.parallel_config.tensor_parallel_size
    model_name = runner.vllm_config.model_config.model.rstrip("/").rsplit("/", 1)[-1]
    model_dir = os.path.join(path, "snapshot_weight", f"{model_name}_dp{runner.dp_size}_tp{tp_size}")
    os.makedirs(model_dir, exist_ok=True)
    rank_in_group = get_tp_group().rank_in_group
    dump_state_dict(
        runner.get_model(),
        os.path.join(model_dir, f"model_ckpt.{runner.dp_rank}tp{rank_in_group}.pth"),
    )
    # The drafter owns a separate model and derived tensors, so it cannot share
    # the target model's checkpoint file.
    drafter_model = get_drafter_model(runner)
    if drafter_model is not None:
        dump_state_dict(
            drafter_model,
            os.path.join(model_dir, f"model_ckpt_drafter.{runner.dp_rank}tp{rank_in_group}.pth"),
        )


def _restore_one_model(runner, model: nn.Module, model_save_path: str, label: str) -> None:
    restore_state_dict(model, model_save_path, label)
    restore_derived_tensor_state(model, runner.model_config.dtype, label)


def restore_model_runner(runner, path: str = "/mnt") -> None:
    tp_size = runner.vllm_config.parallel_config.tensor_parallel_size
    model_name = runner.vllm_config.model_config.model.rstrip("/").rsplit("/", 1)[-1]
    model_dir = os.path.join(
        path,
        "snapshot_weight",
        f"{model_name}_dp{runner.dp_size}_tp{tp_size}",
    )
    rank_in_group = get_tp_group().rank_in_group
    model = runner.get_model()
    _restore_one_model(
        runner,
        model,
        os.path.join(model_dir, f"model_ckpt.{runner.dp_rank}tp{rank_in_group}.pth"),
        "model",
    )
    drafter_model = get_drafter_model(runner)
    if drafter_model is not None:
        _restore_one_model(
            runner,
            drafter_model,
            os.path.join(model_dir, f"model_ckpt_drafter.{runner.dp_rank}tp{rank_in_group}.pth"),
            "drafter",
        )

    # Restore persistent and derived model state before clearing transient
    # request/capture state used by the first post-resume forward.
    restore_global_tensor_state(model, runner.model_config.hf_config, runner.device)
    _clear_spec_decode_carryover(runner)
    restore_drafter_runtime_buffers(runner)
    _reset_attention_builder_runtime_states(runner)
    _reset_runtime_tensor_states(runner)
    _reset_block_table_device_buffers(runner)


def restore_drafter_runtime_buffers(runner) -> None:
    if isinstance(runner.drafter, AscendEagleProposer):
        runner.drafter.restore_runtime_buffers()


def _clear_spec_decode_carryover(runner) -> None:
    if hasattr(runner, "_draft_token_req_ids"):
        runner._draft_token_req_ids = None
    if hasattr(runner, "_draft_token_ids"):
        runner._draft_token_ids = None
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is not None and hasattr(input_batch, "prev_req_id_to_index"):
        input_batch.prev_req_id_to_index = None


def _reset_attention_builder_runtime_states(runner) -> None:
    builders = [
        builder
        for kv_groups in runner.attn_groups
        for attn_group in kv_groups
        for builder in attn_group.metadata_builders
    ]
    owners = builders + [builder.attn_mask_builder for builder in builders if hasattr(builder, "attn_mask_builder")]
    reset = reset_runtime_tensor_state(owners)
    logger.info(
        "[restore model] attention builder runtime reset: total=%d reset=%d",
        len(builders),
        reset,
    )


def _reset_runtime_tensor_states(runner) -> None:
    """Reset runner staging buffers and model-owned reusable runtime tensors."""
    for staged in (runner.group_len, runner.group_key_idx, runner.group_key_cache_idx):
        staged.gpu.fill_(0)
        staged.cpu.fill_(0)

    reset = reset_model_runtime_tensor_state((runner.get_model(), get_drafter_model(runner)))
    logger.info(
        "[restore model] reset model-owned runtime tensor state for %d owners",
        reset,
    )


def _reset_block_table_device_buffers(runner) -> None:
    # Clear both CPU source rows and device rows. Graph recapture can otherwise
    # copy snapshot-time block ids back to the device before a real request
    # repopulates the active rows.
    block_table = runner.input_batch.block_table
    block_table.clear()
    logger.info(
        "[restore model] zeroed %d block-table device tensor(s)",
        len(block_table.block_tables),
    )
