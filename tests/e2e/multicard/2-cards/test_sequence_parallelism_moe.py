#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Two-card e2e tests for SequenceParallelismMoePass patterns:
# - MiddleLayerAllgatherAddRMSNormPattern (all_gather + slice + RMSNorm)
# - Qwen3VLMiddleLayerAllgatherAddRMSNormPattern (all_gather + slice + add + RMSNorm)
# - AllGatherChunkNoOpPattern (all_gather + sequence_parallel_chunk_impl -> identity)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import queue
import traceback
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn as nn
import vllm.config
from vllm.compilation.passes.fx_utils import OpOverload
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    init_distributed_environment,
    tensor_model_parallel_all_gather,
)
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    initialize_model_parallel,
)
from vllm.utils.system_utils import update_environment_variables

import vllm_ascend.ops.register_custom_ops  # noqa
from tests.e2e.singlecard.compile.backend import TestBackend as CompileTestBackend
from vllm_ascend.compilation.passes.sequence_parallelism_moe import (
    SequenceParallelismMoePass,
)
from vllm_ascend.utils import enable_custom_op

MASTER_PORT = 29500
WORLD_SIZE = 2
WORKER_READY = "__ready__"
WORKER_STOP = "__stop__"
WORKER_RESULT_TIMEOUT_S = 180
WORKER_JOIN_TIMEOUT_S = 30


class BaseAllGatherRMSNormModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dtype: torch.dtype,
        eps: float = 1e-6,
        device: str = "npu",
    ):
        super().__init__()
        self.eps = eps
        self.norm_w = torch.randn(hidden_size, dtype=dtype, device=device)

    def _all_gather_sliced(self, x: torch.Tensor, num_tokens_helper: torch.Tensor) -> torch.Tensor:
        num_tokens = num_tokens_helper.shape[0]
        activated = torch.relu(x)
        gathered = tensor_model_parallel_all_gather(activated, 0)
        return gathered[:num_tokens]

    @staticmethod
    def ops_in_model_after() -> tuple[tuple[OpOverload, int], ...]:
        return (
            (torch.ops.vllm.all_gather.default, 1),
            (torch.ops._C_ascend.npu_add_rms_norm_bias.default, 1),
            (torch.ops.vllm.maybe_chunk_residual.default, 1),
        )


class AllGatherRMSNormModel(BaseAllGatherRMSNormModel):
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        num_tokens_helper: torch.Tensor,
    ) -> torch.Tensor:
        sliced = self._all_gather_sliced(x, num_tokens_helper)
        rms_out = torch.ops._C_ascend.npu_add_rms_norm_bias(sliced, residual, self.norm_w, None, self.eps)
        return rms_out[0]

    @staticmethod
    def ops_in_model_before() -> tuple[tuple[OpOverload, int], ...]:
        return (
            (torch.ops.vllm.all_gather.default, 1),
            (torch.ops._C_ascend.npu_add_rms_norm_bias.default, 1),
        )


class Qwen3VLAllGatherRMSNormModel(BaseAllGatherRMSNormModel):
    """Exercises Qwen3VLMiddleLayerAllgatherAddRMSNormPattern (all_gather + slice + add + RMSNorm)."""

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        num_tokens_helper: torch.Tensor,
        deepstack_input_embeds: torch.Tensor,
    ) -> torch.Tensor:
        sliced = self._all_gather_sliced(x, num_tokens_helper)
        add_ = sliced + deepstack_input_embeds
        result, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(add_, residual, self.norm_w, None, self.eps)
        # Keep the residual output live so the traced graph preserves the full pattern.
        result = result - residual
        return result

    @staticmethod
    def ops_in_model_before() -> tuple[tuple[OpOverload, int], ...]:
        return (
            (torch.ops.vllm.all_gather.default, 1),
            (torch.ops.aten.add.Tensor, 1),
            (torch.ops._C_ascend.npu_add_rms_norm_bias.default, 1),
        )


class AllGatherChunkNoOpModel(nn.Module):
    """Exercises AllGatherChunkNoOpPattern (all_gather + sequence_parallel_chunk_impl -> identity)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(x)
        gathered = tensor_model_parallel_all_gather(z, 0)
        return torch.ops.vllm.sequence_parallel_chunk_impl(gathered)

    @staticmethod
    def ops_in_model_before() -> tuple[tuple[OpOverload, int], ...]:
        return (
            (torch.ops.vllm.all_gather.default, 1),
            (torch.ops.vllm.sequence_parallel_chunk_impl.default, 1),
        )

    @staticmethod
    def ops_in_model_after() -> tuple[tuple[OpOverload, int], ...]:
        return (
            (torch.ops.vllm.all_gather.default, 0),
            (torch.ops.vllm.sequence_parallel_chunk_impl.default, 0),
        )


def _build_all_gather_rms_norm_inputs(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    tp_size: int,
) -> tuple[torch.Tensor, ...]:
    local_tokens = batch_size * seq_len
    num_tokens = local_tokens * tp_size
    x = torch.randn(local_tokens, hidden_size, dtype=dtype)
    residual = torch.zeros(num_tokens, hidden_size, dtype=dtype)
    num_tokens_helper = torch.empty(num_tokens, device=x.device, dtype=dtype)
    return (x, residual, num_tokens_helper)


def _build_qwen3vl_inputs(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    tp_size: int,
) -> tuple[torch.Tensor, ...]:
    x, residual, num_tokens_helper = _build_all_gather_rms_norm_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        dtype=dtype,
        tp_size=tp_size,
    )
    deepstack = torch.randn(num_tokens_helper.shape[0], hidden_size, dtype=dtype)
    return (x, residual, num_tokens_helper, deepstack)


def _build_allgather_chunk_noop_inputs(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    tp_size: int,
) -> tuple[torch.Tensor, ...]:
    del tp_size
    local_tokens = batch_size * seq_len
    x = torch.randn(local_tokens, hidden_size, dtype=dtype)
    return (x,)


def _create_all_gather_rms_norm_model(
    hidden_size: int,
    dtype: torch.dtype,
    eps: float,
    device: str,
) -> nn.Module:
    return AllGatherRMSNormModel(hidden_size=hidden_size, dtype=dtype, eps=eps, device=device)


def _create_qwen3vl_model(
    hidden_size: int,
    dtype: torch.dtype,
    eps: float,
    device: str,
) -> nn.Module:
    return Qwen3VLAllGatherRMSNormModel(hidden_size=hidden_size, dtype=dtype, eps=eps, device=device)


def _create_allgather_chunk_noop_model(
    hidden_size: int,
    dtype: torch.dtype,
    eps: float,
    device: str,
) -> nn.Module:
    del hidden_size, dtype, eps, device
    return AllGatherChunkNoOpModel()


@dataclass(frozen=True)
class PatternTestCase:
    model_factory: Any
    input_builder: Any
    dynamic_input_indices: tuple[int, ...]
    pre_pass_expected_counts_factory: Any
    post_pass_expected_counts_factory: Any


PATTERN_TEST_CASES = {
    "middle_layer_allgather_add_rms_norm": PatternTestCase(
        model_factory=_create_all_gather_rms_norm_model,
        input_builder=_build_all_gather_rms_norm_inputs,
        dynamic_input_indices=(0, 2),
        pre_pass_expected_counts_factory=AllGatherRMSNormModel.ops_in_model_before,
        post_pass_expected_counts_factory=AllGatherRMSNormModel.ops_in_model_after,
    ),
    "qwen3vl_middle_layer_allgather_add_rms_norm": PatternTestCase(
        model_factory=_create_qwen3vl_model,
        input_builder=_build_qwen3vl_inputs,
        dynamic_input_indices=(0, 2, 3),
        pre_pass_expected_counts_factory=Qwen3VLAllGatherRMSNormModel.ops_in_model_before,
        post_pass_expected_counts_factory=Qwen3VLAllGatherRMSNormModel.ops_in_model_after,
    ),
    "allgather_chunk_noop": PatternTestCase(
        model_factory=_create_allgather_chunk_noop_model,
        input_builder=_build_allgather_chunk_noop_inputs,
        dynamic_input_indices=(0,),
        pre_pass_expected_counts_factory=AllGatherChunkNoOpModel.ops_in_model_before,
        post_pass_expected_counts_factory=AllGatherChunkNoOpModel.ops_in_model_after,
    ),
}


def _assert_op_counts(
    backend: CompileTestBackend,
    expected_counts: tuple[tuple[OpOverload, int], ...],
    before: bool = False,
) -> None:
    for op, expected_count in expected_counts:
        actual_count = backend.op_count(op, before=before)
        stage = "before" if before else "after"
        assert actual_count == expected_count, (
            f"op {stage} pass: {op} expected {expected_count}, but got {actual_count}"
        )


def _run_single_pattern_case(
    local_rank: int,
    case_name: str,
    vllm_config: VllmConfig,
    tp_size: int,
    batch_size: int = 8,
    seq_len: int = 16,
    hidden_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    eps: float = 1e-5,
) -> None:
    case = PATTERN_TEST_CASES[case_name]
    sp_moe_pass = SequenceParallelismMoePass(vllm_config)
    backend = CompileTestBackend(custom_passes=[sp_moe_pass])
    model = case.model_factory(
        hidden_size=hidden_size,
        dtype=dtype,
        eps=eps,
        device=f"npu:{local_rank}",
    )
    inputs = case.input_builder(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        dtype=dtype,
        tp_size=tp_size,
    )
    for dynamic_input_index in case.dynamic_input_indices:
        torch._dynamo.mark_dynamic(inputs[dynamic_input_index], 0)

    unfused = model(*inputs)
    compiled = torch.compile(model, backend=backend)
    fused = compiled(*inputs)
    assert unfused.shape == fused.shape

    assert sp_moe_pass.matched_count == 1
    _assert_op_counts(backend, case.pre_pass_expected_counts_factory(), before=True)
    _assert_op_counts(backend, case.post_pass_expected_counts_factory())


def _run_sequence_parallelism_moe_test(
    local_rank: int,
    world_size: int,
    master_port: int,
    command_queue: Any,
    result_queue: Any,
    batch_size: int = 8,
    seq_len: int = 16,
    hidden_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    eps: float = 1e-5,
) -> None:
    torch.npu.set_device(local_rank)
    torch.set_default_device(f"npu:{local_rank}")
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
        }
    )

    vllm_config = VllmConfig(model_config=ModelConfig(dtype=dtype))

    try:
        with vllm.config.set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=world_size,
                rank=local_rank,
                local_rank=local_rank,
                backend="hccl",
            )
            initialize_model_parallel(tensor_model_parallel_size=world_size)

            if not enable_custom_op():
                raise RuntimeError("vllm_ascend custom ops are not available")

            _ = get_tp_group().unique_name
            tp_size = get_tensor_model_parallel_world_size()
            result_queue.put((WORKER_READY, local_rank, "ok", ""))

            while True:
                case_name = command_queue.get()
                if case_name == WORKER_STOP:
                    return

                try:
                    _run_single_pattern_case(
                        local_rank=local_rank,
                        case_name=case_name,
                        vllm_config=vllm_config,
                        tp_size=tp_size,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        hidden_size=hidden_size,
                        dtype=dtype,
                        eps=eps,
                    )
                except Exception:
                    result_queue.put((case_name, local_rank, "error", traceback.format_exc()))
                else:
                    result_queue.put((case_name, local_rank, "ok", ""))
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _worker_entrypoint(
    local_rank: int,
    world_size: int,
    master_port: int,
    command_queue: Any,
    result_queue: Any,
) -> None:
    try:
        _run_sequence_parallelism_moe_test(
            local_rank=local_rank,
            world_size=world_size,
            master_port=master_port,
            command_queue=command_queue,
            result_queue=result_queue,
        )
    except Exception:
        result_queue.put((WORKER_READY, local_rank, "error", traceback.format_exc()))


def _wait_for_worker_reports(
    result_queue: Any,
    case_name: str,
    expected_reports: int,
) -> None:
    errors = []
    for _ in range(expected_reports):
        try:
            reported_case_name, local_rank, status, payload = result_queue.get(timeout=WORKER_RESULT_TIMEOUT_S)
        except queue.Empty as exc:
            raise TimeoutError(f"Timed out waiting for worker reports for {case_name}") from exc

        assert reported_case_name == case_name, f"Expected worker report for {case_name}, but got {reported_case_name}"
        if status != "ok":
            errors.append(f"rank {local_rank}:\n{payload}")

    if errors:
        raise AssertionError("\n\n".join(errors))


@pytest.fixture(scope="module")
def sequence_parallelism_moe_workers() -> Generator[Callable[[str], None], None, None]:
    ctx = torch.multiprocessing.get_context("spawn")
    command_queues = [ctx.Queue() for _ in range(WORLD_SIZE)]
    result_queue = ctx.Queue()
    workers = []

    for local_rank in range(WORLD_SIZE):
        worker = ctx.Process(
            target=_worker_entrypoint,
            args=(local_rank, WORLD_SIZE, MASTER_PORT, command_queues[local_rank], result_queue),
        )
        worker.start()
        workers.append(worker)

    try:
        _wait_for_worker_reports(result_queue, WORKER_READY, WORLD_SIZE)

        def _run_case(case_name: str) -> None:
            for command_queue in command_queues:
                command_queue.put(case_name)
            _wait_for_worker_reports(result_queue, case_name, WORLD_SIZE)

        yield _run_case
    finally:
        for command_queue in command_queues:
            command_queue.put(WORKER_STOP)
        for worker in workers:
            worker.join(timeout=WORKER_JOIN_TIMEOUT_S)
            if worker.is_alive():
                worker.terminate()
                worker.join()


@pytest.mark.parametrize("case_name", tuple(PATTERN_TEST_CASES), ids=tuple(PATTERN_TEST_CASES))
def test_sequence_parallelism_moe_patterns(
    sequence_parallelism_moe_workers: Callable[[str], None], case_name: str
) -> None:
    sequence_parallelism_moe_workers(case_name)
