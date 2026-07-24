#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""
Service profiling configuration generator module.

This module generates the service_profiling_symbols.yaml configuration file
to ~/.config/vllm_ascend/ directory.
"""

# ruff: noqa: E501

import contextlib
import tempfile
from pathlib import Path

import vllm
from vllm.logger import logger

VLLM_VERSION = vllm.__version__
# Configuration file name
CONFIG_FILENAME = f"service_profiling_symbols.{VLLM_VERSION}.yaml"

# Hard-coded YAML content, default symbols changed by user can be added here.
SERVICE_PROFILING_SYMBOLS_YAML = """
# ===== OpenAI API entry =====

- symbol: vllm.entrypoints.openai.chat_completion.serving:OpenAIServingChat.create_chat_completion
  domain: OpenAI
  name: Chat.create_chat_completion

# ===== Batch / Scheduler =====

- symbol: vllm.v1.core.sched.scheduler:Scheduler.schedule
  min_version: "0.9.1"
  handler: ms_service_profiler.patcher.vllm.handlers.v1.batch_handlers:schedule
  name: batchFrameworkProcessing

- symbol: vllm_ascend.core.scheduler:AscendScheduler.schedule
  min_version: "0.9.1"
  handler: ms_service_profiler.patcher.vllm.handlers.v1.batch_handlers:schedule
  name: batchFrameworkProcessing

- symbol: vllm.v1.core.sched.scheduler:Scheduler.add_request
  min_version: "0.9.1"
  handler: ms_service_profiler.patcher.vllm.handlers.v1.batch_handlers:add_request

# ===== KV Cache =====
- symbol: vllm.v1.core.kv_cache_manager:KVCacheManager.free
  min_version: "0.9.1"
  handler: ms_service_profiler.patcher.vllm.handlers.v1.kvcache_handlers:free

- symbol: vllm.v1.core.kv_cache_manager:KVCacheManager.get_computed_blocks
  min_version: "0.9.1"
  handler: ms_service_profiler.patcher.vllm.handlers.v1.kvcache_handlers:get_computed_blocks

# ===== Model Execute =====
- symbol: vllm.model_executor.layers.logits_processor:LogitsProcessor.forward
  min_version: "0.9.1"
  handler: ms_service_profiler.patcher.vllm.handlers.v1.model_handlers:compute_logits
  name: computing_logits

- symbol: vllm.v1.sample.sampler:Sampler.forward
  min_version: "0.9.1"
  handler: ms_service_profiler.patcher.vllm.handlers.v1.model_handlers:sampler_forward
  name: sample

- symbol: vllm.v1.executor.abstract:Executor.execute_model
  min_version: "0.9.1"
  handler: ms_service_profiler.patcher.vllm.handlers.v1.model_handlers:execute_model
  name: modelExec

- symbol: vllm.v1.executor.multiproc_executor:MultiprocExecutor.execute_model
  min_version: "0.9.1"
  handler: ms_service_profiler.patcher.vllm.handlers.v1.model_handlers:execute_model
  name: modelExec

- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner.execute_model
  name: modelRunnerExec
  handler: ms_service_profiler.patcher.vllm.handlers.v1.model_handlers:execute_model_runner
  domain: Execute

- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner._update_states
  name: _update_states
  domain: Execute

- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner._prepare_inputs
  name: _prepare_inputs
  domain: Execute

- symbol: "vllm.model_executor.models.*:*.embed_multimodal"
  name: multimodalEmbedding
  domain: Multimodal

- symbol: vllm_ascend.utils:ProfileExecuteDuration.capture_async
  min_version: "0.9.1"
  max_version: "0.14.0rc1"
  handler: ms_service_profiler.patcher.vllm.handlers.v1.model_handlers:capture_async

- symbol: vllm.v1.utils:record_function_or_nullcontext
  min_version: "0.15.0rc1"
  handler: ms_service_profiler.patcher.vllm.handlers.v1.model_handlers:record_function_or_nullcontext

# ===== MTP / NPUModelRunner =====
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner.propose_draft_token_ids
  min_version: "0.9.1"
  handler: ms_service_profiler.patcher.vllm.handlers.v1.mtp_handlers:propose_draft_token_ids_npu

- symbol: vllm_ascend.sample.rejection_sampler:rejection_sample
  min_version: "0.9.1"
  handler: ms_service_profiler.patcher.vllm.handlers.v1.mtp_handlers:capture_rejection_output

# ===== Request Lifecycle =====
- symbol: vllm.v1.engine.async_llm:AsyncLLM.add_request
  min_version: "0.9.1"
  handler: ms_service_profiler.patcher.vllm.handlers.v1.request_handlers:add_request_async

- symbol: vllm.engine.async_llm_engine:AsyncLLMEngine.add_request
  min_version: "0.9.1"
  max_version: "0.11.0"
  handler: ms_service_profiler.patcher.vllm.handlers.v1.request_handlers:add_request_async

- symbol: vllm.v1.engine.output_processor:OutputProcessor.process_outputs
  min_version: "0.9.1"
  handler: ms_service_profiler.patcher.vllm.handlers.v1.request_handlers:process_outputs

# ===== Meta =====

- symbol: vllm.v1.engine.core:DPEngineCoreProc.add_request
  min_version: "0.9.1"
  handler: ms_service_profiler.patcher.vllm.handlers.v1.meta_handlers:init_data_parallel

- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner.execute_model
  min_version: "0.9.1"
  handler: ms_service_profiler.patcher.vllm.handlers.v1.meta_handlers:init_data_parallel_worker

# ===== Extended OpenAI-to-ops trace points =====

- symbol: vllm.entrypoints.openai.completion.serving:OpenAIServingCompletion.create_completion
  domain: OpenAI
  name: Completion.create_completion
- symbol: vllm.v1.engine.async_llm:AsyncLLM.generate
  min_version: "0.9.1"
  domain: Request
  name: AsyncLLM.generate
- symbol: vllm.v1.engine.core:EngineCore.add_request
  min_version: "0.9.1"
  domain: Engine
  name: EngineCore.add_request
- symbol: vllm.v1.engine.core:EngineCore.step
  min_version: "0.9.1"
  domain: Engine
  name: EngineCore.step
- symbol: vllm.v1.engine.core:EngineCore.step_with_batch_queue
  min_version: "0.9.1"
  domain: Engine
  name: EngineCore.step_with_batch_queue
- symbol: vllm_ascend.core.recompute_scheduler:RecomputeScheduler.schedule
  min_version: "0.9.1"
  domain: Scheduler
  name: RecomputeScheduler.schedule
  attributes:
  - name: req_ids
    expr: return | attr num_scheduled_tokens | str
  - name: num_scheduled_tokens
    expr: return | attr num_scheduled_tokens | str
- symbol: vllm_ascend.core.recompute_scheduler:AsyncRecomputeScheduler.schedule
  min_version: "0.9.1"
  domain: Scheduler
  name: AsyncRecomputeScheduler.schedule
  attributes:
  - name: req_ids
    expr: return | attr num_scheduled_tokens | str
  - name: num_scheduled_tokens
    expr: return | attr num_scheduled_tokens | str
- symbol: vllm_ascend.core.scheduler_dynamic_batch:SchedulerDynamicBatch.schedule
  min_version: "0.9.1"
  domain: Scheduler
  name: SchedulerDynamicBatch.schedule
  attributes:
  - name: req_ids
    expr: return | attr num_scheduled_tokens | str
  - name: num_scheduled_tokens
    expr: return | attr num_scheduled_tokens | str
- symbol: vllm_ascend.patch.platform.patch_balance_schedule:BalanceScheduler.schedule
  min_version: "0.9.1"
  domain: Scheduler
  name: BalanceScheduler.schedule
  attributes:
  - name: req_ids
    expr: return | attr num_scheduled_tokens | str
  - name: num_scheduled_tokens
    expr: return | attr num_scheduled_tokens | str
- symbol: vllm.v1.executor.uniproc_executor:UniProcExecutor.execute_model
  min_version: "0.9.1"
  handler: ms_service_profiler.patcher.vllm.handlers.v1.model_handlers:execute_model
  domain: Execute
  name: modelExec
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner._calc_spec_decode_metadata
  min_version: "0.9.1"
  domain: SpecDecode
  name: NPUModelRunner._calc_spec_decode_metadata
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner._build_attn_state
  min_version: "0.9.1"
  domain: SpecDecode
  name: NPUModelRunner._build_attn_state
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner._determine_batch_execution_and_padding
  min_version: "0.9.1"
  domain: Execute
  name: NPUModelRunner._determine_batch_execution_and_padding
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner._pad_for_sequence_parallelism
  min_version: "0.9.1"
  domain: Execute
  name: NPUModelRunner._pad_for_sequence_parallelism
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner._sync_metadata_across_dp
  min_version: "0.9.1"
  domain: Execute
  name: NPUModelRunner._sync_metadata_across_dp
- symbol: vllm.v1.cudagraph_dispatcher:CudagraphDispatcher.dispatch
  min_version: "0.9.1"
  domain: Execute
  name: CudagraphDispatcher.dispatch
- symbol: vllm_ascend.worker.model_runner_v1:_post_process_cudagraph_mode
  min_version: "0.9.1"
  domain: Execute
  name: _post_process_cudagraph_mode
- symbol: vllm_ascend.utils:should_skip_allreduce_across_dp_group
  min_version: "0.9.1"
  domain: Execute
  name: should_skip_allreduce_across_dp_group
- symbol: vllm_ascend.spec_decode.llm_base_proposer:AscendSpecDecodeBaseProposer._propose
  min_version: "0.9.1"
  domain: SpecDecode
  name: draft_propose
- symbol: vllm_ascend.spec_decode.llm_base_proposer:AscendSpecDecodeBaseProposer._run_merged_draft
  min_version: "0.9.1"
  domain: SpecDecode
  name: draft_model_forward
- symbol: vllm_ascend.spec_decode.llm_base_proposer:AscendSpecDecodeBaseProposer.maybe_all_gather_and_unpad
  min_version: "0.9.1"
  domain: SpecDecode
  name: draft_hidden_states_all_gather
- symbol: vllm_ascend.spec_decode.llm_base_proposer:AscendSpecDecodeBaseProposer.compute_draft_token_ids
  min_version: "0.9.1"
  domain: SpecDecode
  name: draft_compute_token_ids
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner._build_attention_metadata
  min_version: "0.9.1"
  domain: Execute
  name: NPUModelRunner._build_attention_metadata
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner._sanitize_placeholder_input_ids_for_forward
  min_version: "0.9.1"
  domain: SpecDecode
  name: NPUModelRunner._sanitize_placeholder_input_ids_for_forward
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner._preprocess
  min_version: "0.9.1"
  domain: Execute
  name: NPUModelRunner._preprocess
- symbol: vllm_ascend.ops.rotary_embedding:update_cos_sin
  min_version: "0.9.1"
  domain: Execute
  name: rotary.update_cos_sin
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner._model_forward
  min_version: "0.9.1"
  domain: ModelForward
  name: NPUModelRunner._model_forward
  attributes:
  - name: req_ids
    expr: this | attr input_batch | attr req_ids | str
  - name: dp_rank
    expr: this | attr dp_rank | str
  - name: npu_id
    expr: this | attr device | attr index | str
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner._update_full_graph_params_if_needed
  min_version: "0.9.1"
  domain: ModelForward
  name: NPUModelRunner._update_full_graph_params_if_needed
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner._all_gather_hidden_states_and_aux
  min_version: "0.9.1"
  domain: ModelForward
  name: NPUModelRunner._all_gather_hidden_states_and_aux
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner._dummy_run
  min_version: "0.9.1"
  domain: Execute
  name: NPUModelRunner._dummy_run
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner.sample_tokens
  min_version: "0.9.1"
  domain: Sample
  name: NPUModelRunner.sample_tokens
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner._sample
  min_version: "0.9.1"
  domain: Sample
  name: NPUModelRunner._sample
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner._bookkeeping_sync
  min_version: "0.9.1"
  domain: Execute
  name: NPUModelRunner._bookkeeping_sync
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner._poll_late_kv_connector_output
  min_version: "0.9.1"
  domain: KVTransfer
  name: NPUModelRunner._poll_late_kv_connector_output
- symbol: vllm.v1.worker.gpu_model_runner:GPUModelRunner._compute_cascade_attn_prefix_lens
  min_version: "0.9.1"
  domain: Execute
  name: GPUModelRunner._compute_cascade_attn_prefix_lens
- symbol: vllm.v1.worker.gpu_model_runner:GPUModelRunner._execute_mm_encoder
  min_version: "0.9.1"
  domain: Multimodal
  name: GPUModelRunner._execute_mm_encoder
- symbol: vllm.v1.worker.gpu_model_runner:GPUModelRunner._pool
  min_version: "0.9.1"
  domain: Execute
  name: GPUModelRunner._pool
- symbol: vllm.v1.worker.gpu_model_runner:GPUModelRunner._update_states_after_model_execute
  min_version: "0.9.1"
  domain: Execute
  name: GPUModelRunner._update_states_after_model_execute
- symbol: vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector:MooncakeConnectorScheduler.get_num_new_matched_tokens
  min_version: "0.9.1"
  domain: KVTransfer
  name: MooncakeScheduler.matchRemoteKV
  attributes:
  - name: req_id
    expr: args[1] | attr request_id | str
  - name: kv_transfer_params
    expr: args[1] | attr kv_transfer_params | str
  - name: matched
    expr: return | str
- symbol: vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector:MooncakeConnectorScheduler.update_state_after_alloc
  min_version: "0.9.1"
  domain: KVTransfer
  name: MooncakeScheduler.updateAfterAlloc
  attributes:
  - name: req_id
    expr: args[1] | attr request_id | str
  - name: num_external_tokens
    expr: args[3] | str
  - name: kv_transfer_params
    expr: args[1] | attr kv_transfer_params | str
- symbol: vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector:MooncakeConnectorScheduler.build_connector_meta
  min_version: "0.9.1"
  domain: KVTransfer
  name: MooncakeScheduler.buildMeta
  attributes:
  - name: req_ids
    expr: return | attr requests | str
  - name: reqs_in_batch
    expr: return | attr reqs_in_batch | str
  - name: requests_to_send
    expr: return | attr requests_to_send | str
- symbol: vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector:MooncakeConnector.start_load_kv
  min_version: "0.9.1"
  domain: KVTransfer
  name: MooncakeConnector.startLoadKV
  attributes:
  - name: req_ids
    expr: this | attr _connector_metadata | attr requests | str
  - name: reqs_in_batch
    expr: this | attr _connector_metadata | attr reqs_in_batch | str
- symbol: vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector:MooncakeConnectorWorker.start_load_kv
  min_version: "0.9.1"
  domain: KVTransfer
  name: MooncakeWorker.startLoadKV
  attributes:
  - name: req_ids
    expr: args[1] | attr requests | str
  - name: reqs_in_batch
    expr: args[1] | attr reqs_in_batch | str
  - name: tp_rank
    expr: this | attr tp_rank | str
  - name: kv_role
    expr: this | attr kv_role | str
- symbol: vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector:KVCacheRecvingThread.add_request
  min_version: "0.9.1"
  domain: KVTransfer
  name: MooncakeRecvThread.enqueue
  attributes:
  - name: req_id
    expr: kwargs['request_id'] | str
  - name: remote_req_id
    expr: kwargs['remote_request_id'] | str
  - name: remote_engine_id
    expr: kwargs['remote_engine_id'] | str
  - name: remote_host
    expr: kwargs['remote_host'] | str
  - name: remote_handshake_port
    expr: kwargs['remote_handshake_port'] | str
  - name: offset
    expr: kwargs['offset'] | str
  - name: tp_rank
    expr: this | attr tp_rank | str
  - name: engine_id
    expr: this | attr local_engine_id | str
- symbol: vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector:KVCacheRecvingThread._transfer_kv_cache
  min_version: "0.9.1"
  domain: Execute
  name: MooncakeSyncRead
  attributes:
  - name: req_id
    expr: args[1]['remote_request_id'] | str
  - name: tp_rank
    expr: this | attr tp_rank | str
  - name: engine_id
    expr: this | attr local_engine_id | str
- symbol: vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector:KVCacheTaskTracker.update_done_task_count
  min_version: "0.9.1"
  domain: KVTransfer
  name: MooncakeTaskDone
  attributes:
  - name: req_id
    expr: args[1] | str
  - name: finished_req_ids
    expr: this | attr finished_requests | str
  - name: reqs_to_process
    expr: this | attr reqs_to_process | str
- symbol: vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector:KVCacheTaskTracker.get_and_clear_finished_requests
  min_version: "0.9.1"
  domain: KVTransfer
  name: MooncakeTaskFinishedPoll
  attributes:
  - name: req_ids
    expr: return | str
- symbol: vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector:KVCacheRecvingThread.get_and_clear_finished_requests
  min_version: "0.9.1"
  domain: KVTransfer
  name: MooncakeRecvFinishedPoll
  attributes:
  - name: req_ids
    expr: return | str
  - name: tp_rank
    expr: this | attr tp_rank | str
  - name: engine_id
    expr: this | attr local_engine_id | str
- symbol: vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector:MooncakeConnectorWorker.get_finished
  min_version: "0.9.1"
  domain: KVTransfer
  name: MooncakeWorkerGetFinished
  attributes:
  - name: req_ids
    expr: return | str
  - name: finished
    expr: return | str
  - name: tp_rank
    expr: this | attr tp_rank | str
  - name: kv_role
    expr: this | attr kv_role | str
- symbol: vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector:MooncakeConnector.get_finished
  min_version: "0.9.1"
  domain: KVTransfer
  name: MooncakeConnector.get_finished
  attributes:
  - name: req_ids
    expr: return | str
  - name: finished
    expr: return | str
  - name: scheduler_finished_req_ids
    expr: args[1] | str
- symbol: vllm.distributed.kv_transfer.kv_connector.utils:KVOutputAggregator.aggregate
  min_version: "0.9.1"
  domain: KVTransfer
  name: KVOutputAggregator.aggregate
  attributes:
  - name: req_ids
    expr: return | attr kv_connector_output | attr finished_recving | str
  - name: final_finished_sending
    expr: return | attr kv_connector_output | attr finished_sending | str
  - name: final_finished_recving
    expr: return | attr kv_connector_output | attr finished_recving | str
  - name: recv_remaining_count
    expr: this | attr _recv_remaining_count | str
  - name: send_remaining_count
    expr: this | attr _send_remaining_count | str
  - name: expected_finished_count
    expr: this | attr _expected_finished_count | str
- symbol: vllm.v1.core.sched.scheduler:Scheduler._update_from_kv_xfer_finished
  min_version: "0.9.1"
  domain: KVTransfer
  name: Scheduler.updateKVXferFinished
  attributes:
  - name: req_ids
    expr: args[1] | attr finished_recving | str
  - name: finished_recving
    expr: args[1] | attr finished_recving | str
  - name: finished_sending
    expr: args[1] | attr finished_sending | str
  - name: scheduler_finished_recving
    expr: this | attr finished_recving_kv_req_ids | str
  - name: scheduler_failed_recving
    expr: this | attr failed_recving_kv_req_ids | str
- symbol: vllm.v1.core.sched.scheduler:Scheduler._try_promote_blocked_waiting_request
  min_version: "0.9.1"
  domain: Schedule
  name: Scheduler.tryPromoteBlockedWaiting
  attributes:
  - name: req_id
    expr: args[1] | attr request_id | str
  - name: req_status
    expr: args[1] | attr status | str
  - name: promoted
    expr: return | str
  - name: scheduler_finished_recving
    expr: this | attr finished_recving_kv_req_ids | str
"""


def get_config_dir() -> Path:
    """
    Get the vllm_ascend configuration directory path.

    Returns:
        Path: The path to ~/.config/vllm_ascend/ directory.
    """
    home_dir = Path.home()
    config_dir = home_dir / ".config" / "vllm_ascend"
    return config_dir


def _cleanup_temp_file(tmp_path: Path | None) -> None:
    """
    Clean up a temporary file if it exists.

    Args:
        tmp_path: Path to the temporary file to clean up.
    """
    if tmp_path is not None and tmp_path.exists():
        with contextlib.suppress(OSError):
            tmp_path.unlink()


def generate_service_profiling_config() -> Path | None:
    """
    Generate the service_profiling_symbols.yaml configuration file
    to ~/.config/vllm_ascend/ directory.

    If the configuration file already exists, this function will skip
    creating it and return the existing file path.

    If any error occurs during file creation, it will be logged but
    will not interrupt the execution. The function will return None
    to indicate that the file could not be created.

    Returns:
        Optional[Path]: The path to the generated (or existing) configuration file.
                       Returns None if file creation failed.
    """
    config_dir = get_config_dir()
    config_file = config_dir / CONFIG_FILENAME

    # Check if the configuration file already exists
    if config_file.exists():
        return config_file

    # Create the configuration directory if it doesn't exist
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        logger.exception("Failed to create configuration directory %s: %s", config_dir, e)
        return None

    # Write the configuration file atomically using a temporary file
    # This ensures the file is only written if the write succeeds completely
    tmp_path = None
    try:
        # Create a temporary file in the same directory for atomic write
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=config_dir, delete=False, suffix=".tmp", prefix=CONFIG_FILENAME + "."
        ) as tmp_file:
            tmp_file.write(SERVICE_PROFILING_SYMBOLS_YAML)
            tmp_path = Path(tmp_file.name)

        # Atomically replace the target file with the temporary file
        tmp_path.replace(config_file)
        return config_file
    except (OSError, PermissionError) as e:
        logger.exception("Failed to write configuration file %s: %s", config_file, e)
        return None
    finally:
        # Clean up the temporary file if it wasn't successfully replaced
        _cleanup_temp_file(tmp_path)
