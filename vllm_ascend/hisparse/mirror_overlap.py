# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Overlap HiSparse Host mirroring with Target sampling and guarded MTP."""

from __future__ import annotations

import torch


class SamplingMirrorOverlapConnector:
    """Let Target logits and sampling run before joining the Host mirror."""

    def __init__(self, connector, worker, output_copy_stream, logger, rank, snapshot=None):
        self._connector = connector
        self._worker = worker
        self._output_copy_stream = output_copy_stream
        self._logger = logger
        self._rank = rank
        self._snapshot = snapshot
        self._overlap_draft = False
        self._pending_event = None
        self._allow_step = False
        self._eligible_forward = False
        self._batch_size = 0
        self._logged_batch_sizes: set[int] = set()
        self._logged_snapshot_sizes: set[int] = set()
        self._logged_direct_sizes: set[int] = set()

    def __getattr__(self, name):
        return getattr(self._connector, name)

    def join_pending(self) -> None:
        if self._pending_event is not None:
            torch.accelerator.current_stream().wait_event(self._pending_event)
            self._pending_event = None
        self._overlap_draft = False

    def begin_execution(self, allow_step: bool) -> None:
        # The runner calls this before preemption or any scheduler state writes.
        self.join_pending()
        self._allow_step = allow_step
        self._eligible_forward = False

    def pre_forward(self, scheduler_output, input_batch=None, attn_metadata=None):
        self._eligible_forward = (
            self._allow_step
            and scheduler_output.has_sync_kv_loads
            and input_batch is not None
            and attn_metadata is not None
            and not input_batch.has_prefill
            and 0 < input_batch.num_tokens == input_batch.num_tokens_after_padding
        )
        if self._eligible_forward:
            self._batch_size = input_batch.num_reqs
        return self._connector.pre_forward(scheduler_output, input_batch, attn_metadata=attn_metadata)

    def finish_forward(self) -> None:
        eligible = self._eligible_forward
        self._eligible_forward = False
        if not eligible or self._connector._disabled:
            return self._connector.finish_forward()

        worker = self._worker
        compute_stream = torch.accelerator.current_stream()
        worker._forward_ready_event.record(compute_stream)
        snapshot = self._snapshot
        pending_before = 0
        if snapshot is not None:
            pending_before = len(worker._pending_dma_descriptors)
            snapshot.submitted_completion = None
            snapshot.armed = (
                not worker._dma_submitted and not worker._per_layer_mirrored and not worker._submitted_mirror_layers
            )
        try:
            worker._finish_mirror_phase(worker._forward_ready_event)
        except BaseException:
            # The runner's failure boundary joins an already-recorded copy.
            if worker._dma_submitted:
                self._pending_event = worker.host_write_event
            raise
        finally:
            if snapshot is not None:
                snapshot.armed = False
        if worker._dma_submitted:
            self._pending_event = worker.host_write_event
            # Preserve the Host durability fence on AsyncOutput's copy stream.
            self._output_copy_stream.wait_event(self._pending_event)
            worker._dma_submitted = False
            if snapshot is not None and snapshot.submitted_completion is not None:
                pending = worker._pending_dma_descriptors
                # Direct fallback or an extra transfer keeps the sampling-only fence.
                self._overlap_draft = (
                    len(pending) == pending_before + 1 and pending[-1][0] is snapshot.submitted_completion
                )
                if self._overlap_draft and self._batch_size not in self._logged_snapshot_sizes:
                    self._logged_snapshot_sizes.add(self._batch_size)
                    self._logger.info(
                        "HiSparse Target snapshot used: rank=%d requests=%d", self._rank, self._batch_size
                    )
            if snapshot is not None and not self._overlap_draft and self._batch_size not in self._logged_direct_sizes:
                self._logged_direct_sizes.add(self._batch_size)
                self._logger.info(
                    "HiSparse Target snapshot fallback: rank=%d requests=%d", self._rank, self._batch_size
                )
            if self._batch_size not in self._logged_batch_sizes:
                self._logged_batch_sizes.add(self._batch_size)
                self._logger.info("HiSparse MTP mirror overlap used: rank=%d requests=%d", self._rank, self._batch_size)
        else:
            worker.host_write_event.record(compute_stream)
        worker._release_completed_dma_descriptors()

    def post_forward(self, finished_req_ids, defer_finalize=False):
        if not defer_finalize or not self._overlap_draft:
            self.join_pending()
        return self._connector.post_forward(finished_req_ids, defer_finalize=defer_finalize)

    def finalize_step(self):
        self.join_pending()
        return self._connector.finalize_step()

    def no_forward(self, scheduler_output):
        self.begin_execution(False)
        return self._connector.no_forward(scheduler_output)

    def set_disabled(self, disabled: bool) -> None:
        self.begin_execution(False)
        self._connector.set_disabled(disabled)

    def reset_capture_state(self) -> None:
        self.begin_execution(False)
        self._connector.reset_capture_state()


def overlap_mtp_host_mirror(runner):
    """Bind only the audited single-stage GLM MTP sampling path."""
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import HiSparseConnector
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
    from vllm.logger import init_logger
    from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

    from vllm_ascend.hisparse.backend import AscendHiSparseBackend
    from vllm_ascend.hisparse.target_snapshot import TargetMirrorSnapshot

    connector = runner.kv_connector
    config = runner.vllm_config.parallel_config
    if (
        type(connector) is not ActiveKVConnector
        or type(connector.kv_connector) is not HiSparseConnector
        or runner.speculator is None
        or runner.speculative_config.method != "mtp"
        or runner.model_config.architecture != "GlmMoeDsaForCausalLM"
        or config.pipeline_parallel_size != 1
        or config.data_parallel_size != 1
        or config.decode_context_parallel_size != 1
        or runner.pcp_manager is not None
        or runner.is_pooling_model
        or runner.supports_mm_inputs
        or runner.aux_output_connector is not None
        or runner.kvpp.scheduler is not None
        or getattr(runner, "pd_dspark_aux_layer_ids", ())
    ):
        return connector
    worker = connector.kv_connector.connector_worker
    if (
        worker is None
        or not worker._initialized
        or not isinstance(worker.backend, AscendHiSparseBackend)
        or worker.shared_host_region is not None
        or not worker.is_host_writer
    ):
        return connector
    output = runner.output_copy_stream
    output_stream = torch.Stream(
        stream_id=output.stream_id, device_index=output.device_index, device_type=output.device_type
    )
    if output_stream == torch.accelerator.current_stream():
        return connector
    logger = init_logger(__name__)
    rank = get_tensor_model_parallel_rank()
    snapshot = TargetMirrorSnapshot.bind(worker, runner.speculator)
    worker._ascend_target_snapshot = snapshot
    logger.info("HiSparse MTP mirror overlap enabled: rank=%d", rank)
    if snapshot is not None:
        logger.info("HiSparse Target snapshot enabled: rank=%d", rank)
    return SamplingMirrorOverlapConnector(connector, worker, output_stream, logger, rank, snapshot)
