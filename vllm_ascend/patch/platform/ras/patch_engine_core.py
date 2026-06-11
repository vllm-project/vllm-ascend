import copy
import signal
import time
import threading
import uuid
from contextlib import ExitStack
from typing import cast

import msgspec.msgpack
import zmq
from vllm import pooling_params
from vllm.config import ParallelConfig, VllmConfig
from vllm.logger import logger
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils.system_utils import set_process_title
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutputs, EngineCoreRequest, EngineCoreRequestType
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc, EngineShutdownState
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder
from vllm.utils.network_utils import make_zmq_socket
from vllm_ascend.recovery.engine_core_recovery_handler import RecoveryHandler
from vllm_ascend.recovery.types import FUTURE_TIMEOUT_SECONDS, NetworkCheck
from vllm_ascend.recovery.utils import get_engine_recovery_bind_address


_RECOVERY_MSG_PREFIX = b"\x00REC"


class RasDPEngineCoreProc(DPEngineCoreProc):

    def __init__(self, *args, **kwargs):
        vllm_config: VllmConfig = kwargs["vllm_config"]

        recover_step_xpub_addr, recover_report_pull_addr, \
            recover_step_result_pull_addr = \
            get_engine_recovery_bind_address(
                vllm_config.parallel_config.data_parallel_rank
            )

        self._recovery_handler = RecoveryHandler(
            engine_core=None,
            vllm_config=vllm_config,
        )

        logger.info(
            "[RAS] RecoveryHandler bind recovery_addrs=%s",
            {
                "recover_step_xpub_addr": recover_step_xpub_addr,
                "recover_report_pull_addr": recover_report_pull_addr,
                "recover_step_result_pull_addr": recover_step_result_pull_addr,
            }
        )
        self._recovery_handler.setup_recover_sockets(
            recover_step_xpub_addr,
            recover_report_pull_addr,
            recover_step_result_pull_addr,
        )

        super().__init__(*args, **kwargs)
        self._recovery_handler._engine_core = self
        self._recovery_handler.start()
        self._recovery_handler.wait_for_worker_subscriptions()

    def process_input_sockets(
        self,
        input_addresses,
        coord_input_address,
        identity,
        ready_event,
    ):
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        input_address,
                        zmq.DEALER,
                        identity=identity,
                        bind=False,
                    )
                )
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        coord_input_address,
                        zmq.XSUB,
                        identity=identity,
                        bind=False,
                    )
                )
                coord_socket.send(b"\x01")

            poller = zmq.Poller()
            for input_socket in input_sockets:
                input_socket.send(b"")
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)

                buffer = coord_socket.recv()
                if buffer.startswith(_RECOVERY_MSG_PREFIX):
                    payload = buffer[len(_RECOVERY_MSG_PREFIX) :]
                    try:
                        msg = msgspec.msgpack.decode(payload)
                        if msg[0] == "RECOVERY_ADDRESSES":
                            _, sub_addr, push_addr = msg
                            logger.info(
                                "[RAS][engine=%d] Received recovery addresses from "
                                "dp coordinator: sub=%s, push=%s",
                                self.dp_rank,
                                sub_addr,
                                push_addr,
                            )
                            self._recovery_handler.connect_coordinator(
                                sub_addr, push_addr
                            )
                    except Exception as e:
                        logger.exception(
                            "[RAS][engine=%d] Failed to deserialize recovery addr msg: %s",
                            self.dp_rank, e
                        )

            ready_event.set()
            del ready_event

            while True:
                for input_socket, _ in poller.poll():
                    type_frame, *data_frames = input_socket.recv_multipart(
                        copy=False
                    )

                    if (
                        input_socket == coord_socket
                        and type_frame.buffer == b"READY"
                    ):
                        continue

                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    request = None
                    if request_type == EngineCoreRequestType.ADD:
                        req = add_request_decoder.decode(data_frames)
                        try:
                            request = self.preprocess_add_request(req)
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                    else:
                        request = generic_decoder.decode(data_frames)

                        if request_type == EngineCoreRequestType.ABORT:
                            self.aborts_queue.put_nowait(request)

                    self.input_queue.put_nowait((request_type, request))

    def run_busy_loop(self):
        exception_occurred = False
        while self._handle_shutdown():
            if self._recovery_handler.is_recovering:
                self._wait_for_recovery()
                exception_occurred = True
                continue
            if exception_occurred:
                if self.batch_queue is not None:
                    while self.batch_queue:
                        future, _, _ = self.batch_queue.pop()
                        try:
                            logger.info("[RAS][engine=%d] main thread pop future", self.dp_rank)
                            future.result()
                        except Exception:
                            pass
                    self.batch_queue.clear()
                    logger.info("[RAS][engine=%d] batch_queue drained", self.dp_rank)
                scheduler = cast(Scheduler, self.scheduler)
                while scheduler.running:
                    request = scheduler.running.pop()
                    
                    scheduler.requests.pop(request.request_id, None)
                    old_blocks = scheduler.kv_cache_manager.coordinator.get_blocks(
                        request.request_id
                    )
                    old_block_ids = set()
                    for blocks_list in old_blocks:
                        for block in blocks_list:
                            old_block_ids.add(block.block_id)
                    
                    scheduler.kv_cache_manager.free(request)
                    scheduler.encoder_cache_manager.free(request)
                    if old_block_ids:
                        scheduler.kv_cache_manager.evict_blocks(old_block_ids)
                    
                    new_samping_param = copy.deepcopy(request.sampling_params)
                    num_decoded_tokens = len(request._output_token_ids)
                    new_samping_param.max_tokens -= num_decoded_tokens

                    new_engine_core_request = EngineCoreRequest(
                        request_id=request.request_id,
                        prompt_token_ids=request._all_token_ids.copy(),
                        mm_features=request.mm_features,
                        sampling_params=new_samping_param,
                        pooling_params=request.pooling_params,
                        arrival_time=request.arrival_time,
                        lora_request=request.lora_request,
                        cache_salt=request.cache_salt,
                        data_parallel_rank=None,
                        prompt_embeds = request.prompt_embeds,
                        client_index=request.client_index,
                        priority=request.priority,
                        trace_headers=request.trace_headers,
                        resumable=request.resumable,
                        reasoning_ended=None,
                    )
                    new_request, wave = self.preprocess_add_request(new_engine_core_request)
                    scheduler.requests[new_request.request_id] = new_request
                    scheduler.waiting.prepend_request(new_request)
                    scheduler.prev_step_scheduled_req_ids.discard(request.request_id)
            try:
                if not exception_occurred:
                    self._process_input_queue()
                exception_occurred = False
                if self.eep_scaling_state is not None:
                    _ = self.eep_scaling_state.progress()
                    if self.eep_scaling_state.is_complete():
                        self.process_input_queue_block = True
                        self.eep_scaling_state = None

                executed = self._process_engine_step()
                self._maybe_publish_request_counts()

                local_unfinished_reqs = self.scheduler.has_unfinished_requests()
                if not executed:
                    if not local_unfinished_reqs and not self.engines_running:
                        continue
                    self.execute_dummy_batch()

                self.engines_running = self._has_global_unfinished_reqs(
                    local_unfinished_reqs
                )

                if not self.engines_running:
                    if self.dp_rank == 0 or not self.has_coordinator:
                        logger.debug(
                            "Wave %d finished, pausing engine loop.",
                            self.current_wave,
                        )
                        client_index = -1 if self.has_coordinator else 0
                        self.output_queue.put_nowait(
                            (
                                client_index,
                                EngineCoreOutputs(
                                    wave_complete=self.current_wave
                                ),
                            )
                        )
                    self.current_wave += 1
                    self.step_counter = 0

            except Exception:
                if self._wait_for_recovery_on_exception():
                    exception_occurred = True
                    continue
                raise

        raise SystemExit

    def _wait_for_recovery(self) -> None:
        logger.info(
            "[RAS][engine=%d] EngineCore entering recovery wait (wave=%d)",
            self.dp_rank, self.current_wave
        )
        self._recovery_handler.wait_for_recovery()
        if self._recovery_handler.get_recovery_success():
            logger.info("[RAS][engine=%d] Recovery succeeded, resuming engine loop",
                        self.dp_rank)
        else:
            logger.error("[RAS][engine=%d] Recovery failed, raising exception",
                         self.dp_rank)
            raise RuntimeError("Recovery failed")

    def _wait_for_recovery_on_exception(self) -> bool:
        logger.info(
            "[RAS][engine=%d] EngineCore caught exception, waiting up to 5s for "
            "recovery signal...",
            self.dp_rank,
        )
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self._recovery_handler.is_recovering:
                logger.info(
                    "[RAS][engine=%d] Recovery signal received, suppressing exception",
                    self.dp_rank,
                )
                return True
            time.sleep(0.1)
        logger.info(
            "[RAS][engine=%d] No recovery signal within 5s, re-raising exception",
            self.dp_rank,
        )
        return False


def _patched_run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
    maybe_register_config_serialize_by_value()

    vllm_config: VllmConfig = kwargs["vllm_config"]
    parallel_config: ParallelConfig = vllm_config.parallel_config

    engine_core: EngineCoreProc | None = None
    try:
        data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
        if data_parallel:
            parallel_config.data_parallel_rank_local = local_dp_rank
            process_title = f"EngineCore_DP{dp_rank}"
        else:
            process_title = "EngineCore"
        set_process_title(process_title)

        if data_parallel and vllm_config.kv_transfer_config is not None:
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
            )

        parallel_config.data_parallel_index = dp_rank
        if data_parallel and vllm_config.model_config.is_moe:
            parallel_config.data_parallel_rank = dp_rank
            engine_core = RasDPEngineCoreProc(*args, **kwargs)
        else:
            parallel_config.data_parallel_size = 1
            parallel_config.data_parallel_size_local = 1
            parallel_config.data_parallel_rank = 0
            engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

        def signal_handler(signum, frame):
            engine_core.shutdown_state = EngineShutdownState.REQUESTED

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core.run_busy_loop()

    except SystemExit:
        raise
    except Exception as e:
        if engine_core is None:
            logger.exception("EngineCore failed to start.")
        else:
            logger.exception("EngineCore encountered a fatal error.")
            engine_core._send_engine_dead()
        raise e
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """

        batch_queue = self.batch_queue
        assert batch_queue is not None

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule()
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
            if self.is_ec_consumer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if self.is_pooling_model or not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                batch_queue.appendleft((future, scheduler_output, exec_future))
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, True

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            def _on_future_timeout():
                logger.warning(
                    "[RAS][engine=%d] future.result() timed out after %ds, "
                    "sending NetworkCheck to Coordinator",
                    self.dp_rank, FUTURE_TIMEOUT_SECONDS,
                )
                network_check = NetworkCheck(engine_index=self.dp_rank)
                network_encode = msgspec.msgpack.encode(("networkcheck", network_check))
                self._recovery_handler._coord_push_sock.send(network_encode)

            timer = threading.Timer(FUTURE_TIMEOUT_SECONDS, _on_future_timeout)
            timer.daemon = True
            timer.start()
            try:
                model_output = future.result()
            finally:
                timer.cancel()
            
            if model_output is None:
                # None from sample_tokens() implies that the original execute_model()
                # call failed - raise that exception.
                exec_model_fut.result()
                raise RuntimeError("unexpected error")

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # If we are doing speculative decoding with structured output,
            # we need to get the draft token ids from the prior step before
            # we can compute the grammar bitmask for the deferred request.
            if self.use_spec_decode:
                draft_token_ids = self.model_executor.take_draft_token_ids()
                assert draft_token_ids is not None
                # Update the draft token ids in the scheduler output to
                # filter out the invalid spec tokens, which will be padded
                # with -1 and skipped by the grammar bitmask computation.
                self.scheduler.update_draft_token_ids_in_output(
                    draft_token_ids, deferred_scheduler_output
                )
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

        return engine_core_outputs, model_executed

EngineCoreProc.run_engine_core = _patched_run_engine_core
