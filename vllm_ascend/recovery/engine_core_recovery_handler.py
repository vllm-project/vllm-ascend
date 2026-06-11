import threading
import time
from typing import Any, Tuple
  
import msgspec.msgpack
import zmq
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_socket
from vllm_ascend.recovery.types import (
    FaultReport,
    NetworkCheck,
    RecoveryPlan,
    RecoveryComplete,
    RecoveryPlanResult,
    RecoveryAction,
    RecoveryStep,
    StepResult,
    StepTarget,
    WorkerStepDispatch,
)


class RecoveryHandler:
    def __init__(
        self,
        engine_core: Any,
        vllm_config: VllmConfig,
    ):
        self._engine_core = engine_core
        self._vllm_config = vllm_config
        self.is_recovering = False
        self._recovery_done_event = threading.Event()
        self._recovery_success = False

        parallel_config = vllm_config.parallel_config
        self._worker_count = parallel_config.local_world_size
        self._engine_index = parallel_config.data_parallel_rank
        self._expect_coordinator = vllm_config.needs_dp_coordinator

        self._ctx = zmq.Context()
        self._coord_push_sock: zmq.Socket | None = None
        self._coord_sub_sock: zmq.Socket | None = None
        self._coord_ready_event = threading.Event()

        self._report_decoder = msgspec.msgpack.Decoder(FaultReport)
        self._result_decoder = msgspec.msgpack.Decoder(StepResult)
    
    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="RecoveryHandler", daemon=True
        )
        self._thread.start()

    def setup_recover_sockets(
        self,
        recover_step_pub_addr: str,
        recover_report_pull_addr: str,
        recover_step_result_pull_addr: str,
    ) -> None:
        self._recover_step_pub_sock = make_zmq_socket(
            self._ctx, recover_step_pub_addr, zmq.XPUB, bind=True,
        )
        self._recover_report_pull_sock = make_zmq_socket(
            self._ctx, recover_report_pull_addr, zmq.PULL, bind=True,
        )
        self._recover_step_result_pull_sock = make_zmq_socket(
            self._ctx, recover_step_result_pull_addr, zmq.PULL, bind=True,
        )
        logger.info(
            "[RecoveryHandler][engine=%d] Engine sockets bound: "
            "recover_step=%s, recover_report=%s, recover_step_result=%s",
            self._engine_index,
            recover_step_pub_addr,
            recover_report_pull_addr,
            recover_step_result_pull_addr,
        )

    def wait_for_worker_subscriptions(self) -> None:
        for _ in range(self._worker_count):
            msg = self._recover_step_pub_sock.recv()
            if msg != b"\x01":
                logger.error(
                    "[RecoveryHandler][engine=%d] Unexpected subscription "
                    "message: %s",
                    self._engine_index, msg,
                )
        logger.info(
            "[RecoveryHandler][engine=%d] All %d workers subscribed to "
            "recovery step channel",
            self._engine_index, self._worker_count,
        )

    def connect_coordinator(self, coord_sub_addr: str, coord_push_addr: str) -> None:
        self._coord_push_sock = make_zmq_socket(
            self._ctx, coord_push_addr, zmq.PUSH, bind=False,
        )
        self._coord_sub_sock = make_zmq_socket(
            self._ctx, coord_sub_addr, zmq.SUB, bind=False,
        )
        self._coord_sub_sock.setsockopt_string(zmq.SUBSCRIBE, "")

        self._coord_ready_event.set()
        logger.info(
            "[RecoveryHandler][engine=%d] Connected to coordinator: "
            "sub=%s, push=%s",
            self._engine_index, coord_sub_addr, coord_push_addr,
        )

    def wait_for_recovery(self, timeout: float | None = None) -> bool:
        return self._recovery_done_event.wait(timeout=timeout)

    def get_recovery_success(self) -> bool:
        return self._recovery_success

    def _begin_recovery(self) -> None:
        self._recovery_done_event.clear()
        self._recovery_success = False
        self.is_recovering = True

    def _finish_recovery(self, success: bool) -> None:
        if success:
            finished_step = RecoveryStep(
                name="completed_recovery",
                target=StepTarget.WORKER.value,
                timeout_s=5,
                actions=[RecoveryAction(name="recovery_finished")],
            )
            self._dispatch_worker_step(finished_step, {})
        self._recovery_success = success
        self.is_recovering = False
        self._recovery_done_event.set()

    def _run(self) -> None:
        poller = zmq.Poller()
        if self._expect_coordinator:
            logger.info(
                "[RecoveryHandler][engine=%d] Waiting for coordinator "
                "connection...",
                self._engine_index,
            )
            self._coord_ready_event.wait()
            logger.info(
                "[RecoveryHandler][engine=%d] Coordinator connected, "
                "entering main loop",
                self._engine_index,
            )
            poller.register(self._coord_sub_sock, zmq.POLLIN)
        if self._recover_report_pull_sock is not None:
            poller.register(self._recover_report_pull_sock, zmq.POLLIN)
        while True:
            events = dict(poller.poll(timeout=1000))
            if self._coord_sub_sock is not None and self._coord_sub_sock in events:
                self._handle_coord_msg()
            if self._recover_report_pull_sock is not None and self._recover_report_pull_sock in events:
                self._handle_worker_msg()

    def _handle_worker_msg(self) -> None:
        buffer = self._recover_report_pull_sock.recv()
        try:
            msg = self._report_decoder.decode(buffer)
        except Exception:
            logger.exception(
                "[RecoveryHandler][engine=%d] Failed to deserialize FaultReport",
                self._engine_index,
            )
            return

        if not isinstance(msg, FaultReport):
            logger.warning(
                "[RecoveryHandler][engine=%d] Expected FaultReport, got %s",
                self._engine_index, type(msg),
            )
            return

        if self.is_recovering:
            logger.info(
                "[RecoveryHandler][engine=%d] Already recovering, ignoring "
                "FaultReport from worker %d",
                self._engine_index, msg.worker_rank,
            )
            return
        logger.info(
            "[RecoveryHandler][engine=%d] Received FaultReport from "
            "worker %d: %s",
            self._engine_index, msg.worker_rank, msg.exp.exception_msg,
        )
        self._begin_recovery()
        if self._expect_coordinator:
            if self._coord_push_sock is not None:
                self._coord_push_sock.send(msgspec.msgpack.encode(("faultreport", msg)))
                logger.info(
                    "[RecoveryHandler][engine=%d] Forwarded FaultReport to "
                    "coordinator, waiting for RecoveryPlan",
                    self._engine_index,
                )
            else:
                logger.warning(
                    "[RecoveryHandler][engine=%d] Coordinator expected but "
                    "not connected, waiting for RecoveryPlan",
                    self._engine_index,
                )
        else:
            logger.info(
                "[RecoveryHandler][engine=%d] No coordinator, executing "
                "recovery plan directly",
                self._engine_index,
            )
            self._execute_recovery(msg.plan)

    def _handle_coord_msg(self) -> None:
        buffer = self._coord_sub_sock.recv()
        try:
            msg = msgspec.msgpack.decode(buffer)
            msg_type = msg[0]
            msg_data = msg[1]
        except Exception:
            logger.exception(
                "[RecoveryHandler][engine=%d] Failed to deserialize coord msg",
                self._engine_index,
            )
            return

        if msg_type == "recoveryplan":
            recovery_plan = msgspec.convert(msg_data, type=RecoveryPlan)
            logger.info(
                "[RecoveryHandler][engine=%d] Received RecoveryPlan: %s",
                self._engine_index, recovery_plan.name,
            )
            if not self.is_recovering:
                self._begin_recovery()
            self._execute_recovery(recovery_plan)
        elif msg_type == "networkcheck":
            network_check = msgspec.convert(msg_data, type=NetworkCheck)
            logger.info(
                "[RecoveryHandler][engine=%d] Received NetworkCheck from "
                "engine %d, dispatching to workers",
                self._engine_index, network_check.engine_index,
            )
            self._recover_step_pub_sock.send(
                msgspec.msgpack.encode(("networkcheck", network_check))
            )
        elif msg_type == "recoverycomplete":
            recovery_complete = msgspec.convert(msg_data, type=RecoveryComplete)
            self._handle_recovery_complete(recovery_complete)
        else:
            logger.warning(
                "[RecoveryHandler][engine=%d] Unknown coord msg type: %s",
                self._engine_index, msg_type,
            )

    def _handle_recovery_complete(self, msg: RecoveryComplete) -> None:
        if msg.success:
            logger.info(
                "[RecoveryHandler][engine=%d] RecoveryComplete: plan='%s' "
                "success=True wave=%d, syncing engine state",
                self._engine_index, msg.plan_name, msg.current_wave,
            )
        else:
            logger.error(
                "[RecoveryHandler][engine=%d] RecoveryComplete: plan='%s' "
                "success=False wave=%d",
                self._engine_index, msg.plan_name, msg.current_wave,
            )

        if self._engine_core is not None:
            self._engine_core.current_wave = msg.current_wave
            self._engine_core.step_counter = 0
            self._engine_core.engines_running = True

        self._finish_recovery(success=msg.success)

    def _execute_recovery(self, plan: RecoveryPlan) -> None:
        logger.info(
            "[RecoveryHandler][engine=%d] Executing RecoveryPlan '%s' with "
            "%d steps",
            self._engine_index, plan.name, len(plan.steps),
        )
        step_results: list[StepResult] = []
        success = True
        cfg = plan.cfg

        for step in plan.steps:
            if step.target == StepTarget.ENGINE_CORE.value:
                cfg, step_success = self._execute_engine_core_step(step, cfg)
            elif step.target == StepTarget.WORKER.value:
                cfg, step_success = self._dispatch_worker_step(step, cfg)
            else:
                logger.error(
                    "[RecoveryHandler][engine=%d] Unknown step target: %s",
                    self._engine_index, step.target,
                )
                step_success = False

            step_results.append(
                StepResult(
                    step_name=step.name,
                    success=step_success,
                    cfg=cfg,
                    worker_rank=-1,
                )
            )
            if not step_success:
                logger.error(
                    "[RecoveryHandler][engine=%d] Step '%s' failed, "
                    "aborting plan '%s'",
                    self._engine_index, step.name, plan.name,
                )
                success = False
                break

        result = RecoveryPlanResult(
            plan_name=plan.name,
            engine_index=self._engine_index,
            success=success,
            step_results=step_results,
        )
        self._finalize_recovery(plan, result)

    def _execute_engine_core_step(
        self, step: Any, cfg: dict
    ) -> Tuple[dict, bool]:
        cfg, success = step.execute(self._engine_core, cfg)
        if not success:
            logger.error(
                "[RecoveryHandler][engine=%d] EngineCore step '%s' failed",
                self._engine_index, step.name,
            )
        return cfg, success

    def _finalize_recovery(
        self, plan: RecoveryPlan, result: RecoveryPlanResult
    ) -> None:
        if self._expect_coordinator:
            self._report_plan_result(result)
            if result.success:
                logger.info(
                    "[RecoveryHandler][engine=%d] RecoveryPlan '%s' executed, "
                    "waiting for RecoveryComplete from coordinator",
                    self._engine_index, plan.name,
                )
            else:
                logger.error(
                    "[RecoveryHandler][engine=%d] RecoveryPlan '%s' failed, "
                    "aborting recovery",
                    self._engine_index, plan.name,
                )
                self._finish_recovery(success=False)
        else:
            if self._engine_core is not None:
                self._engine_core.current_wave += 1
                self._engine_core.step_counter = 0
                self._engine_core.engines_running = True
            self._finish_recovery(success=result.success)
            if result.success:
                logger.info(
                    "[RecoveryHandler][engine=%d] RecoveryPlan '%s' completed "
                    "successfully",
                    self._engine_index, plan.name,
                )
            else:
                logger.error(
                    "[RecoveryHandler][engine=%d] RecoveryPlan '%s' failed",
                    self._engine_index, plan.name,
                )

    def _dispatch_worker_step(
        self, step: Any, cfg: dict
    ) -> Tuple[dict, bool]:
        logger.info(
            "[RecoveryHandler][engine=%d] Dispatching worker step '%s' to "
            "%d workers",
            self._engine_index, step.name, self._worker_count,
        )
        self._recover_step_pub_sock.send(
            msgspec.msgpack.encode(("workerstepdispatch", WorkerStepDispatch(step=step, cfg=cfg)))
        )

        received = 0
        deadline = time.monotonic() + step.timeout_s
        poller = zmq.Poller()
        poller.register(self._recover_step_result_pull_sock, zmq.POLLIN)
        while received < self._worker_count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error(
                    "[RecoveryHandler][engine=%d] Worker step '%s' timed out: "
                    "%d/%d results received",
                    self._engine_index, step.name,
                    received, self._worker_count,
                )
                return cfg, False

            events = dict(poller.poll(
                timeout=min(1000, int(remaining * 1000))
            ))
            if self._recover_step_result_pull_sock in events:
                buffer = self._recover_step_result_pull_sock.recv()
                try:
                    msg = self._result_decoder.decode(buffer)
                except Exception:
                    logger.exception(
                        "[RecoveryHandler][engine=%d] Failed to deserialize "
                        "worker step result",
                        self._engine_index,
                    )
                    continue
                if not isinstance(msg, StepResult):
                    logger.warning(
                        "[RecoveryHandler][engine=%d] Expected "
                        "StepResult, got %s",
                        self._engine_index, type(msg),
                    )
                    continue
                received += 1
                cfg.update(msg.cfg)
                if not msg.success:
                    logger.error(
                        "[RecoveryHandler][engine=%d] Worker %d reported "
                        "failure for step '%s'",
                        self._engine_index, msg.worker_rank, step.name,
                    )
                    return cfg, False

        logger.info(
            "[RecoveryHandler][engine=%d] Worker step '%s' completed: "
            "%d/%d workers succeeded",
            self._engine_index, step.name,
            received, self._worker_count,
        )
        return cfg, True

    def _report_plan_result(self, result: RecoveryPlanResult) -> None:
        if self._coord_push_sock is not None:
            self._coord_push_sock.send(msgspec.msgpack.encode(("recoveryplanresult", result)))
            logger.info(
                "[RecoveryHandler][engine=%d] Reported RecoveryPlanResult to "
                "coordinator: plan=%s success=%s",
                self._engine_index, result.plan_name, result.success,
            )
        else:
            logger.info(
                "[RecoveryHandler][engine=%d] No coordinator connection, "
                "RecoveryPlanResult not reported: plan=%s success=%s",
                self._engine_index, result.plan_name, result.success,
            )
