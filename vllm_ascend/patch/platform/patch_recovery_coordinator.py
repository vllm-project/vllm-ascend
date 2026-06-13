import time
import weakref

import msgspec
import msgspec.msgpack
import zmq
from vllm.logger import logger
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.coordinator import DPCoordinator, DPCoordinatorProc
from vllm.v1.engine.coordinator import shutdown as coord_shutdown
from vllm.v1.serial_utils import MsgpackDecoder
from vllm.v1.utils import get_engine_client_zmq_addr
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import get_mp_context
from vllm_ascend.recovery.types import (
    FaultReport,
    NetworkCheck,
    RecoveryComplete,
    RecoveryPlanResult,
)


_RECOVERY_MSG_PREFIX = b"\x00REC"


def _patched_dp_coordinator_init(
    self, parallel_config, enable_wave_coordination=True
):
    dp_size = parallel_config.data_parallel_size
    assert dp_size > 1, "Coordinator only used for data parallel"

    host = parallel_config.data_parallel_master_ip
    local_only = not parallel_config.local_engines_only
    front_publish_address = get_engine_client_zmq_addr(
        local_only=local_only, host=host
    )

    local_only_eng = dp_size == parallel_config.data_parallel_size_local
    if parallel_config.enable_elastic_ep:
        local_only_eng = False
    back_publish_address = get_engine_client_zmq_addr(local_only_eng, host)
    back_output_address = get_engine_client_zmq_addr(local_only_eng, host)

    recovery_pub_address = get_engine_client_zmq_addr(local_only_eng, host)
    recovery_pull_address = get_engine_client_zmq_addr(local_only_eng, host)
    logger.info(
        "recovery_pub_address=%s, recovery_pull_address=%s",
        recovery_pub_address, recovery_pull_address,
    )

    context = get_mp_context()
    self.proc = context.Process(
        target=DPCoordinatorProc.run_coordinator,
        name="VLLM_DP_Coordinator",
        kwargs={
            "engine_count": parallel_config.data_parallel_size,
            "front_publish_address": front_publish_address,
            "back_output_address": back_output_address,
            "back_publish_address": back_publish_address,
            "enable_wave_coordination": enable_wave_coordination,
            "recovery_pub_address": recovery_pub_address,
            "recovery_pull_address": recovery_pull_address,
        },
        daemon=True,
    )
    self.proc.start()

    self.stats_publish_address = front_publish_address
    self.coord_in_address = back_publish_address
    self.coord_out_address = back_output_address
    self.recovery_pub_address = recovery_pub_address
    self.recovery_pull_address = recovery_pull_address
    self._finalizer = weakref.finalize(self, coord_shutdown, [self.proc])


DPCoordinator.__init__ = _patched_dp_coordinator_init


def _patched_run_coordinator(
    engine_count,
    front_publish_address,
    back_output_address,
    back_publish_address,
    min_stats_update_interval_ms=100,
    enable_wave_coordination=True,
    recovery_pub_address=None,
    recovery_pull_address=None,
):
    coordinator = DPCoordinatorProc(
        engine_count=engine_count,
        min_stats_update_interval_ms=min_stats_update_interval_ms,
        enable_wave_coordination=enable_wave_coordination,
    )
    try:
        coordinator.process_input_socket(
            front_publish_address,
            back_output_address,
            back_publish_address,
            recovery_pub_address=recovery_pub_address,
            recovery_pull_address=recovery_pull_address,
        )
    except KeyboardInterrupt:
        logger.info("DP Coordinator process exiting")


DPCoordinatorProc.run_coordinator = _patched_run_coordinator


def _patched_process_input_socket(
    self,
    front_publish_address,
    back_output_address,
    back_publish_address,
    recovery_pub_address,
    recovery_pull_address,
):
    decoder = MsgpackDecoder(EngineCoreOutputs)

    current_wave = 0
    engines_running = False
    stats_changed = False
    last_stats_step = -1
    last_stats_wave = -1
    last_step_counts = None
    is_recovering = False
    engine_count = len(self.engines)

    with (
        make_zmq_socket(
            path=front_publish_address, ctx=self.ctx,
            socket_type=zmq.XPUB, bind=True,
        ) as publish_front,
        make_zmq_socket(
            path=back_output_address, ctx=self.ctx,
            socket_type=zmq.PULL, bind=True,
        ) as output_back,
        make_zmq_socket(
            path=back_publish_address, ctx=self.ctx,
            socket_type=zmq.XPUB, bind=True,
        ) as publish_back,
        make_zmq_socket(
            path=recovery_pub_address, ctx=self.ctx,
            socket_type=zmq.XPUB, bind=True,
        ) as recovery_pub,
        make_zmq_socket(
            path=recovery_pull_address, ctx=self.ctx,
            socket_type=zmq.PULL, bind=True,
        ) as recovery_pull,
    ):
        for _ in self.engines:
            if publish_back.recv() != b"\x01":
                logger.error(
                    "DP Coordinator received unexpected message while "
                    "waiting for engines to subscribe"
                )
                return
        publish_back.send(b"READY")

        logger.info("All engine subscriptions received by DP coordinator")

        recovery_addr_msg = _RECOVERY_MSG_PREFIX + msgspec.msgpack.encode(
            ("RECOVERY_ADDRESSES", recovery_pub_address, recovery_pull_address)
        )
        publish_back.send(recovery_addr_msg)
        logger.info(
            "Broadcast recovery addresses via publish_back: pub=%s, pull=%s",
            recovery_pub_address, recovery_pull_address,
        )

        for _ in self.engines:
            sub_msg = recovery_pub.recv()
            if sub_msg != b"\x01":
                logger.error(
                    "DP Coordinator received unexpected message while "
                    "waiting for recovery subscriptions"
                )
                return
        logger.info("All engine recovery subscriptions received")

        poller = zmq.Poller()
        poller.register(publish_front, zmq.POLLIN)
        poller.register(publish_back, zmq.POLLIN)
        poller.register(output_back, zmq.POLLIN)
        if recovery_pull is not None:
            poller.register(recovery_pull, zmq.POLLIN)

        last_publish_time = 0

        while True:
            elapsed = int(time.time() * 1000) - last_publish_time
            wait_for = self.stats_update_interval_ms if stats_changed else 5000
            min_timeout = 50 if last_step_counts is None else 0

            events = poller.poll(timeout=max(min_timeout, wait_for - elapsed))
            if not events:
                if last_step_counts is not None:
                    engine_req_counts_list = last_step_counts
                    last_step_counts = None
                else:
                    engine_req_counts_list = self._get_engine_counts()
                    stats_changed = False

                to_publish = (engine_req_counts_list, current_wave, engines_running)
                publish_front.send(msgspec.msgpack.encode(to_publish))
                last_publish_time = int(time.time() * 1000)
                continue

            events = dict(events)
            wave_state_changed = False

            if publish_back in events:
                buffer = publish_back.recv()
                if buffer == b"\x01":
                    publish_back.send(b"READY")
                elif buffer != b"\x00":
                    logger.error(
                        "DP Coordinator received unexpected message from engines"
                    )

            if publish_front in events:
                buffer = publish_front.recv()
                if buffer in (b"\x01", b"\x00"):
                    continue

                decoded = msgspec.msgpack.decode(buffer)
                if (
                    isinstance(decoded, (list, tuple))
                    and len(decoded) == 2
                    and decoded[0] == "SCALE_ELASTIC_EP"
                ):
                    new_engine_count = decoded[1]
                    current_count = len(self.engines)
                    if new_engine_count > current_count:
                        for _ in range(new_engine_count - current_count):
                            self.engines.append(
                                type(self.engines[0])()
                            )
                        logger.info(
                            "DPCoordinator scaled up from %s to %s engines",
                            current_count, new_engine_count,
                        )
                    else:
                        self.engines = self.engines[:new_engine_count]
                        logger.info(
                            "DPCoordinator scaled down from %s to %s engines",
                            current_count, new_engine_count,
                        )
                    continue

                if self.enable_wave_coordination:
                    engine_to_exclude, wave = decoded
                    if not engines_running:
                        if wave < current_wave:
                            engine_to_exclude = None
                        engines_running = True
                        wave_state_changed = True
                        self._send_start_wave(
                            publish_back, current_wave, engine_to_exclude
                        )

            if output_back in events:
                buffer = output_back.recv()
                outputs = decoder.decode(buffer)

                assert not outputs.outputs
                assert outputs.utility_output is None

                eng_index = outputs.engine_index
                scheduler_stats = outputs.scheduler_stats
                if scheduler_stats:
                    stats = self.engines[eng_index].request_counts
                    stats_step = scheduler_stats.step_counter
                    stats_wave = scheduler_stats.current_wave
                    if (
                        stats_wave > last_stats_wave
                        or stats_wave == last_stats_wave
                        and stats_step > last_stats_step
                    ):
                        if stats_changed:
                            last_step_counts = self._get_engine_counts(do_copy=True)
                        last_stats_step = stats_step
                        last_stats_wave = stats_wave
                    elif stats_wave != last_stats_wave or (
                        stats_step != last_stats_step
                    ):
                        logger.warning(
                            "Received stats for out-of-order "
                            "step (%d, %d) from engine %d (expected "
                            "> (%d, %d))",
                            stats_wave, stats_step, eng_index,
                            last_stats_wave, last_stats_step,
                        )
                    stats[0] = scheduler_stats.num_waiting_reqs
                    stats[1] = scheduler_stats.num_running_reqs
                    stats_changed = True

                if self.enable_wave_coordination:
                    if (wave := outputs.wave_complete) is not None:
                        if current_wave <= wave:
                            new_wave = wave + 1
                            logger.debug(
                                "Moving DP wave from %d to %d.",
                                current_wave, new_wave,
                            )
                            current_wave = new_wave
                            engines_running = False
                            wave_state_changed = True
                    elif (wave := outputs.start_wave) is not None and (
                        wave > current_wave
                        or (wave == current_wave and not engines_running)
                    ):
                        logger.debug(
                            "Starting wave %d after notification of "
                            "stale wave request from engine.",
                            wave,
                        )
                        current_wave = wave
                        engines_running = True
                        wave_state_changed = True
                        self._send_start_wave(publish_back, wave, eng_index)

            if recovery_pull in events:
                buffer = recovery_pull.recv()
                try:
                    msg = msgspec.msgpack.decode(buffer)
                    msg_type = msg[0]
                    msg_data = msg[1]
                except Exception:
                    logger.exception("Failed to deserialize recovery msg")
                    msg = None

                if msg is not None:
                    if msg_type == "faultreport":
                        fault_report = msgspec.convert(msg_data, type=FaultReport)
                        if is_recovering:
                            logger.info(
                                "[RAS] Ignoring FaultReport from engine %d worker %d "
                                "while recovering, exp=%s",
                                fault_report.engine_index,
                                fault_report.worker_rank,
                                fault_report.exp.exception_msg,
                            )
                            continue
                        is_recovering = True
                        plan_results: dict[int, RecoveryPlanResult] = {}
                        plan_deadline = time.time() + fault_report.plan.timeout_s
                        logger.info(
                            "[RAS] Received FaultReport from engine %d worker %d: %s",
                            fault_report.engine_index,
                            fault_report.worker_rank,
                            fault_report.exp.exception_msg,
                        )
                        recovery_pub.send(msgspec.msgpack.encode(("recoveryplan", fault_report.plan)))
                        logger.info(
                            "[RAS] Broadcast RecoveryPlan '%s' to all engines "
                            "(timeout=%ds, deadline=%.3f)",
                            fault_report.plan.name,
                            fault_report.plan.timeout_s,
                            plan_deadline,
                        )
                    elif msg_type == "recoveryplanresult":
                        recovery_plan_result = msgspec.convert(msg_data, type=RecoveryPlanResult)
                        assert is_recovering, "Received RecoveryPlanResult while not recovering"
                        plan_results[recovery_plan_result.engine_index] = recovery_plan_result
                        logger.info(
                            "[RAS] Received RecoveryPlanResult from engine %d: "
                            "plan=%s success=%s (%d/%d)",
                            recovery_plan_result.engine_index,
                            recovery_plan_result.plan_name,
                            recovery_plan_result.success,
                            len(plan_results),
                            engine_count,
                        )
                        if not recovery_plan_result.success:
                            logger.error(
                                "[RAS] Engine %d reported FAILURE for plan '%s', "
                                "aborting recovery",
                                recovery_plan_result.engine_index,
                                recovery_plan_result.plan_name,
                            )
                            recovery_pub.send(msgspec.msgpack.encode(
                                ("recoverycomplete", RecoveryComplete(
                                    plan_name=recovery_plan_result.plan_name,
                                    success=False,
                                    current_wave=current_wave,
                                ))
                            ))
                            is_recovering = False
                            logger.info(
                                "[RAS] Broadcast RecoveryComplete(failed) to all engines"
                            )
                        elif len(plan_results) == engine_count:
                            is_recovering = False
                            logger.info(
                                "[RAS] All engines reported: ALL SUCCESS",
                            )
                            recovery_pub.send(msgspec.msgpack.encode(
                                ("recoverycomplete", RecoveryComplete(
                                    plan_name=recovery_plan_result.plan_name,
                                    success=True,
                                    current_wave=current_wave,
                                ))
                            ))
                            logger.info(
                                "[RAS] Broadcast RecoveryComplete(success, wave=%d) "
                                "to all engines",
                                current_wave,
                            )
                            
                    elif msg_type == "networkcheck":
                        network_check = msgspec.convert(msg_data, type=NetworkCheck)
                        logger.info(
                            "[RAS] Received NetworkCheck from engine %d, "
                            "broadcasting to all engines",
                            network_check.engine_index,
                        )
                        recovery_pub.send(msgspec.msgpack.encode(("networkcheck", network_check)))
                    else:
                        logger.warning(
                            "[RAS] Unknown recovery msg type: %s", msg_type
                        )

            if is_recovering and time.time() > plan_deadline:
                missing = [
                    i for i in range(engine_count) if i not in plan_results
                ]
                logger.error(
                    "[RAS] RecoveryPlan timed out! Missing results from engines: %s "
                    "(received %d/%d)",
                    missing,
                    len(plan_results),
                    engine_count,
                )
                recovery_pub.send(msgspec.msgpack.encode(
                    ("recoverycomplete", RecoveryComplete(
                        plan_name="",
                        success=False,
                        current_wave=current_wave,
                    ))
                ))
                is_recovering = False
                logger.info(
                    "[RAS] Broadcast RecoveryComplete(timeout) to all engines"
                )

            if wave_state_changed:
                message = (None, current_wave, engines_running)
                publish_front.send(msgspec.msgpack.encode(message))


DPCoordinatorProc.process_input_socket = _patched_process_input_socket