import threading
import time
import torch
import torch_npu
import msgspec.msgpack
import zmq

from vllm.config import VllmConfig


from vllm.logger import logger
from vllm.utils.network_utils import get_open_zmq_ipc_path, make_zmq_socket
from vllm_ascend.recovery.exception_handler import ExceptionHandlerFactory, NetworkExceptionHandler
from vllm_ascend.recovery.types import ExceptionInfo, FaultReport, NetworkCheck, RecoveryPlan, RecoveryStep, StepResult, WorkerStepDispatch
from vllm_ascend.recovery.utils import get_engine_recovery_bind_address

class WorkerMonitor:
    """
    故障处理线程
    搞3个zmq socket
    1. 接收worker的错误信息
    2. 接收EngineCore下发的RecoveryPlan
    3. 向EngineCore发送故障信息和执行结果
    """
    def __init__(self, vllm_config:VllmConfig, worker, ctx:zmq.Context) -> None:
        self.vllm_config = vllm_config
        self._worker = worker
        self.ctx = ctx

        self.exception_handler_factory = self.build_exception_handler_factory()
        self.worker_input_address = get_open_zmq_ipc_path()
        self.engine_index = self.vllm_config.parallel_config.data_parallel_rank
        (
            self.core_input_address,
            self.core_report_address,
            self.core_result_address,
        ) = get_engine_recovery_bind_address(self.engine_index)
        
        self._exception_decoder = msgspec.msgpack.Decoder(ExceptionInfo)
        self._recovery_decoder = msgspec.msgpack.Decoder(WorkerStepDispatch)
        self._network_check_decoder = msgspec.msgpack.Decoder(NetworkCheck)
        self._monitor_thread = threading.Thread | None
    
    def build_exception_handler_factory(self) -> ExceptionHandlerFactory:
        exception_handler_factory = ExceptionHandlerFactory()
        network_handler = NetworkExceptionHandler()

        exception_handler_factory._register_handler(network_handler)
        
        return exception_handler_factory

    def _do_network_check(self):
        def _sync_and_report():
            try:
                logger.info("[WorkerMonitor] NetworkCheck sync begin")
                torch.npu.current_stream().synchronize()
            except Exception as e:
                logger.error(
                    "[WorkerMonitor] NetworkCheck synchronize detected error: %s",
                    e,
                )
                exception_info = ExceptionInfo(
                    exception_type=type(e).__name__,
                    exception_msg=str(e),
                )
                try:
                    self._worker.worker_input_socket.send(
                        msgspec.msgpack.encode(exception_info)
                    )
                except Exception:
                    logger.exception(
                        "[WorkerMonitor] Failed to send exception via worker_input_socket"
                    )

        t = threading.Thread(target=_sync_and_report, name="NetworkCheckSync", daemon=True)
        t.start()

    def start(self):
        self._monitor_thread = threading.Thread(
            target=self._run_monitor,
            name="WorkerMonitorThread",
            daemon=True,
        )
        self._monitor_thread.start()

    def _run_monitor(self):
        with (
            make_zmq_socket(
                path=self.worker_input_address, 
                ctx=self.ctx,
                socket_type=zmq.PULL,
                bind=True,
            ) as worker_input_socket,
            make_zmq_socket(
                path=self.core_input_address,
                ctx=self.ctx,
                socket_type=zmq.XSUB,
                bind=None,
            ) as core_input_socket,
            make_zmq_socket(
                path=self.core_report_address,
                ctx=self.ctx,
                socket_type=zmq.PUSH,
                bind=False,
            ) as core_report_socket,
            make_zmq_socket(
                path=self.core_result_address,
                ctx=self.ctx,
                socket_type=zmq.PUSH,
                bind=False,
            ) as core_result_socket,
        ):
            core_input_socket.send(b"\x01")

            poller = zmq.Poller()
            poller.register(worker_input_socket, zmq.POLLIN)
            poller.register(core_input_socket, zmq.POLLIN)

            while True:
                events = poller.poll()
                events = dict(events)
                if worker_input_socket in events:
                    logger.info("[WorkerMonitor] WorkerProc hit with an exception")
                    buffer = worker_input_socket.recv()
                    try:
                        exc = self._exception_decoder.decode(buffer)
                    except msgspec.DecodeError as e:
                        logger.error("[WorkerMonitor] Failed to decode exception info from worker thread: %s", e)
                        continue
                    handler = self.exception_handler_factory.get_handler(exc)
                    if handler is None:
                        logger.info("[WorkerMonitor] Non-recoverable error detected in worker thread.")
                        pass
                    else:
                        recovery_plan = handler.generate_plan(exc, self.vllm_config)

                        fault_report = FaultReport(
                            worker_rank=self._worker.rank,
                            engine_index=self.engine_index,
                            exp=exc,
                            plan=recovery_plan,
                        )
                        report_encode = msgspec.msgpack.encode(fault_report)
                        core_report_socket.send(report_encode)

                if core_input_socket in events:
                    buffer = core_input_socket.recv()
                    try:
                        msg = msgspec.msgpack.decode(buffer)
                        msg_type = msg[0]
                        msg_data = msg[1]
                    except Exception:
                        logger.exception("Failed to deserialize recovery msg")
                        continue
                    if msg is not None:
                        if msg_type == "networkcheck":
                            network_check = msgspec.msgpack.decode(msg_data, type=NetworkCheck)
                            logger.info(
                                "[WorkerMonitor] Received NetworkCheck from engine %d, "
                                "starting synchronize check",
                                network_check.engine_index,
                            )
                            self._do_network_check(worker_input_socket)
                            continue
                        elif msg_type == "recoverystep":
                            recovery_step_with_cfg = self._recovery_decoder.decode(buffer, type=WorkerStepDispatch)
                            logger.info("[WorkerMonitor] Receive recovery_step from EngineCoreProc")
                            recovery_step = recovery_step_with_cfg.step
                            cfg = recovery_step_with_cfg.cfg
                            cfg, is_success = recovery_step.execute(self._worker, cfg)
                            step_result = StepResult(
                                step_name=recovery_step.name,
                                success=is_success,
                                worker_rank=self._worker.rank,
                                cfg=cfg
                            )
                            step_result_encode = msgspec.msgpack.encode(step_result)
                            core_result_socket.send(step_result_encode)
    

def create_worker_monitor(worker, vllm_config:VllmConfig):
    if hasattr(worker, 'worker_monitor') and worker.worker_monitor is not None:
        logger.info("WorkerMonitor already exists, skipping creation")
        return

    ctx=zmq.Context()
    worker.in_recovery = False
    worker.exception_occur = False
    worker.device_stopped = False
    worker_monitor = WorkerMonitor(vllm_config, worker, ctx)
    worker.worker_monitor = worker_monitor
    worker.worker_input_socket = make_zmq_socket(
        path=worker_monitor.worker_input_address,
        ctx=ctx,
        socket_type=zmq.PUSH,
        bind=None,
    )
