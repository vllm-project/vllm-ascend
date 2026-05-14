import threading
import torch
import torch_npu
import msgspec.msgpack
import pickle
import zmq

from vllm.tests.kernels.moe.test_moe import vllm_config
from vllm.vllm.config import VllmConfig


from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_zmq_ipc_path, make_zmq_socket
from vllm_ascend.recovery.exception_handler import ExceptionHandlerFactory, NetworkExceptionHandler
from vllm_ascend.recovery.types import ExceptionInfo, FaultReport, RecoveryAction, RecoveryStep, RecoveryPlan 
from vllm_ascend.recovery.recovery_executor import RecoveryExecutor

logger = init_logger(__name__)

class WorkerMonitor:
    """
    故障处理线程
    搞3个zmq socket
    1. 接收worker的错误信息
    2. 接收EngineCore下发的RecoveryPlan
    3. 向EngineCore发送故障信息和执行结果
    """
    def __init__(self, vllm_config:VllmConfig, worker) -> None:
        self.vllm_config = vllm_config
        self.worker = worker

        self.exception_handler_factory = self.build_exception_handler_factory()
        self.recovery_executor = self.register_recovery_executor()

        self.worker_input_address = get_open_zmq_ipc_path()
        self.core_input_address = get_open_zmq_ipc_path()
        self.core_output_address = get_open_zmq_ipc_path()

        self._monitor_thread = threading.Thread | None
    
    def build_exception_handler_factory(self) -> ExceptionHandlerFactory:
        exception_handler_factory = ExceptionHandlerFactory()
        network_handler = NetworkExceptionHandler()

        exception_handler_factory._register_handler(network_handler)
        
        return exception_handler_factory

    def register_recovery_executor(self):
        recovery_executor = RecoveryExecutor(component_type="worker")

        recovery_executor.register_handler("stop_device", self.stop_device)
        recovery_executor.register_handler("restart_device", self.restart_device)
        recovery_executor.register_handler("reinit_process_group", self.reinit_process_group)
        recovery_executor.register_handler("clean_cache", self.clean_cache)
        return recovery_executor

    def stop_device(self) -> bool:
        pass     
    
    def restart_device(self) -> bool:
        pass

    def reinit_process_group(self) -> bool:
        pass

    def clean_cache(self) -> bool:
        pass

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
                path=worker_input_address, 
                ctx=self.ctx,
                socket_type=zmq.PULL,
                bind=True,
            ) as worker_input_socket,
            make_zmq_socket(
                path=core_input_address,
                ctx=self.ctx,
                socket_type=zmq.SUB,
                bind=None,
            ) as core_input_socket,
            make_zmq_socket(
                path=core_output_address,
                ctx=self.ctx,
                socket_type=zmq.PUSH,
                bind=None,
            ) as core_output_socket,
        ):
            poller = zmq.Poller()
            poller.register(worker_input_socket, zmq.POLLIN)
            poller.register(core_input_socket, zmq.POLLIN)

            while True:
                events = poller.poll()
                events = dict(events)
                if worker_input_socket in events:
                    logger.info("[WorkerMonitorThread] WorkerProc hit with an exception")
                    buffer = worker_input_socket.recv()
                    exc = msgspec.msgpack.decode(buffer, type=ExceptionInfo)
                    handler = self.exception_handler_factory.get_handler(exc)
                    if handler is None:
                        logger.info(f"Non-recoverable error detected in worker thread.")
                        continue
                    
                    recovery_plan = handler.generate_plan(exc, self.vllm_config)
                    plan_encode = msgspec.msgpack.encode(("recoveryplan", recovery_plan))
                    core_output_socket.send(plan_encode)

                if core_input_socket in events:
                    # TODO: 从EngineCore接收到故障信息，准备解包RecoveryPlan并执行对应逻辑,
                    buffer = core_input_socket.recv()
                    decoded = msgspec.msgpack.decode(buffer, type=RecoveryStep)
                    logger.info("Receive recovery step from enginecore")
    