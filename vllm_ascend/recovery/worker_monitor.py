import threading
import torch
import torch_npu
import msgspec.msgpack
import zmq

from vllm.config import VllmConfig


from vllm.logger import logger
from vllm.utils.network_utils import get_open_zmq_ipc_path, make_zmq_socket
from vllm_ascend.recovery.exception_handler import ExceptionHandlerFactory, NetworkExceptionHandler
from vllm_ascend.recovery.types import ExceptionInfo, FaultReport, RecoveryPlan, StepResult 
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
    def __init__(self, vllm_config:VllmConfig, worker, ctx:zmq.Context) -> None:
        self.vllm_config = vllm_config
        self.worker = worker
        self.ctx = ctx
        
        self.exception_handler_factory = self.build_exception_handler_factory()
        self.recovery_executor = self.register_recovery_executor()

        self.worker_input_address = get_open_zmq_ipc_path()
        self.core_input_address = get_open_zmq_ipc_path()
        self.core_output_address = get_open_zmq_ipc_path()
        
        self._exception_decoder = msgspec.msgpack.Decoder(ExceptionInfo)
        self._recovery_decoder = msgspec.msgpack.Decoder(RecoveryPlan)
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

    def stop_device(self, context:dict | None) -> bool:
        try:
            stop_result = torch_npu.npu.stop_device(torch.npu.current_device())
            if stop_result == 0:
                logger.info("stop_device executed successfully")
                return True
            else:
                logger.error(f"stop_device failed with result: {stop_result}")
                return False
        except Exception as e:
            logger.error(f"stop_device executed failed with exception: {e}")
            return False
    
    def restart_device(self, context:dict | None) -> bool:
        try:
            ctx = context or {}
            torch_npu.npu.restart_device(
                torch.npu.current_device(), rebuild_all_resources=ctx.get("rebuild_all_resources", False)
            )
            return True
        except Exception as e:
            logger.error(f"restart_device executed failed with exception: {e}")
            return False

    def reinit_process_group(self, context:dict | None) -> bool:
        try:
            ctx = context or {}
            torch.distributed.reinit_process_group(
                group=ctx.get("group", None), rebuild_link=ctx.get("rebuild_link", True)
            )
            return True
        except Exception as e:
            logger.error(f"reinit_process_group executed failed with exception: {e}")
            return False

    def clean_cache(self, context:dict | None) -> bool:
        try:
            ctx = context or {}
            abort_list = context.get("abort_list", [])
            model_runner = self.worker.model_runner
            for req_id in abort_list:
                model_runner.requests.pop(req_id, None)
                model_runner.num_prompt_logprobs.pop(req_id, None)
                model_runner.input_batch.remove_request(req_id)
            return True
        except Exception as e:
            logger.error(f"worker clean_cached failed with exception: {e}")
            return False

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
                socket_type=zmq.SUB,
                bind=None,
            ) as core_input_socket,
            make_zmq_socket(
                path=self.core_output_address,
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
                    try:
                        exc = self._exception_decoder.decode(buffer)
                    except msgspec.DecodeError as e:
                        logger.error(f"Failed to decode exception info from worker thread: {e}")
                        continue
                    handler = self.exception_handler_factory.get_handler(exc)
                    if handler is None:
                        logger.info(f"Non-recoverable error detected in worker thread.")
                        continue
                    
                    recovery_plan = handler.generate_plan(exc, self.vllm_config)

                    fault_report = FaultReport(
                        worker_rank=self.worker.rank,
                        fault_type=recovery_plan,
                        recovery_plan=recovery_plan,
                        context=None,
                        timestamp=time.time(),
                    )
                    report_encode = msgspec.msgpack.encode(("faultreport", fault_report))
                    core_output_socket.send(report_encode)

                if core_input_socket in events:
                    # TODO: 从EngineCore接收到故障信息，准备解包RecoveryStep并执行对应逻辑,
                    buffer = core_input_socket.recv()
                    try:
                        recovery_step = self._recovery_decoder.decode(buffer)
                    except msgspec.DecodeError as e:
                        logger.error(f"Failed to decode recovery plan from enginecore: {e}")
                        continue
                    is_success = self.recovery_executor.execute_step(recovery_step)
                    step_result = StepResult(
                        step_name=recovery_step.name,
                        worker_rank=self.worker.rank,
                        is_success=is_success,
                    )
                    step_result_encode = msgspec.msgpack.encode(("stepresult", step_result))
                    core_output_socket.send(step_result_encode)

def create_worker_monitor(worker, vllm_config:VllmConfig):
    if hasattr(worker, 'worker_monitor') and worker.worker_monitor is not None:
        logger.info("WorkerMonitor already exists, skipping creation")
        return

    ctx=zmq.Context()
    worker.in_recovery = False
    worker_monitor = WorkerMonitor(vllm_config, worker, ctx)
    worker.worker_monitor = worker_monitor
    worker.worker_input_socket = make_zmq_socket(
        path=worker_monitor.worker_input_address,
        ctx=ctx,
        socket_type=zmq.PUSH,
        bind=None,
    )
