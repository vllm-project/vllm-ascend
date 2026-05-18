import pickle
import signal
from contextlib import ExitStack

import zmq
from vllm.config import ParallelConfig, VllmConfig
from vllm.logger import logger
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils.system_utils import set_process_title
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc, EngineShutdownState
from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.serial_utils import MsgpackDecoder
from vllm.v1.utils import get_engine_client_zmq_addr
from vllm.utils.network_utils import make_zmq_socket
from vllm_ascend.recovery import RecoveryHandler


_RECOVERY_MSG_PREFIX = b"\x00REC"


class RasDPEngineCoreProc(DPEngineCoreProc):

    def __init__(self, *args, **kwargs):
        vllm_config: VllmConfig = kwargs["vllm_config"]

        recover_step_pub_addr = get_engine_client_zmq_addr(
            local_only=True, host="127.0.0.1"
        )
        recover_report_pull_addr = get_engine_client_zmq_addr(
            local_only=True, host="127.0.0.1"
        )
        self._recovery_addrs = {
            "recover_step_pub_addr": recover_step_pub_addr,
            "recover_report_pull_addr": recover_report_pull_addr,
        }

        self._recovery_handler = RecoveryHandler(
            engine_core=None,
            vllm_config=vllm_config,
        )

        logger.info(
            "[RAS] RecoveryHandler started: recovery_addrs=%s", self._recovery_addrs
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
                        msg = pickle.loads(payload)
                        if msg[0] == "RECOVERY_ADDRESSES":
                            _, sub_addr, push_addr = msg
                            logger.info(
                                "[RAS] Received recovery addresses from "
                                "dp coordinator: sub=%s, push=%s",
                                sub_addr,
                                push_addr,
                            )
                            self._recovery_handler.connect_coordinator(
                                sub_addr, push_addr
                            )
                    except Exception as e:
                        logger.exception(
                            "[RAS] Failed to deserialize recovery addr msg: %s", e
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


EngineCoreProc.run_engine_core = _patched_run_engine_core
