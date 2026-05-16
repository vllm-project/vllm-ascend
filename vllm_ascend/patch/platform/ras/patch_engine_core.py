import pickle
from contextlib import ExitStack

import zmq
from vllm.logger import logger
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.serial_utils import MsgpackDecoder
from vllm.v1.utils import get_engine_client_zmq_addr
from vllm.utils.network_utils import make_zmq_socket
from vllm_ascend.recovery import RecoveryHandler


_RECOVERY_MSG_PREFIX = b"\x00REC"


_original_engine_core_proc_init = EngineCoreProc.__init__


def _patched_engine_core_proc_init(
    self,
    vllm_config,
    local_client,
    handshake_address,
    executor_class,
    log_stats,
    client_handshake_address=None,
    *,
    engine_index=0,
):
    worker_count = vllm_config.parallel_config.local_world_size
    expect_coordinator = vllm_config.needs_dp_coordinator

    recover_step_pub_addr = get_engine_client_zmq_addr(local_only=True, host="127.0.0.1")
    recover_report_pull_addr = get_engine_client_zmq_addr(local_only=True, host="127.0.0.1")
    self._recovery_addrs = {
        "recover_step_pub_addr": recover_step_pub_addr,
        "recover_report_pull_addr": recover_report_pull_addr,
    }
    WorkerProc._recovery_addrs_for_spawn = self._recovery_addrs

    self._recovery_handler = RecoveryHandler(
        engine_core=self,
        worker_count=worker_count,
        engine_index=engine_index,
        expect_coordinator=expect_coordinator,
    )

    self._recovery_handler.start()

    _original_engine_core_proc_init(
        self,
        vllm_config,
        local_client,
        handshake_address,
        executor_class,
        log_stats,
        client_handshake_address,
        engine_index=engine_index,
    )

    self._recovery_handler.wait_for_worker_subscriptions()


EngineCoreProc.__init__ = _patched_engine_core_proc_init


_original_process_input_sockets = EngineCoreProc.process_input_sockets


def _patched_process_input_sockets(
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
                    ctx, input_address, zmq.DEALER, identity=identity, bind=False
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

            if self._recovery_handler is not None:
                buffer = coord_socket.recv()
                if buffer.startswith(_RECOVERY_MSG_PREFIX):
                    payload = buffer[len(_RECOVERY_MSG_PREFIX):]
                    try:
                        msg = pickle.loads(payload)
                        if msg[0] == "RECOVERY_ADDRESSES":
                            _, sub_addr, push_addr = msg
                            logger.info(
                                "Received recovery addresses from dp coordinator: sub=%s, push=%s",
                                sub_addr, push_addr,
                            )
                            self._recovery_handler.connect_coordinator(sub_addr, push_addr)
                    except Exception as e:
                        logger.exception("Failed to deserialize recovery addr msg: %s", e)

        ready_event.set()
        del ready_event

        while True:
            for input_socket, _ in poller.poll():
                type_frame, *data_frames = input_socket.recv_multipart(copy=False)

                if input_socket == coord_socket and type_frame.buffer == b"READY":
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


EngineCoreProc.process_input_sockets = _patched_process_input_sockets
