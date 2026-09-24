# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Local socket protocol for the isolated STAIR planning process."""

import pickle
import socket
import struct
import sys
import traceback
from typing import BinaryIO

from vllm_ascend.ascend_config import StairConfig

_LENGTH = struct.Struct("!Q")


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("STAIR planner socket closed")
        chunks.extend(chunk)
    return bytes(chunks)


def _send(stream: BinaryIO, value) -> None:
    # Both endpoints are trusted processes created from one local socketpair.
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_LENGTH.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _receive(stream: BinaryIO):
    (size,) = _LENGTH.unpack(_read_exact(stream, _LENGTH.size))
    return pickle.loads(_read_exact(stream, size))


def send_planner_request(stream: BinaryIO, request: tuple) -> None:
    _send(stream, request)


def receive_planner_response(stream: BinaryIO) -> tuple[str | None, str | None, tuple | None]:
    return _receive(stream)


def _serve(socket_fd: int) -> None:
    from vllm_ascend.distributed.eplb.policy.stair import StairEplbPolicy

    with socket.socket(fileno=socket_fd) as planner_socket, planner_socket.makefile("rwb") as stream:
        while True:
            try:
                request = _receive(stream)
            except EOFError:
                return
            try:
                (
                    logical_load_values,
                    current_rank_expert_ids,
                    last_committed_mean_ratios,
                    rank_node_ids,
                    config_values,
                    layer_ids,
                    sample_counts,
                ) = request
                plan = StairEplbPolicy.plan_rebalance(
                    logical_load_values,
                    current_rank_expert_ids,
                    last_committed_mean_ratios,
                    rank_node_ids,
                    StairConfig(**config_values),
                    layer_ids=layer_ids,
                    sample_counts=sample_counts,
                )
                response = (
                    None,
                    None,
                    (
                        plan.rank_expert_ids,
                        plan.source_rank_ids,
                        plan.source_slot_ids,
                        plan.predicted_mean_ratios,
                    ),
                )
            except Exception as error:
                response = (type(error).__name__, traceback.format_exc(), None)
            _send(stream, response)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: _stair_process.py SOCKET_FD")
    _serve(int(sys.argv[1]))
