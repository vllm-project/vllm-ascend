# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Local socket protocol for the isolated STAIR planning process."""

import json
import socket
import struct
import sys
import traceback
from io import BytesIO
from typing import BinaryIO, cast

import numpy as np

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


def _send_arrays(stream: BinaryIO, arrays: dict[str, np.ndarray]) -> None:
    buffer = BytesIO()
    np.savez(buffer, **arrays)
    payload = buffer.getvalue()
    stream.write(_LENGTH.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _receive_arrays(stream: BinaryIO) -> dict[str, np.ndarray]:
    (size,) = _LENGTH.unpack(_read_exact(stream, _LENGTH.size))
    with np.load(BytesIO(_read_exact(stream, size)), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _encode_text(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8)


def _decode_text(value: np.ndarray) -> str:
    return value.tobytes().decode("utf-8")


def send_planner_request(stream: BinaryIO, request: tuple) -> None:
    (
        logical_load_values,
        current_rank_expert_ids,
        last_committed_mean_ratios,
        rank_node_ids,
        config_values,
        layer_ids,
        sample_counts,
    ) = request
    _send_arrays(
        stream,
        {
            "logical_load_values": np.asarray(logical_load_values),
            "current_rank_expert_ids": np.asarray(current_rank_expert_ids),
            "last_committed_mean_ratios": np.asarray(last_committed_mean_ratios),
            "rank_node_ids": np.asarray(rank_node_ids),
            "config_values": _encode_text(json.dumps(config_values)),
            "layer_ids": np.asarray([] if layer_ids is None else layer_ids, dtype=np.int64),
            "has_layer_ids": np.asarray(layer_ids is not None, dtype=np.bool_),
            "sample_counts": np.asarray([] if sample_counts is None else sample_counts),
            "has_sample_counts": np.asarray(sample_counts is not None, dtype=np.bool_),
        },
    )


def _receive_planner_request(stream: BinaryIO) -> tuple:
    values = _receive_arrays(stream)
    layer_ids = values["layer_ids"].tolist() if values["has_layer_ids"].item() else None
    sample_counts = values["sample_counts"] if values["has_sample_counts"].item() else None
    return (
        values["logical_load_values"],
        values["current_rank_expert_ids"],
        values["last_committed_mean_ratios"],
        values["rank_node_ids"],
        json.loads(_decode_text(values["config_values"])),
        layer_ids,
        sample_counts,
    )


def receive_planner_response(stream: BinaryIO) -> tuple[str | None, str | None, tuple | None]:
    values = _receive_arrays(stream)
    error_type = _decode_text(values["error_type"]) or None
    error = _decode_text(values["error"]) or None
    plan_fields = None
    if values["has_plan"].item():
        plan_fields = tuple(
            values[name] for name in ("rank_expert_ids", "source_rank_ids", "source_slot_ids", "predicted_mean_ratios")
        )
    return error_type, error, plan_fields


def _send_planner_response(
    stream: BinaryIO,
    response: tuple[str | None, str | None, tuple | None],
) -> None:
    error_type, error, plan_fields = response
    arrays = {
        "error_type": _encode_text(error_type or ""),
        "error": _encode_text(error or ""),
        "has_plan": np.asarray(plan_fields is not None, dtype=np.bool_),
    }
    if plan_fields is not None:
        arrays.update(
            zip(
                ("rank_expert_ids", "source_rank_ids", "source_slot_ids", "predicted_mean_ratios"),
                map(np.asarray, plan_fields),
            )
        )
    _send_arrays(stream, arrays)


def _serve(socket_fd: int) -> None:
    from vllm_ascend.distributed.eplb.policy.stair import StairEplbPolicy

    with socket.socket(fileno=socket_fd) as planner_socket, planner_socket.makefile("rwb") as raw_stream:
        stream = cast(BinaryIO, raw_stream)
        while True:
            try:
                request = _receive_planner_request(stream)
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
                response: tuple[str | None, str | None, tuple | None] = (
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
            _send_planner_response(stream, response)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: _stair_process.py SOCKET_FD")
    _serve(int(sys.argv[1]))
