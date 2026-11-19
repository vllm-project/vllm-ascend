# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Local socket protocol for the isolated STAIR planning process."""

import json
import sys
import traceback
from io import BytesIO
from multiprocessing.connection import Connection

import numpy as np

from vllm_ascend.ascend_config import StairConfig

_PLAN_FIELDS = (
    "rank_expert_ids",
    "source_rank_ids",
    "source_slot_ids",
    "predicted_mean_ratios",
    "imbalance_ratios",
)
_REQUEST_FIELDS = (
    "logical_load_values",
    "current_rank_expert_ids",
    "last_committed_mean_ratios",
    "rank_node_ids",
)


def _send_arrays(connection: Connection, arrays: dict[str, np.ndarray]) -> None:
    buffer = BytesIO()
    np.savez(buffer, **arrays)
    connection.send_bytes(buffer.getvalue())


def _receive_arrays(connection: Connection) -> dict[str, np.ndarray]:
    with np.load(BytesIO(connection.recv_bytes()), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _encode_text(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8)


def _decode_text(value: np.ndarray) -> str:
    return value.tobytes().decode("utf-8")


def send_planner_request(connection: Connection, request: tuple) -> None:
    *array_values, config_values, sample_counts = request
    arrays = dict(zip(_REQUEST_FIELDS, map(np.asarray, array_values)))
    arrays["metadata"] = _encode_text(
        json.dumps(
            {
                "config": config_values,
                "sample_counts": None if sample_counts is None else np.asarray(sample_counts).tolist(),
            }
        )
    )
    _send_arrays(
        connection,
        arrays,
    )


def _receive_planner_request(connection: Connection) -> tuple:
    values = _receive_arrays(connection)
    metadata = json.loads(_decode_text(values["metadata"]))
    sample_counts = metadata["sample_counts"]
    return (
        *(values[name] for name in _REQUEST_FIELDS),
        metadata["config"],
        None if sample_counts is None else np.asarray(sample_counts, dtype=np.int64),
    )


def receive_planner_response(connection: Connection) -> tuple[str | None, str | None, tuple | None]:
    values = _receive_arrays(connection)
    error = _decode_text(values.pop("error"))
    if error:
        error_type, _, details = error.partition("\n")
        return error_type, details, None
    return None, None, tuple(values[name] for name in _PLAN_FIELDS)


def _send_planner_response(
    connection: Connection,
    response: tuple[str | None, str | None, tuple | None],
) -> None:
    error_type, error, plan_fields = response
    arrays = {"error": _encode_text(f"{error_type}\n{error}" if error else "")}
    if plan_fields is not None:
        arrays.update(zip(_PLAN_FIELDS, map(np.asarray, plan_fields)))
    _send_arrays(connection, arrays)


def _serve(socket_fd: int) -> None:
    from vllm_ascend.distributed.eplb.policy.stair import StairEplbPolicy

    with Connection(socket_fd) as connection:
        while True:
            try:
                request = _receive_planner_request(connection)
            except EOFError:
                return
            try:
                (
                    logical_load_values,
                    current_rank_expert_ids,
                    last_committed_mean_ratios,
                    rank_node_ids,
                    config_values,
                    sample_counts,
                ) = request
                plan = StairEplbPolicy.plan_rebalance(
                    logical_load_values,
                    current_rank_expert_ids,
                    last_committed_mean_ratios,
                    rank_node_ids,
                    StairConfig(**config_values),
                    sample_counts=sample_counts,
                )
                response: tuple[str | None, str | None, tuple | None] = (
                    None,
                    None,
                    tuple(getattr(plan, name) for name in _PLAN_FIELDS),
                )
            except Exception as error:
                response = (type(error).__name__, traceback.format_exc(), None)
            _send_planner_response(connection, response)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: _stair_process.py SOCKET_FD")
    _serve(int(sys.argv[1]))
