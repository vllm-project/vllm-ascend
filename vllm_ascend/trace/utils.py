#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import logging
import multiprocessing
import os
from pathlib import Path
import socket
import threading
import time
import uuid
from typing import Any

PREFILL = "prefill"
DECODE = "decode"

_LOGGER_LOCK = threading.Lock()


def safe_print(directory: str, message: str) -> None:
    process_id = multiprocessing.current_process().pid
    thread_id = threading.get_ident()

    Path(directory).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(directory, f"log_pid_{process_id}_tid_{thread_id}.log")

    logger = logging.getLogger(f"trace_safe_print_{process_id}_{thread_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    with _LOGGER_LOCK:
        if not logger.handlers:
            handler = logging.FileHandler(filepath)
            formatter = logging.Formatter("%(asctime)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    logger.info(message)


def get_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("10.255.255.255", 1))
        local_ip = sock.getsockname()[0]
        sock.close()
        return local_ip
    except Exception as exc:  # pragma: no cover - depends on host networking.
        return f"Error getting local IP: {exc}"


ip_str = get_ip()
trace_output_directory = os.getenv(
    "TRACE_OUTPUT_DIRECTORY", "/tmp/trace_output_directory"
)


def get_role() -> str:
    return os.getenv("ROLE", "unknown_role")


def log_action(action: str, request_id: Any, role: str | None = None) -> None:
    safe_print(
        trace_output_directory,
        f"<<<Action: {action}; Timestamp:{time.time()}; "
        f"RequestID:{request_id}; Role:{role or get_role()}_{ip_str}",
    )


def log_raw(message: str) -> None:
    safe_print(trace_output_directory, message)


def _headers_get(raw_request: Any, key: str) -> str | None:
    headers = getattr(raw_request, "headers", None)
    if headers is None:
        return None
    try:
        return headers.get(key)
    except Exception:
        return None


def get_or_set_chat_request_id(request: Any, raw_request: Any | None = None) -> str:
    base_id = _headers_get(raw_request, "X-Request-Id")
    if base_id is None:
        base_id = getattr(request, "request_id", None)
    if base_id is None:
        base_id = uuid.uuid4().hex
        try:
            setattr(request, "request_id", base_id)
        except Exception:
            pass

    request_id = str(base_id)
    if not request_id.startswith("chatcmpl-"):
        request_id = f"chatcmpl-{request_id}"
    try:
        setattr(request, "raw_request_id", request_id)
    except Exception:
        pass
    return request_id


def request_id_from_obj(obj: Any, default: str = "unknown") -> str:
    for attr in ("request_id", "req_id", "raw_request_id"):
        value = getattr(obj, attr, None)
        if value is not None:
            return str(value)
    return default
