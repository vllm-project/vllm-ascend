# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import threading
import multiprocessing
from datetime import datetime
import logging
import sys
from pathlib import Path
import socket
import requests


def safe_print(directory, message):
    process_id = multiprocessing.current_process().pid
    thread_id = threading.get_ident()

    Path(directory).mkdir(parents=True, exist_ok=True)
    
    filename = f"log_pid_{process_id}_tid_{thread_id}.log"
    filepath = os.path.join(directory, filename)

    logger = logging.getLogger(f"safe_print_{process_id}_{thread_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False # do not print screen
    logger.handlers = []
    
    if not logger.handlers:
        handler = logging.FileHandler(filepath)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.info(message)

def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        return f"Error getting local IP: {e}"

ip_str = get_ip()
trace_output_directory = os.getenv("TRACE_OUTPUT_DIRECTORY", "/tmp/trace_output_directory")