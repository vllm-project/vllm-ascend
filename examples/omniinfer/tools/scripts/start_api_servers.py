#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import argparse
import glob
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import weakref

# Get the terminal width
terminal_width = shutil.get_terminal_size().columns


def is_port_available(port, host="0.0.0.0"):
    """Check if a port is available on the specified host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False


def find_available_port(base_port, max_attempts=10, host="0.0.0.0"):
    """Find the next available port starting from base_port."""
    for offset in range(max_attempts):
        port = base_port + offset
        print(f"INFO: Try to bind port {port}.")
        if is_port_available(port, host):
            print(f"INFO: Port {port} is available.")
            return port
        else:
            print(f"WARNING: Port {port} is not available!")

    raise RuntimeError(f"No available port found between {base_port} and {base_port + max_attempts - 1}")


class ProcessManager:
    """Class to hold processes and enable weakref.finalize."""
    def __init__(self, processes):
        self.processes = processes

def use_vllm_logging_config_path():
    config_path = os.getenv("VLLM_LOGGING_CONFIG_PATH")
    return int(os.getenv("VLLM_CONFIGURE_LOGGING", "1")) and config_path and os.path.exists(config_path)

def replace_logger_file(config_json, logger_file_path):
    for handler_name, handler_config in config_json.get("handlers", {}).items():
        if "filename" in handler_config:
            handler_config["filename"] = logger_file_path
    return config_json


def start_single_node_api_servers(
    num_servers,
    model_path,
    base_api_port,
    master_ip,
    master_port,
    total_dp_size,
    gpu_util,
    block_size,
    tp,
    served_model_name,
    server_offset=0,
    kv_transfer_config=None,
    log_dir="logs",
    max_port_attempts=10,
    max_tokens=4096,
    extra_args=None,
    additional_config=None,
    enable_mtp=False,
    no_enable_prefix_caching=False,
    num_speculative_tokens=1,
    no_enable_chunked_prefill=False,
):
    """Start multiple VLLM API servers with specified configurations."""

    # Hard code dp=1, cuz current we want one api server one DP
    dp_per_server = 1

    if additional_config:
        try:
            json.loads(additional_config)
        except json.JSONDecodeError as e:
            raise ValueError(
                "additional_config must be a valid JSON string, e.g., '{\"key\":\"value\"}'"
            ) from e

    os.makedirs(log_dir, exist_ok=True)
    processes = []

    # Check if base api port is available. Raise error if it's unavailable.
    if not is_port_available(base_api_port):
        raise RuntimeError(
            f"Port {base_api_port} is not available. "
            "Use --base_api_port to specify a different port, or terminate the process using this port."
        )

    for rank in range(num_servers):
        # Set environment variables for each server
        env = os.environ.copy()
        env["VLLM_DP_SIZE"] = str(total_dp_size)
        env["VLLM_DP_RANK"] = str(rank + server_offset // tp)
        env["VLLM_DP_RANK_LOCAL"] = str(rank + server_offset // tp)
        env["VLLM_DP_MASTER_IP"] = master_ip
        env["VLLM_DP_MASTER_PORT"] = str(master_port)

        # Find an available port
        try:
            port = find_available_port(base_api_port + rank, max_attempts=max_port_attempts)
        except RuntimeError as e:
            print(f"Error: {e}")
            cleanup_processes(processes)
            sys.exit(1)

        # Construct the vllm serve command
        cmd = [
            "vllm", "serve", model_path,
            "--trust-remote-code",
            "--gpu-memory-utilization", str(gpu_util),
            "--block_size", str(block_size),
            "--tensor-parallel-size", str(tp),
            "--data-parallel-size", str(dp_per_server),   # one engine core for one dp
            "--data-parallel-size-local", "1",            # 'Number of data parallel replicas '
            "--data-parallel-address", master_ip,         # 'Address of data parallel cluster '
            "--data-parallel-rpc-port", str(master_port), # 'Port for data parallel RPC '
            "--port", str(port),
            "--served-model-name", served_model_name,
            "--max-model-len", str(max_tokens)
        ]
        if enable_mtp:
            cmd.extend(["--speculative_config", '{"method": "deepseek_mtp", "num_speculative_tokens": 1}'])
        if kv_transfer_config:
            cmd.extend(["--kv-transfer-config", str(kv_transfer_config)])
        if extra_args:
            cmd.extend(extra_args.split())
        if additional_config:
            cmd.extend(["--additional-config", additional_config])
        if no_enable_prefix_caching:
            cmd.extend(["--no-enable-prefix-caching"])
        if no_enable_chunked_prefill:
            cmd.extend(["--no-enable-chunked-prefill"])

        logger_path = os.path.join(log_dir, f"server_{rank}.log")
        existed_logger_files = [f for f in glob.glob(logger_path + "*") if re.search(f"server_{rank}\.log(\.\d+)?$", f)]
        for existed_file in existed_logger_files:
            print(f"The historical log {existed_file} that already exists will be removed.")
            os.remove(existed_file)

        if use_vllm_logging_config_path():
            with open(os.getenv("VLLM_LOGGING_CONFIG_PATH"), "r") as f:
                config_json = json.load(f)
            config_json = replace_logger_file(config_json, logger_path)
            print(f"Use VLLM_LOGGING_CONFIG: {config_json}")

            tmp_file_name = tempfile.mkstemp()[1]
            tmp_file = open(tmp_file_name, "w")
            json.dump(config_json, tmp_file)
            tmp_file.close()

            env["VLLM_LOGGING_CONFIG_PATH"] = tmp_file_name
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL
            # occupy space
            log_file = tmp_file
        else:
            # Open a single log file for combined stdout and stderr
            log_file = open(logger_path, "w")

            stdout = log_file
            stderr = subprocess.STDOUT  # Redirect stderr to stdout (same log file)

        # Start the server process in the background with combined log redirection
        print('=' * terminal_width)
        print(f"Starting API server {rank} on port {port}, logging to {logger_path}")
        print('=' * terminal_width)
        print(f"Server {rank} on port {port}>>>{' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=stdout,
            stderr=stderr  # Redirect stderr to stdout (same log file)
        )
        processes.append((process, log_file))

    process_manager = ProcessManager(processes)

    # Define cleanup function for weakref.finalize
    def cleanup_processes():
        for i, (proc, log) in enumerate(process_manager.processes):
            if proc.poll() is None:  # Process is still running
                print(f"Cleaning up: Terminating server {i} (PID: {proc.pid})")
                proc.terminate()
                try:
                    proc.wait(timeout=5)  # Wait up to 5 seconds for clean exit
                except subprocess.TimeoutExpired:
                    proc.kill()  # Force kill if timeout occurs
                    print(f"API Server {i} did not terminate gracefully, killed")
            log.close()
            print(f"Closed log file for server {i}")

    # Set up finalizer for garbage collection
    weakref.finalize(process_manager, cleanup_processes)

    # Provide feedback on how to monitor logs
    print('-' * terminal_width)
    print(f"Started {num_servers} servers. Logs are in {log_dir}/")
    print(f"Run 'tail -f {log_dir}/server_*.log' to monitor logs in real-time.")
    return processes, process_manager


def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) to cleanly exit."""
    print("\nReceived SIGINT, shutting down servers...")
    for i, (proc, log) in enumerate(process_manager.processes):
        if proc.poll() is None:
            print(f"Terminating API server {i} (PID: {proc.pid})")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"API Server {i} did not terminate gracefully, killed")
        log.close()
        print(f"Closed log file for server {i}")
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description=(
        "Start multiple VLLM API servers with combined "
        "logging, cleanup, and port checking."
    ))
    parser.add_argument("--num-servers", type=int, default=2, help="Number of API servers to start")
    parser.add_argument("--num-dp", type=int, default=None, help="Number of data parallel size.")
    parser.add_argument("--server-offset", type=int, default=0, help="Server offset for multi-nodes")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--base-api-port", type=int, default=9000, help="Base port for the first API server")
    parser.add_argument("--master-ip", type=str, required=True, help="Master IP for data parallelism")
    parser.add_argument("--master-port", type=int, default=8000, help="Master port for data parallelism")
    parser.add_argument(
        "--gpu-memory-utilization", "--gpu-util", 
        dest='gpu_util', 
        type=float, 
        default=0.9, 
        help="GPU memory utilization")
    parser.add_argument("--block-size", type=int, default=128, help="Block size for VLLM")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--served-model-name", type=str, required=True, help="Name of the served model")
    parser.add_argument("--max-model-len", default=16384, type=int, help="max number of tokens")
    parser.add_argument("--max-port-attempts", type=int, default=20, help="Max attempts to find an available port")
    parser.add_argument("--kv-transfer-config", type=str, default="", help="kv transfer config for VLLM")
    parser.add_argument(
        "--extra-args", 
        type=str, 
        default="", 
        help="Additional VLLM arguments (space-separated, e.g., '--enable-expert-parallel')")
    parser.add_argument(
        "--additional-config", 
        type=str, 
        default="", 
        help="JSON-formatted additional platform-specific config, e.g., '{\"key\":\"value\"}'")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to store log files")
    parser.add_argument("--enable-mtp", default=False, action='store_true')
    parser.add_argument("--no-enable-prefix-caching", default=False, action="store_true")
    parser.add_argument("--no-enable-chunked-prefill", default=False, action="store_true")
    parser.add_argument("--num-speculative-tokens", type=int, default=1)

    args = parser.parse_args()
    if not args.num_dp:
        args.num_dp = args.num_servers
    if args.num_dp < args.num_servers:
        raise ValueError(
            "Number of DP should be larger or eaqual to number of API servers."
        )

    processes, process_manager = start_single_node_api_servers(
        num_servers=args.num_servers,
        model_path=args.model_path,
        base_api_port=args.base_api_port,
        master_ip=args.master_ip,
        master_port=args.master_port,
        total_dp_size=args.num_dp,
        server_offset=args.server_offset,
        no_enable_prefix_caching=args.no_enable_prefix_caching,
        gpu_util=args.gpu_util,
        block_size=args.block_size,
        tp=args.tp,
        served_model_name=args.served_model_name,
        log_dir=args.log_dir,
        max_port_attempts=args.max_port_attempts,
        kv_transfer_config=args.kv_transfer_config,
        max_tokens=args.max_model_len, 
        extra_args=args.extra_args,
        additional_config=args.additional_config,
        enable_mtp=args.enable_mtp,
        num_speculative_tokens=args.num_speculative_tokens,
        no_enable_chunked_prefill=args.no_enable_chunked_prefill
    )

    # Register SIGINT handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Keep the script running to allow servers to operate
    print(f"{args.num_servers} API servers are running. Press Ctrl+C to stop.")
    try:
        server_down=False
        while True:
            time.sleep(1)  # Keep script alive, check processes periodically
            for i, (proc, _) in enumerate(processes):
                if proc.poll() is not None:
                    print(
                        f"API Server {i} (PID: {proc.pid}) stopped with exit code {proc.returncode}. "
                        f"Check {args.log_dir}/server_{i}.log for details."
                    )
                    server_down=True
            if server_down:
                break
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
