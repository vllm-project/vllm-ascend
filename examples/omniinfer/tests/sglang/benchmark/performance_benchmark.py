# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from argparse import Namespace, ArgumentParser
import json
import stat
from typing import Optional, List, Dict, Any

import paramiko
import os
import sys
import time
import requests
import logging
from paramiko.client import SSHClient
from paramiko.rsakey import RSAKey

from requests import RequestException
from scp import SCPClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def _create_ssh_client(hostname: str, port: int, username: str, password: Optional[str] = None,
                       key_filename: Optional[str] = None) -> Optional[SSHClient]:
    """Create and return an SSH client connection"""
    client: SSHClient = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        if key_filename:
            # Use private key for authentication
            private_key: RSAKey = paramiko.RSAKey.from_private_key_file(key_filename)
            client.connect(hostname, port=port, username=username, pkey=private_key)
            logging.info(f"<<< Successfully connected to server using key {hostname}:{port}")
        elif password:
            # Use password for authentication
            client.connect(hostname, port=port, username=username, password=password)
            logging.info(f"<<< Successfully connected to server using password {hostname}:{port}")
        else:
            raise ValueError("Must provide either password or key file path")

        return client
    except Exception as exception:
        logging.exception(f"<<< SSH connection failed: {str(exception)}")
        return None


def _execute_ssh_command(ssh_client: SSHClient, command: str, description: str = "",
                         get_pty: bool = False, timeout: int = 180) -> int:
    """Execute a command on remote server and stream output in real-time"""
    if description:
        logging.info(f"<<< Executing remote command ({description}): {command}")
    else:
        logging.info(f"<<< Executing remote command: {command}")
    try:
        stdin, stdout, stderr = ssh_client.exec_command(command, get_pty=get_pty, timeout=timeout)
        # Stream stdout in real-time
        output_lines: List[str] = []
        while not stdout.channel.exit_status_ready():
            if stdout.channel.recv_ready():
                line: str = stdout.readline()
                if line:
                    output_lines.append(line)
                    logging.info(line)  # Print to console in real-time
                    sys.stdout.flush()  # Flush output immediately
            else:
                time.sleep(0.1)
        # Read remaining output (handle buffered data)
        for line in stdout:
            output_lines.append(line)
            logging.info(line)
            sys.stdout.flush()
        # Read error output
        error_lines: str = stderr.read().decode('utf-8')
        exit_status: int = stdout.channel.recv_exit_status()
    except Exception as exception:
        logging.exception(f"<<< SSH command execution failed: {str(exception)}")
        return -1

    if exit_status == 0:
        logging.info(f"\n<<< Remote command executed successfully ({description})")
    else:
        logging.info(f"\n<<< Remote command failed ({description}), exit status: {exit_status}")
        if error_lines.strip():
            logging.info(f"<<< Remote command error output ({description}):\n{error_lines}")

    return exit_status


def _wait_for_model_api(args: Namespace, timeout: int = 10 * 60, wait_interval: int = 30) -> bool:
    """
    Wait for inference service to become ready by calling Completions API
    Increased retry interval to 30 seconds, extended timeout to 10 minutes
    """
    host: str = args.host
    port: int = args.port
    model_name: str = args.model
    logging.info(
        f"<<< Waiting for inference service {host}:{port} to be ready (timeout: {timeout}s, "
        f"retry interval: {wait_interval}s)...")

    api_url: str = f"http://{host}:{port}/v1/chat/completions"
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Who are you"}],
        "max_tokens": 3
    }
    logging.info(f"<<< API payload: {payload}")
    start_time: float = time.time()
    check_count: int = 0

    while time.time() - start_time < timeout:
        check_count += 1
        current_time: float = time.time()
        elapsed: float = current_time - start_time

        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=20
            )
            # Check response status code
            if response.status_code == 200:
                logging.info(f"<<< Service is ready! (elapsed: {elapsed:.1f}s, checks: {check_count})")
                try:
                    # Try to parse response content
                    response_data: Any = response.json()
                    logging.info(f"<<< API response content: {response_data}")
                except json.JSONDecodeError as e:
                    logging.warning(f"<<< Failed to parse JSON response but service returned 200 status: {str(e)}")
                return True

            logging.info(
                f"<<< Check #{check_count}: Service response status {response.status_code} (elapsed: {elapsed:.1f}s)")

        except RequestException as e:
            logging.warning(
                f"<<< Check #{check_count}: Connection error - {type(e).__name__}: {str(e)} (elapsed: {elapsed:.1f}s)")

        # Wait before next check
        logging.info(f"<<< Sleeping {wait_interval}s...")
        time.sleep(wait_interval)

    # Handle timeout
    logging.error(f"<<< Timeout: Inference service {host}:{port} not ready after {timeout} seconds")
    return False


def _run_ansible_deployment(ssh_client: SSHClient, inventory_path: str, playbook_path: str,
                            tag: Optional[str] = None) -> bool:
    """Run Ansible deployment on remote server via SSH"""
    if tag is None:
        ansible_cmd = (
            f'ansible-playbook -i {inventory_path} {playbook_path}'
        )
    else:
        ansible_cmd = (
            f'ansible-playbook -i {inventory_path} {playbook_path} --tags "{tag}"'
        )

    exit_status: int = _execute_ssh_command(ssh_client, ansible_cmd, description="Ansible deployment", get_pty=True)
    return exit_status == 0


def _wait_for_benchmark(ssh_client: SSHClient, src_dir: str, timeout: int = 15 * 60) -> bool:
    """Wait for benchmark result files (*.jsonl) to appear in remote directory, timeout in seconds"""
    start_time: float = time.time()
    logging.info(f"<<< Monitoring {src_dir} for *.jsonl files, timeout: {timeout} seconds")

    while True:
        # Execute remote command to find files
        cmd: str = f"ls {src_dir}/*.jsonl 2>/dev/null || true"
        stdin, stdout, stderr = ssh_client.exec_command(cmd)
        result = stdout.read().decode().strip()

        if result:
            logging.info(f"<<< Found benchmark result files: {result}")
            return True

        if time.time() - start_time > timeout:
            logging.error("<<< Timeout: No *.jsonl files found")
            return False

        logging.info(f"<<< Sleeping 30s...")
        time.sleep(30)


def _copy_remote_dir(ssh_client: SSHClient, src_dir: str, target_dir: str) -> bool:
    """
    Recursively copy remote directory to local using existing SSH connection
    Reuses existing authentication without additional configuration
    """
    logging.info(f"<<< Copying remote directory {src_dir} to local {target_dir}")
    # Ensure local target directory exists
    os.makedirs(target_dir, exist_ok=True)
    try:
        # Create SCPClient using existing SSH transport
        with SCPClient(ssh_client.get_transport()) as scp:
            # Verify remote path is a directory
            sftp = ssh_client.open_sftp()
            try:
                remote_attr = sftp.stat(src_dir)
                if not stat.S_ISDIR(remote_attr.st_mode):
                    logging.error(f"<<< Error: {src_dir} is not a remote directory")
                    return False
            except FileNotFoundError as e:
                logging.exception(f"<<< Error: Remote directory {src_dir} does not exist, {str(e)}")
                return False
            finally:
                sftp.close()
            # Recursively copy remote directory to local
            # preserve_times=True keeps original file timestamps
            scp.get(src_dir, local_path=target_dir, recursive=True, preserve_times=True)

        logging.info(f"<<< Directory copy complete, local path: {target_dir}")
        return True

    except Exception as e:
        logging.exception(f"<<< Directory copy failed: {str(e)}")
        return False


def _get_args() -> Namespace:
    """Parse command line arguments"""
    parser = ArgumentParser(description="Performance benchmark script")
    # Benchmark parameters
    parser.add_argument("--benchmark-src-dir", type=str, required=True,
                        help="Path to benchmark result files on global proxy machine (e.g., /xx/xx/0808)")
    parser.add_argument("--benchmark-target-dir", type=str, required=True,
                        help="Local path to save benchmark results on Jenkins machine")
    parser.add_argument("--host", type=str, required=True, help="Global proxy host address")
    parser.add_argument("--port", type=int, default=8011, help="Global proxy port")
    parser.add_argument("--model", type=str, required=True, help="Model path")

    # SSH connection parameters
    parser.add_argument("--ansible-ssh-host", type=str, required=True, help="Ansible host address")
    parser.add_argument("--ansible-ssh-port", type=int, default=22, help="Ansible SSH port")
    parser.add_argument("--ansible-ssh-username", type=str, default="root", required=False,
                        help="Benchmark server username")
    parser.add_argument("--benchmark-ssh-host", type=str, required=True, help="Benchmark server host address")
    parser.add_argument("--benchmark-ssh-port", type=int, default=22, help="Benchmark server SSH port")
    parser.add_argument("--benchmark-ssh-username", type=str, default="root", required=False,
                        help="Benchmark server username")

    # SSH authentication (mutually exclusive)
    ansible_ssh_auth_group = parser.add_mutually_exclusive_group(required=True)
    ansible_ssh_auth_group.add_argument("--ansible-ssh-password", type=str, help="Ansible server password")
    ansible_ssh_auth_group.add_argument("--ansible-ssh-key-file", type=str, help="Ansible server private key file")

    benchmark_ssh_auth_group = parser.add_mutually_exclusive_group(required=True)
    benchmark_ssh_auth_group.add_argument("--benchmark-ssh-password", type=str, help="Benchmark server password")
    benchmark_ssh_auth_group.add_argument("--benchmark-ssh-key-file", type=str,
                                          help="Benchmark server private key file")

    # Ansible parameters
    parser.add_argument("--inventory-path", type=str, default="omni_infer_inventory_used_for_1P1D.yml",
                        help="Ansible inventory file path (default: omni_infer_inventory_used_for_1P1D.yml)")
    parser.add_argument("--playbook-path", type=str, default="omni_infer_server_used_for_1P1D.yml",
                        help="Ansible playbook file path (default: omni_infer_server_used_for_1P1D.yml)")

    # Service startup parameters
    parser.add_argument("--wait-timeout", type=int, default=120,
                        help="Timeout for model service startup in seconds (default: 120)")

    args: Namespace = parser.parse_args()
    return args


def _deploy(ansible_ssh_client: SSHClient, args: Namespace):
    """Execute deployment process"""
    deployment_success: bool = _run_ansible_deployment(ansible_ssh_client, args.inventory_path, args.playbook_path)
    if not deployment_success:
        logging.info("<<< Ansible deployment failed, exiting.")
        sys.exit(1)
    logging.info("<<< Ansible deployment successful.")


def main():
    """Main execution function"""
    args: Namespace = _get_args()

    ansible_ssh_client: Optional[SSHClient] = None
    benchmark_ssh_client: Optional[SSHClient] = None

    try:
        # 1. Create SSH connection and execute Ansible deployment
        ansible_ssh_client: SSHClient = _create_ssh_client(
            args.ansible_ssh_host, args.ansible_ssh_port, args.ansible_ssh_username,
            password=args.ansible_ssh_password, key_filename=args.ansible_ssh_key_file
        )

        if not ansible_ssh_client:
            logging.info("<<< Failed to establish SSH connection, exiting.")
            sys.exit(1)

        # 2. Deploy and wait for service startup (with retries)
        service_ready: bool = False
        max_deploy_retries: int = 3
        deploy_retry_count: int = 0

        while not service_ready and deploy_retry_count < max_deploy_retries:
            if deploy_retry_count > 0:
                logging.info(f"<<< Service startup failed, retrying deployment (attempt {deploy_retry_count})...")
            else:
                logging.info("<<< Starting initial deployment...")

            # Execute Ansible deployment
            _deploy(ansible_ssh_client, args)

            # Wait for service to become ready
            service_ready = _wait_for_model_api(args, timeout=args.wait_timeout)
            if service_ready:
                logging.info("<<< Service is ready.")
            else:
                logging.info(f"<<< Service startup timeout on attempt {deploy_retry_count + 1}.")
                deploy_retry_count += 1

        if not service_ready:
            logging.info(f"<<< Service startup failed after {max_deploy_retries} retries, exiting.")
            sys.exit(1)

        # 3. Execute performance benchmark
        benchmark_success: bool = _run_ansible_deployment(
            ansible_ssh_client, args.inventory_path, args.playbook_path, "performance_benchmark"
        )
        if benchmark_success:
            logging.info("<<< Performance benchmark started.")
            benchmark_ssh_client: SSHClient = _create_ssh_client(
                args.benchmark_ssh_host, args.benchmark_ssh_port, args.benchmark_ssh_username,
                password=args.benchmark_ssh_password, key_filename=args.benchmark_ssh_key_file
            )
            # Wait for benchmark results and copy them
            if _wait_for_benchmark(benchmark_ssh_client, args.benchmark_src_dir):
                _copy_remote_dir(benchmark_ssh_client, args.benchmark_src_dir, args.benchmark_target_dir)
            sys.exit(0)
        else:
            logging.info("<<< Performance benchmark failed.")
            sys.exit(1)

    finally:
        # Clean up SSH connections
        if ansible_ssh_client:
            ansible_ssh_client.close()
            logging.info("<<< ansible_ssh_client closed")
        if benchmark_ssh_client:
            benchmark_ssh_client.close()
            logging.info("<<< benchmark_ssh_client closed")


if __name__ == "__main__":
    main()