import re
import requests
import json
import argparse
import sys
import paramiko
import time
import logging
import os
from requests import RequestException
from scp import SCPClient
import stat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def strip_blank_chars(input_txt):
    """
    Remove the whitespace characters from the input string
    """
    res = re.sub(r"\s+", "", input_txt)
    return res


def check_table_format(input_text):
    """
    Check if the input string is a table separator
    """
    pattern1 = re.compile(r"\s*-+|-+\s*\|")
    pattern2 = re.compile(r"^\s*\|\s*\|\s*\|\s*")
    pattern3 = re.compile(r"^\s+$")
    pattern4 = re.compile(r"^-+$")
    pattern5 = re.compile(r"^\s*\|\s*$")
    pattern6 = re.compile(r"^_+$")
    pattern7 = re.compile(r"^\s*\|*\s*\?\s*\|*\s*$")
    pattern8 = re.compile(r"|:---:|:---:|")
    pattern9 = re.compile(r"|...|...|")
    if pattern1.search(input_text) is not None:
        return True
    elif pattern2.search(input_text) is not None:
        return True
    elif pattern3.search(input_text) is not None:
        return True
    elif pattern4.search(input_text) is not None:
        return True
    elif pattern5.search(input_text) is not None:
        return True
    elif pattern6.search(input_text) is not None:
        return True
    elif pattern7.search(input_text) is not None:
        return True
    elif pattern8.find(strip_blank_chars(input_text)) != -1:
        return True
    elif pattern9.find(strip_blank_chars(input_text)) != -1:
        return True
    else:
        return False


def check_consecutive_duplicates(text, min_length=3, threshold=3, large_min_length=10, large_repeat=50):
    duplicates = []
    try:
        if not isinstance(text, str):
            raise TypeError("The input must be of string type")
        if not text:
            return False, duplicates, 0, ''
        for length in range(min_length, len(text) // threshold + 1):
            for i in range(len(text) - length * threshold + 1):
                substr = text[i:i + length]
                expected = substr * threshold
                if text.startswith(expected, i):
                    actual_repeats = 0
                    while text.startswith(substr * (actual_repeats + 1), i):
                        actual_repeats += 1
                    if not check_table_format(substr):
                        repeat_type = 'continuous repetition'
                        msg = f"Substring:'{substr}' continuous repetition {actual_repeats} times"
                        print(msg)
                        duplicates = [(substr, actual_repeats, msg, repeat_type)]
                        return True, duplicates, actual_repeats, repeat_type

        text_length = len(text)
        for i in range(text_length - large_min_length + 1):
            substr = text[i:i + large_min_length]
            actual_repeats = text.count(substr)
            if len(strip_blank_chars(substr)) >= 10:
                if actual_repeats > large_repeat:
                    repeat_type = 'non-continuous repetition'
                    msg = f"Substring:'{substr}' non-continuous repetition {actual_repeats} times"
                    duplicates = [(substr, actual_repeats, msg, repeat_type)]
                    print(msg)
                    return True, duplicates, actual_repeats, repeat_type

    except TypeError as e:
        print(f"Type error: {e}")
        return False, [], 0, ''
    except Exception as e:
        print(f"An unexpected error occurred while processing the text: {e}")
        return False, [], 0, ''

    if duplicates:
        for substr, repeats, msg, repeat_type in duplicates:
            return True, duplicates, repeats, repeat_type
    else:
        return False, [], 0, ''


def check_garble_code(text: str):
    if len(text) == 0:
        return False

    control_chars = re.findall(f'\x00-\x08\x0b\x0c\x0e-\x1f\x7f', text)
    if len(control_chars) > 0:
        return True

    if '\ufffd' in text:
        return True

    valid_pattren = re.compile(r'[\u4e00-\u9fff_a-zA-Z0-9，。,./;:\'\"(){}[\\\]<>?！@#$%^&*+-= \t\n]')
    valid_chars = valid_pattren.findall(text)
    total_length = len(text)
    if total_length == 0:
        return False
    valid_radio = len(valid_chars) / total_length
    if valid_radio < 0.7:
        return True

    return False


def _call(url, body):
    """
    Send a request
    """
    model_url = url
    headers = {
        "Content-Type": "application/json"
    }
    model_answer = ""
    reasoning_content = ""
    finish_reason = ""
    try:
        with requests.post(model_url, json=body, headers=headers, verify=False, stream=True) as response:
            if response.status_code != 200:
                print(response.content)
                raise Exception("Request failed with status code", response.status_code)

            for line in response.iter_lines():
                if line:
                    temp = line.decode('utf-8')

                try:
                    if finish_reason in ["length", "stop"]:
                        break
                    temp = json.loads(temp)
                    delta = temp["choices"][0]["message"]
                    if "content" in delta.keys() and delta["content"] != None:
                        model_answer += delta["content"]
                    elif "reasoning_content" in delta.keys():
                        reasoning_content += delta["reasoning_content"]
                    if "finish_reason" in temp["choices"][0].keys():
                        finish_reason = temp["choices"][0]["finish_reason"]
                except json.JSONDecodeError as e:
                    print(e)
        return reasoning_content, model_answer, finish_reason
    except Exception as e:
        print("request failed:", e)
        sys.exit(1)
        return reasoning_content, model_answer, finish_reason


def create_ssh_client(hostname, port, username, password=None, key_filename=None):
    """Create and return an SSH client connection"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        if key_filename:
            # Use private key for authentication
            private_key = paramiko.RSAKey.from_private_key_file(key_filename)
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


def execute_ansible_script(ssh_client, inventory_path, playbook_path, tag=None):
    """Run Ansible deployment on remote server via SSH"""
    if tag is None:
        ansible_cmd = (
            f'ansible-playbook -i {inventory_path} {playbook_path}'
        )
    else:
        ansible_cmd = (
            f'ansible-playbook -i {inventory_path} {playbook_path} --tags "{tag}"'
        )
    print(f"<<< ansible_cmd: {ansible_cmd}")
    exit_status = _execute_ssh_command(ssh_client, ansible_cmd, description="Ansible deployment", get_pty=True)
    return exit_status == 0


def _execute_ssh_command(ssh_client, command, description, get_pty=False, timeout=180) -> int:
    """Execute a command on remote server and stream output in real-time"""
    if description:
        logging.info(f"<<< Executing remote command ({description}): {command}")
    else:
        logging.info(f"<<< Executing remote command: {command}")
    try:
        stdin, stdout, stderr = ssh_client.exec_command(command, get_pty=get_pty, timeout=timeout)
        # Stream stdout in real-time
        output_lines = []
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
        error_lines = stderr.read().decode('utf-8')
        exit_status = stdout.channel.recv_exit_status()
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

def wait_for_model_api(host, port, model_name, timeout=120, wait_interval=10):
    """
    Wait for inference service to become ready by calling Completions API
    Increased retry interval to 30 seconds, extended timeout to 10 minutes
    """
    logging.info(
        f"<<< Waiting for inference service {host}:{port} to be ready (timeout: {timeout}s, "
        f"retry interval: {wait_interval}s)...")

    api_url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Who are you"}],
        "max_tokens": 3
    }
    logging.info(f"<<< API payload: {payload}")
    start_time = time.time()
    check_count = 0

    while time.time() - start_time < timeout:
        check_count += 1
        current_time = time.time()
        elapsed = current_time - start_time

        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=json.dumps(payload),
                timeout=120
            )
            # Check response status code
            if response.status_code == 200:
                logging.info(f"<<< Service is ready! (elapsed: {elapsed:.1f}s, checks: {check_count})")
                try:
                    # Try to parse response content
                    response_data = response.json()
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

def execute_request(ip, port, model_name, target_dir):
    body = {
        "messages": [
            {
                "role": "user",
                "content": "赵女士买了一些水果和小食品准备去看望一个朋友，谁知，这些水果和小食品被他们的儿子们偷吃了，但她不知道是哪个儿子，为此，赵女士非常生气，就盘问4个儿子谁偷吃了水果和小食品。老大说道：“是老二吃的”，老二说道：“是老四偷吃的”，老三说道：“反正我没有偷吃”，老四说道：“老二在说谎”。就这四个儿子中只有一个人说了实话，其他3个都在撒谎。那么，到底是是谁偷吃了这些水果和小食品？"
            }
        ],
        "model": model_name,
        "max_tokens": 2100
    }
    url = f"http://{ip}:{port}/v1/chat/completions"
    #Judge content truncation, check for content duplication, and inspect for garbled answers.
    reasoning_content, answer, finish_reason = _call(url, body)
    repeat, _, _, _ = check_consecutive_duplicates(answer)
    is_garble_code = check_garble_code(answer)

    if finish_reason == "length" or repeat or is_garble_code:
        res = 'BAD:'
        if finish_reason == "length":
            res = res + ",content truncation"
        if repeat:
            res = res + ",content duplication"
        if is_garble_code:
            res = res + ",content garbled"
    else:
        res = "GOOD"
    print(res)
    # write the results to the log file
    os.makedirs(target_dir, exist_ok=True)

    # construct a complete file path
    file_path = os.path.join(target_dir, "accuracy_result_log.txt")

    # prepare the content to be written
    content = f"reasoning_content: {reasoning_content}\n"
    content += f"answer: {answer}\n"
    content += f"finish_reason: {finish_reason}\n"
    content += f"accuracy: {res}\n"

    # write to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"The result has been successfully written to {file_path}")

def wait_for_accuracy(ssh_client, src_dir, timeout= 40 * 60):
    """Wait for accuracy result files  to appear in remote directory, timeout in seconds"""
    start_time: float = time.time()
    logging.info(f"<<< Monitoring {src_dir} for accuracy_result_log files, timeout: {timeout} seconds")

    while True:
        # Execute remote command to find files
        cmd: str = f"ls {src_dir}/accuracy_result_log.txt 2>/dev/null || true"
        stdin, stdout, stderr = ssh_client.exec_command(cmd)
        result = stdout.read().decode().strip()

        if result:
            logging.info(f"<<< Found accuracy result files: {result}")
            return True

        if time.time() - start_time > timeout:
            logging.error("<<< Timeout: No accuracy_result_log files files found")
            return False

        logging.info(f"<<< Sleeping 30s...")
        time.sleep(30)

def copy_remote_dir(ssh_client, src_dir, target_dir):
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


def prepare_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Performance accuracy script")
    # accuracy parameters
    parser.add_argument("--accuracy-src-dir", type=str, required=True,
                        help="Path to accuracy result files on global proxy machine (e.g., /xx/xx/0808)")
    parser.add_argument("--accuracy-target-dir", type=str, required=True,
                        help="Local path to save accuracy results on Jenkins machine")
    parser.add_argument("--host", type=str, required=True, help="Global proxy host address")
    parser.add_argument("--port", type=int, default=8011, help="Global proxy port")
    parser.add_argument("--model", type=str, required=True, help="Model path")

    # SSH connection parameters
    parser.add_argument("--ansible-ssh-host", type=str, required=True, help="Ansible host address")
    parser.add_argument("--ansible-ssh-port", type=int, default=22, help="Ansible SSH port")
    parser.add_argument("--ansible-ssh-username", type=str, default="root", required=False,
                        help="accuracy server username")
    parser.add_argument("--accuracy-ssh-host", type=str, required=True, help="accuracy server host address")
    parser.add_argument("--accuracy-ssh-port", type=int, default=22, help="accuracy server SSH port")
    parser.add_argument("--accuracy-ssh-username", type=str, default="root", required=False,
                        help="accuracy server username")

    # SSH authentication (mutually exclusive)
    ansible_ssh_auth_group = parser.add_mutually_exclusive_group(required=True)
    ansible_ssh_auth_group.add_argument("--ansible-ssh-password", type=str, help="Ansible server password")
    ansible_ssh_auth_group.add_argument("--ansible-ssh-key-file", type=str, help="Ansible server private key file")

    accuracy_ssh_auth_group = parser.add_mutually_exclusive_group(required=True)
    accuracy_ssh_auth_group.add_argument("--accuracy-ssh-password", type=str, help="accuracy server password")
    accuracy_ssh_auth_group.add_argument("--accuracy-ssh-key-file", type=str,
                                          help="accuracy server private key file")

    # Ansible parameters
    parser.add_argument("--inventory-path", type=str, default="omni_infer_inventory_used_for_1P1D.yml",
                        help="Ansible inventory file path (default: omni_infer_inventory_used_for_1P1D.yml)")
    parser.add_argument("--playbook-path", type=str, default="omni_infer_server_used_for_1P1D.yml",
                        help="Ansible playbook file path (default: omni_infer_server_used_for_1P1D.yml)")
    parser.add_argument("--ansible-tag", type=str, help="Ansible playbook tags")

    # Service startup parameters
    parser.add_argument("--wait-timeout", type=int, default=300,
                        help="Timeout for model service startup in seconds (default: 300)")

    args = parser.parse_args()
    return args

def main():
    args = prepare_args()
    ansible_ssh_client = None
    accuracy_ssh_client = None
    try:
        # 1. Establish an SSH connection
        ansible_ssh_client = create_ssh_client(args.ansible_ssh_host, args.ansible_ssh_port,args.ansible_ssh_username,
                                               password= args.ansible_ssh_password, key_filename= args.ansible_ssh_key_file)
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
            # Execute the Ansible script
            execute_ansible_script(ansible_ssh_client, args.inventory_path, args.playbook_path, args.ansible_tag)
            # Wait for service to become ready
            service_ready = wait_for_model_api(args.host, args.port, args.model, timeout=args.wait_timeout)
            if service_ready:
                logging.info("<<< Service is ready.")
            else:
                logging.info(f"<<< Service startup timeout on attempt {deploy_retry_count + 1}.")
                deploy_retry_count += 1
        if not service_ready:
            logging.info(f"<<< Service startup failed after {max_deploy_retries} retries, exiting.")
            sys.exit(1)
        # 3. execute the request and return the result
        execute_request(args.host, args.port, args.model, args.accuracy_src_dir)
        accuracy_ssh_client = create_ssh_client(args.accuracy_ssh_host, args.accuracy_ssh_port,args.accuracy_ssh_username,
                                                password= args.accuracy_ssh_password, key_filename= args.accuracy_ssh_key_file)
        # 4. Wait for accuracy results and copy them
        if wait_for_accuracy(accuracy_ssh_client, args.accuracy_src_dir):
            copy_remote_dir(accuracy_ssh_client, args.accuracy_src_dir, args.accuracy_target_dir)

    finally:
        # Clean up SSH connections
        if ansible_ssh_client:
            ansible_ssh_client.close()
            logging.info("<<< ansible_ssh_client closed")
        if accuracy_ssh_client:
            accuracy_ssh_client.close()
            logging.info("<<< accuracy_ssh_client closed")

if __name__ == "__main__":
    main()