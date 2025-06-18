import os
import time
import subprocess
import signal
import psutil
import requests
import pytest
import random
import json


def kill_process_and_children(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            print(f"Killing child process {child.pid}")
            child.kill()
        print(f"Killing parent process {pid}")
        parent.kill()
    except psutil.NoSuchProcess:
        pass


def kill_all_vllm_related():
    current_pid = os.getpid()

    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            if proc.pid == current_pid:
                continue
            cmd = ' '.join(proc.info['cmdline'])
            if "vllm" in cmd or "proxy" in cmd or "engine_worker" in cmd:
                kill_process_and_children(proc.pid)
        except Exception:
            continue


def build_expert_map(expert_map_path, num_redundant_expert=0,
                     num_layer=58, num_device=16, num_original_expert=256,
                     random_seed=42):
    expert_num_list = list(range(num_original_expert))
    random.seed(random_seed)
    if num_redundant_expert > 0:
        expert_num_list = expert_num_list + random.choices(expert_num_list, k=num_redundant_expert)
    local_num_expert = len(expert_num_list) // num_device

    expert_map = {
        "moe_layer_count": num_layer,
        "device_count": num_device,
        "layer_list": []
    }
    for layer_id in range(num_layer):
        random.shuffle(expert_num_list)
        current_expert_distribution = [expert_num_list[i*local_num_expert:(i+1)*local_num_expert] for i in range(num_device)]
        layer_info = {
            "layer_id": layer_id,
            "device_count": num_device,
            "device_list": []
        }
        for device_id in range(num_device):
            layer_info["device_list"].append({
                "device_id": device_id,
                "device_expert": current_expert_distribution[device_id]
            })
        expert_map["layer_list"].append(layer_info)
    with open(expert_map_path, "w") as f:
        json.dump(expert_map, f)


def is_port_in_use(port):
    """Check if a port is currently in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def ensure_port_available(port, timeout=30):
    """Wait for a port to become available."""
    import socket
    start = time.time()
    while time.time() - start < timeout:
        if not is_port_in_use(port):
            return True
        print(f"Port {port} is still in use, waiting...")
        time.sleep(2)
    return False


def wait_for_port(port, timeout=30):
    import socket
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1)
    raise TimeoutError(f"Port {port} not ready after {timeout}s")


SCRIPT_PATH = os.path.abspath("./tests/e2e/run_eplb.sh")
PROXY_PORT = 10102
EXPERT_MAP_PATH = "./tests/e2e/eplb/expert_map.json"


@pytest.mark.parametrize("num_redundant_expert", [0, 16])
def test_eplb_with_redundant_expert(num_redundant_expert):
    # Ensure port is available before starting the test
    if is_port_in_use(PROXY_PORT):
        print(f"Port {PROXY_PORT} is still in use from previous test, waiting for it to become available...")
        if not ensure_port_available(PROXY_PORT, timeout=300):
            pytest.skip(f"Port {PROXY_PORT} is still in use after waiting 60 seconds")
    
    print("Launching bash script to run eplb setup...")
    build_expert_map(EXPERT_MAP_PATH, num_redundant_expert=num_redundant_expert)
    proc = subprocess.Popen(["bash", SCRIPT_PATH, str(num_redundant_expert)])
    try:
        print("Waiting for proxy port to be available...")
        wait_for_port(PROXY_PORT, timeout=600)

        # request
        payload = {
            "model": "Deepseek",
            "prompt": "The future of AI is",
            "max_tokens": 64,
            "temperature": 0,
        }
        response = requests.post(
            f"http://localhost:{PROXY_PORT}/v1/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10)
        assert response.status_code == 200, f"HTTP failed: {response.status_code}"
        result = response.json()
        print("Response:", result)
        assert "text" in result["choices"][0]
        assert len(result["choices"][0]["text"].strip()) > 0
    
    finally:
        # clean up subprocesses
        print("Cleaning up subprocess...")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        if os.path.exists(EXPERT_MAP_PATH):
            os.remove(EXPERT_MAP_PATH)
        kill_all_vllm_related()
        
        # Wait for port to be fully released
        print("Waiting for port to be fully released...")
        time.sleep(3)