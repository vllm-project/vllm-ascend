import os
import signal
import subprocess
import time

import requests

PROXY_PORT = 8192
REGISTER_PORT = 8193

SCRIPT_PATH = os.path.abspath("./tests/e2e/run_disagg_pd.sh")


def wait_for_port(port, timeout=30):
    import socket
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1)
    raise TimeoutError(f"Port {port} not ready after {timeout}s")


def start_and_test_pipeline():
    print("Launching bash script to run vLLM PD setup...")
    proc = subprocess.Popen(["bash", SCRIPT_PATH])
    try:
        print("Waiting for proxy port to be available...")
        wait_for_port(PROXY_PORT, 1200)

        # request
        prompt = "The future of AI is"
        payload = {
            "model": "Deepseek/DeepSeek-V2-Lite-Chat",
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0,
        }
        response = requests.post(f"http://localhost:{PROXY_PORT}/generate",
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


def test_disaggregated_pd_pipeline():
    start_and_test_pipeline()
