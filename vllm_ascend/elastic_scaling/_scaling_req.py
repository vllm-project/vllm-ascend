import argparse
import socket
import time
from concurrent.futures import ThreadPoolExecutor, wait

import psutil
import requests

HEAD_IP = "localhost"
PORT = 8000
BASE = f"http://{HEAD_IP}:{PORT}"


def get_health():
    r = requests.get(f"{BASE}/health", timeout=10)
    r.raise_for_status()
    return r.json()


def get_model_health():
    r = requests.get(f"{BASE}/model_health", timeout=10)
    r.raise_for_status()
    return r.json()


def get_cluster_status():
    r = requests.get(f"{BASE}/cluster_status", timeout=10)
    r.raise_for_status()
    return r.json()


def post_broadcast_metadata():
    r = requests.post(f"{BASE}/broadcast_metadata", timeout=30)
    r.raise_for_status()
    return r.json()


def post_scaleup(num_npus=2):
    url = f"{BASE}/scaleup?num_npus={num_npus}"
    r = requests.post(url, timeout=300)
    r.raise_for_status()
    return r.json()


def post_scaledown(num_npus=2):
    import subprocess

    url = f"{BASE}/scaledown?num_npus={num_npus}"
    cmd = f'curl -v -X POST "{url}"'
    r = subprocess.run(cmd, shell=True)
    return r


def post_addnpus(num_npus=2):
    r = requests.post(f"{BASE}/addnpus", params={"num_npus": num_npus}, timeout=300)
    r.raise_for_status()
    return r.json()


def post_invoke_method(method_name: str, *args, **kwargs):
    payload = {}
    if args:
        payload["args"] = list(args)
    if kwargs:
        payload["kwargs"] = kwargs
    r = requests.post(
        f"{BASE}/invoke_method",
        params={"method_name": method_name},
        json=payload if payload else None,
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def init_inference_engine(inference_port=7102):
    """Wait until resources are available and initialize inference engine"""
    start_t = time.time()

    requests.post(f"http://localhost:{inference_port}/reload_models")
    requests.post(f"http://localhost:{inference_port}/reload_kvcache")

    reload_t = time.time()
    print(f"!!! zero-copy model & kv cache took {reload_t - start_t:.2f} seconds")


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def kill_process_on_port(port):
    # Find all processes with connections on this port
    for proc in psutil.process_iter(["pid", "connections"]):
        try:
            for conn in proc.info["connections"]:
                if conn.laddr.port == port:
                    print(f"Killing PID {proc.pid} using port {port}")
                    # Terminate process gracefully first
                    proc.terminate()
                    try:
                        proc.wait(timeout=1)
                    except psutil.TimeoutExpired:
                        print(f"PID {proc.pid} did not terminate, killing it")
                        proc.kill()
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    print(f"No process found on port {port}")
    return False


def wait_until_port_free(port, timeout=30, interval=0.5):
    start = time.time()
    while is_port_in_use(port):
        if time.time() - start > timeout:
            print(f"Timeout waiting for port {port} to free")
            return False
        time.sleep(interval)
    print(f"Port {port} is now free")
    return True


# python /mnt/t00926703/deepseek-inferencing/example/scripts/_scaling_req.py --num_scale_units 1 --tp 2 --inference_port 7102 --kill_port 7101
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale up NPUs and initialize inference engine")
    parser.add_argument("--num_scale_units", type=int, default=0, help="Number of scale units (default: 1)")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size (default: 2)")
    parser.add_argument("--inference_port", type=int, default=7101, help="Inference server port (default: 7102)")
    parser.add_argument("--kill_port", type=int, default=7100, help="Inference server port to close(default: 7101)")

    args = parser.parse_args()

    num_scale_units = args.num_scale_units
    TP = args.tp
    inference_port = args.inference_port

    # Scale up
    if num_scale_units > 0:
        print("scaleup:", post_scaleup(num_scale_units * TP))
        print("!!! WORM server init done")

    # Initialize inference engine
    scale_start_t = time.time()
    init_inference_engine(inference_port=inference_port)
    print(f"zero-copy time: {time.time() - scale_start_t:.2f} seconds")
