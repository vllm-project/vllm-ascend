import argparse

import httpx

parser = argparse.ArgumentParser(description="Query vLLM server with a prompt.")
parser.add_argument(
    "--port",
    type=int,
    default=7101,
    help="Port number of the vLLM server (default: 7101)",
)
args = parser.parse_args()

curr_port = args.port
VLLM_SERVER_URL = f"http://localhost:{curr_port}/v1/completions"

PAYLOAD = {
    "model": "inference",
    "prompt": "hello, my name",
    "temperature": 0.2,
    "max_tokens": 100,
    "stream": False,
}

# single blocking inference
response = httpx.post(VLLM_SERVER_URL, json=PAYLOAD, timeout=300.0)

if response.status_code == 200:
    data = response.json()
    output_str = data["choices"][0]["text"]
    print(f"!!! output: {output_str}")
else:
    print(f"Error {response.status_code}: {response.text}")
