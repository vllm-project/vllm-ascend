import os
import subprocess
from argparse import ArgumentParser

def main():
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    port = 25000 + global_rank
    npu_device = local_rank

    # 启动vLLM服务
    cmd = [
        "vllm serve",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "--host", "0.0.0.0",
        "--seed" "1024",
        "--served-model-name" "Qwen",
        "--max-model-len" "2000",
        "--max-num-batched-tokens" "2000",
        "--trust-remote-code",
        "--gpu-memory-utilization" "0.9",
        "--tensor-parallel-size", "1",
        "--port", str(port),
    ]

    # 设置进程独有环境变量
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(npu_device)
    env["VLLM_DP_SIZE"] = str(world_size)
    env["VLLM_DP_RANK"] = str(global_rank)
    env["VLLM_HTTP_PORT"] = str(port)

    # 启动子进程
    process = subprocess.Popen(cmd, env=env)
    process.wait()

if __name__ == "__main__":
    main()