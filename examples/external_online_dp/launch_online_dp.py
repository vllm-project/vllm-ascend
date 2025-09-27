import argparse
import multiprocessing
import os
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dp-size",
        type=int,
        required=True,
        help="Data parallel size."
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size."
    )
    parser.add_argument(
        "--dp-size-local",
        type=int,
        default=-1,
        help="Local data parallel size."
    )
    parser.add_argument(
        "--dp-rank-start",
        type=int,
        default=0,
        help="Starting rank for data parallel."
    )
    parser.add_argument(
        "--dp-address",
        type=str,
        required=True,
        help="IP address for data parallel master node."
    )
    parser.add_argument(
        "--dp-rpc-port",
        type=str,
        default=12345,
        help="Port for data parallel master node."
    )
    parser.add_argument(
        "--vllm-start-port",
        type=int,
        default=9000,
        help="Starting port for the engine."
    )
    return parser.parse_args()

args = parse_args()
dp_size = args.dp_size
tp_size = args.tp_size
dp_size_local = args.dp_size_local
if dp_size_local == -1:
    dp_size_local = dp_size
dp_rank_start = args.dp_rank_start
dp_address = args.dp_address
dp_rpc_port = args.dp_rpc_port
vllm_start_port = args.vllm_start_port

def run_command(visiable_devices, dp_rank, vllm_engine_port):
    command = [
        "bash",
        "./run_dp_template.sh",
        visiable_devices,
        str(vllm_engine_port),
        str(dp_size),
        str(dp_rank),
        dp_address,
        dp_rpc_port,
        str(tp_size),
    ]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    template_path = "./run_dp_template.sh"
    if not os.path.exists(template_path):
        print(f"Template file {template_path} does not exist.")
        sys.exit(1)

    processes = []
    num_cards = dp_size_local * tp_size
    for i in range(dp_size_local):
        dp_rank = dp_rank_start + i
        vllm_engine_port = vllm_start_port + i
        visiable_devices = ",".join(str(x) for x in range(i * tp_size, (i + 1) * tp_size))
        process = multiprocessing.Process(target=run_command,
                                        args=(visiable_devices, dp_rank,
                                                vllm_engine_port))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()