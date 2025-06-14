import os
import torch.distributed as dist
import json
import argparse
from vllm_ascend.soc_info import NPUSocInfo

parser = argparse.ArgumentParser(
    description="Arguments of rank table generator",
)
parser.add_argument(
    "--prefill-device-cnt", type=int, required=True, help="number of prefill devices"
)
parser.add_argument(
    "--decode-device-cnt", type=int, required=True, help="number of decode devices"
)
args = parser.parse_args()
prefill_device_cnt = args.prefill_device_cnt
decode_device_cnt = args.decode_device_cnt

print("enter py")

hccn_tool_path = os.environ.get(
    "HCCN_TOOL_PATH", "/usr/local/Ascend/driver/tools/hccn_tool"
)
master_addr = os.environ.get("MASTER_ADDR")
master_port = os.environ.get("MASTER_PORT")
rank = os.environ.get("RANK")
local_rank = os.environ.get("LOCAL_RANK")
# This variable is set by torchrun, 
# and is different from WORLD_SIZE in gen_rank_table.sh.
world_size = os.environ.get("WORLD_SIZE")
soc_info = NPUSocInfo()

def get_cmd_stdout(cmd):
    import subprocess
    return subprocess.run(
        cmd,
        capture_output=True,
        shell=True
    ).stdout.decode("utf-8").strip()

local_host = get_cmd_stdout("hostname -I | awk -F \" \" \'{print$1}\'")
print(f"local_host: {local_host}")
print("gen ranktable.json")

num_cards = get_cmd_stdout("npu-smi info -l | grep \"Total Count\"").split(":")[1].strip()
num_cards = int(num_cards)
chips_per_card = get_cmd_stdout("npu-smi info -l | grep \"Chip Count\"").split("\n")[0].split(":")[1].strip()
chips_per_card = int(chips_per_card)

# generate local device list for local rank 0, and gather it to all ranks
local_device_list: list[dict[str, str]] = list()
if local_rank == "0":
    super_pod_id = "0"
    for card_id in range(num_cards):
        for chip_id in range(chips_per_card):
            device_id = card_id * chips_per_card + chip_id
            if soc_info.is_a3:
                device_ip = get_cmd_stdout(f"{hccn_tool_path} -i {device_id} -vnic -g | grep ipaddr").split(":")[1].strip()
                super_device_id = get_cmd_stdout(f"npu-smi info -t spod-info -i {card_id} -c {chip_id} | grep SDID").split(":")[1].strip()
                super_pod_id = get_cmd_stdout(f"npu-smi info -t spod-info -i {card_id} -c {chip_id} | grep \"Super Pod ID\"").split(":")[1].strip()
            else:
                device_ip = get_cmd_stdout(f"{hccn_tool_path} -i {device_id} -ip -g | grep ipaddr").split(":")[1].strip()

            device_info = {
                    "server_id": local_host,
                    "device_id": str(device_id),
                    "device_ip": str(device_ip),
                }
            if soc_info.is_a3:
                device_info.update({"super_pod_id": str(super_pod_id), "super_device_id": str(super_device_id)})
            local_device_list.append(device_info)

dist.init_process_group(backend=dist.Backend.GLOO)
global_device_list = [None] * dist.get_world_size()
dist.all_gather_object(global_device_list, local_device_list)
global_device_list = [device_info for device_list in global_device_list for device_info in device_list]
cnt = 1
for device_info in global_device_list:
    device_info["cluster_id"] = str(cnt)
    cnt += 1
assert (prefill_device_cnt + decode_device_cnt) <= len(global_device_list), \
"prefill_device_cnt + decode_device_cnt must be less than or equal to number of all devices in cluster"
ranktable = {
    "version": "1.2",
    "server_count": str(world_size),
    "prefill_device_list": global_device_list[:prefill_device_cnt],
    "decode_device_list": global_device_list[prefill_device_cnt:prefill_device_cnt+decode_device_cnt],
    "status": "completed"
}


if local_rank == '0':
    with open("ranktable.json", "w") as f:
        json.dump(ranktable, f, indent=4)

    print("gen ranktable.json done")