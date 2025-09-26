#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import os
import yaml
import argparse
import subprocess
import json
import shlex
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from typing import Dict, Any, List, Tuple, Optional
from omni_cli.config_transform import transform_deployment_config
from omni_cli.config_transform import detect_file_encoding
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import tempfile
import shlex
import omni_cli.proxy
from omni_cli.mk_inventory_yml import add_node, rm_node
from omni_cli.omni_cfg import parse_node_name
from omni_cli.omni_cfg import parse_remaining_args
from omni_cli.omni_cfg import cfg_set_process
from omni_cli.omni_cfg import cfg_delete_process
from omni_cli.omni_inspect import print_node_config

INFO    = "\033[92m[INFO]\033[0m"      # green
WARNING = "\033[93m[WARNING]\033[0m"   # yellow
ERROR   = "\033[91m[ERROR]\033[0m"     # red

@dataclass
class ClusterInfo:
    inventory: Dict[str, Any]
    allhosts: List[Tuple[str, Dict[str, Any]]] = field(init=False)  # [(host, hostvars), ...]

    p_pod_info: Dict[str, Dict[str, Any]] = field(init=False)
    d_pod_info: Dict[str, Dict[str, Any]] = field(init=False)

    def __post_init__(self):
        self.allhosts = _walk_hosts(self.inventory.get("all", self.inventory))
        self._create_pod_info()
        self._update_pod_info()

    def _new_pod_info(self) -> Dict[str, Any]:
        """Create a fresh pod-info dict."""
        return {
            "pod_hosts": [],
            "device_count": {},
            "server_offset": {},
            "master_ip": None,
            "master_port": None,  # all nodes in the pod use the same master port
            "num_dp": None,
            "tp": None,
            "num_servers": None,
            "kv_rank": None,
        }

    def _create_pod_info(self):
        """Get the P/D POD dictionary."""
        self.p_pod_info, self.d_pod_info = {}, {}

        for host, hv in self.allhosts:
            role = hv.get('env', {}).get('ROLE', None)
            master_host = hv.get("master_node", None)
            device_count = hv.get('ascend_rt_visible_devices','').count(',') + 1
            if master_host:
                if role == "prefill":
                    pod_vars = self.p_pod_info.setdefault(master_host, self._new_pod_info())
                elif role == "decode":
                    pod_vars = self.d_pod_info.setdefault(master_host, self._new_pod_info())
                else:
                    continue
                if host == master_host:  # master node
                    pod_vars["master_port"] = hv.get('env', {}).get('MASTER_PORT', None)
                    pod_vars["master_ip"] = hv.get('ansible_host', None)
                pod_vars["pod_hosts"].append(host)
                pod_vars["device_count"][host] = device_count

    def _update_pod_info(self):
        """Calculate the KV rank, server offset, num dp, etc. for P/D POD."""
        rank_count = 0
        dp = 1
        for master_host, pod_vars in self.p_pod_info.items():
            pod_vars["kv_rank"] = rank_count
            rank_count += 1
            for i, host in enumerate(pod_vars["pod_hosts"]):
                pod_vars["server_offset"][host] = 0   # for P always 0
            pod_vars["num_dp"] = dp  # for P always 1
            pod_vars["num_servers"] = 1  # for P always 1
            pod_vars["tp"] = sum(pod_vars["device_count"].values()) // dp

        for master_host, pod_vars in self.d_pod_info.items():
            tp = 1
            pod_vars["kv_rank"] = rank_count
            rank_count += 1
            offset = 0
            for host in pod_vars["pod_hosts"]:
                pod_vars["server_offset"][host] = offset
                offset += pod_vars["device_count"][host]
            pod_vars["num_dp"] = sum(pod_vars["device_count"].values()) // tp
            pod_vars["num_servers"] = pod_vars["device_count"][host]
            pod_vars["tp"] = tp

    @property
    def prefill_pod_num(self):
        return len(self.p_pod_info)

    @property
    def decode_pod_num(self):
        return len(self.d_pod_info)

    def __repr__(self):
        return f"<PodInfo(p_pod_info={self.p_pod_info}, d_pod_info={self.d_pod_info})>"


def execute_command(command):
    """Execute the ansible command"""
    process = subprocess.Popen(
        command,
        shell=True
    )

    return_code = process.wait()
    if return_code != 0:
        print(f"{ERROR} Deployment failed with return code {return_code}")

    return return_code

def _walk_hosts(node: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Flatten an Ansible-style inventory dict into [(host, hostvars)]."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    if not isinstance(node, dict):
        return out
    for h, vars_ in (node.get("hosts") or {}).items():
        out.append((h, vars_ or {}))
    for _g, child in (node.get("children") or {}).items():
        out.extend(_walk_hosts(child))
    return out

def _double_quotes(s: str) -> str:
    """Wrap value in double quotes for a safe shell arg."""
    s = str(s)
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
    return f'"{s}"'

def _build_export_block(env: Dict[str, Any]) -> str:
    """Build a sequence of export lines."""
    items_plain = [(k, v) for k, v in env.items() if "$" not in str(v)]
    items_refs  = [(k, v) for k, v in env.items() if "$" in str(v)]

    lines = []
    for k, v in items_plain + items_refs:
        if v is None:
            v = ""
        lines.append(f'export {k}={_double_quotes(v)}')
    return "\n".join(lines)

def _build_json_args(cfg: Dict[str, Any]) -> str:
    """Turn a dict into a JSON-like string"""
    parts: List[str] = []
    for k, v in cfg.items():
        key = json.dumps(str(k))  # always quoted JSON key
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("$"):              # let shell expand
                parts.append(f"{key}:{s}")
            else:                              # normal JSON string
                parts.append(f"{key}:{json.dumps(s)}")
        elif isinstance(v, bool):
            parts.append(f"{key}:{json.dumps(v)}")
        elif isinstance(v, (int, float)):
            parts.append(f"{key}:{v}")
        else:
            parts.append(f"{key}:{json.dumps(v)}")
    return "{" + ", ".join(parts) + "}"

def _build_string_args(extra_args: dict) -> str:
    """Convert extra-args dict to CLI string"""
    parts = []
    for k, v in extra_args.items():
        if v == "":
            parts.append(f"--{k}")
        else:
            parts.append(f"--{k} {v}")
    return " ".join(parts)

def _build_args_line(args: Dict[str, Any]) -> str:
    """
    Build a flat CLI argument string like:
      --tp "16" --kv-transfer-config "{\"kv_connector\":\"...\",\"kv_rank\":$KV_RANK}"
    Rules:
      - If a value is None -> boolean-like flag (emit `--flag` only)
      - Else -> `--flag "value"` where value is safely double-quoted
      - Assumes caller passes kv-transfer-config as a JSON string already
    """
    parts = []
    for k, v in (args or {}).items():
        flag = f"--{k}"
        if v is None:
            continue
        elif v == "":
            parts.append(flag)
        elif k == "kv-transfer-config" and isinstance(v, dict):
            inline = _build_json_args(v)
            parts.append(f"{flag} {_double_quotes(inline)}")
        elif k == "extra-args" and isinstance(v, dict):
            inline = _build_string_args(v)
            parts.append(f"{flag} {_double_quotes(inline)}")
        elif k == "additional-config" and isinstance(v, dict):
            inline = _build_json_args(v)
            parts.append(f"{flag} {_double_quotes(inline)}")
        else:
            parts.append(f"{flag} {_double_quotes(v)}")
    return " ".join(parts)

def _verify_and_fix_env_vars(
    cluster_info: ClusterInfo,
    inventory: Dict[str, Any],
    inventory_path: str = None,
) -> List[str]:
    """
    Detect port conflicts per machine (same IP). If conflicts found, bump by `offset` repeatedly until unique.
    """
    print(f"{INFO} verifying and fixing environment variables...")
    port_vars = ["API_PORT", "MASTER_PORT", "VLLM_LLMDATADIST_ZMQ_PORT"]
    offset: int = 16
    all_hosts = _walk_hosts(inventory.get("all", inventory))  # {host: {vars}}

    # group by IP
    ip_to_hosts = {}
    for host, hv in all_hosts:
        env = hv.get("env", {}) or {}
        ip = hv.get("ansible_host", {})
        ip_to_hosts.setdefault(ip, []).append((host, env))

    # per IP, ensure uniqueness for each monitored port var
    for ip, items in ip_to_hosts.items():
        for pv in port_vars:
            used_ports: Dict[int, str] = {}  # port -> host that currently holds it

            conflicts = []
            # identify conflicts
            for host, env in items:
                port = int(env.get(pv)) if pv in env and str(env.get(pv)).isdigit() else None
                if port is None:
                    if env.get("ROLE", None) in ["prefill", "decode"]:
                        print(f"{WARNING} host={host} has no {pv}; skipped.")
                    elif env.get("ROLE", None) in ["proxy"] and pv == "API_PORT":
                        print(f"{WARNING} host={host} has no {pv} (proxy port); please fix manually.")
                    continue
                if port in used_ports:
                    conflicts.append((host, env, port))
                else:
                    used_ports[port] = host

            # resolve conflicts by bumping with `offset` until free
            for host, env, original_port in conflicts:
                new_port = original_port
                while True:
                    new_port += offset
                    if new_port not in used_ports:
                        env[pv] = str(new_port)
                        used_ports[new_port] = host
                        print(f"{WARNING} ip={ip} {pv} conflict: host={host} {original_port} -> {new_port}")
                        break
    need_overwrite_inv = len(conflicts) > 0

    # calculate pod num, server ip list and server offset, kv rank
    server_ip_list_temp = []
    for host, hv in inventory['all']['children']['D']['hosts'].items():
        ip = hv.get('ansible_host', None)
        if ip:
            server_ip_list_temp.append(f"{ip}")

    # update num_dp_dict
    server_ip_list = ','.join(server_ip_list_temp)

    ## update inventory
    need_overwrite_inv = False
    for host, hv in all_hosts:
        master_host = hv.get("master_node", None)
        role = hv.get("env", {}).get("ROLE", None)
        if role not in ["prefill", "decode"]:
            continue
        pod_info = cluster_info.p_pod_info if role == "prefill" else cluster_info.d_pod_info
        pod_info = pod_info.get(master_host, None)
        if pod_info is None:
            print(f"{WARNING} host={host} can not find POD_INFO")
            continue

        if master_host is None:
            print(f"{ERROR} host={host} can not find master node")
            raise RuntimeError(f"host={host} can not find master node")

        if "PREFILL_POD_NUM" in hv.get("env", {}):
            if hv.get("env", {}).get("PREFILL_POD_NUM") != cluster_info.prefill_pod_num:
                need_overwrite_inv = True
                hv.get("env", {})["PREFILL_POD_NUM"] = cluster_info.prefill_pod_num
                print(f"{INFO} host={host} PREFILL_POD_NUM set to {cluster_info.prefill_pod_num}")
        if "DECODE_POD_NUM" in hv.get("env", {}):
            if hv.get("env", {}).get("DECODE_POD_NUM") != cluster_info.decode_pod_num:
                need_overwrite_inv = True
                hv.get("env", {})["DECODE_POD_NUM"] = cluster_info.decode_pod_num
                print(f"{INFO} host={host} DECODE_POD_NUM set to {cluster_info.decode_pod_num}")
        if  "RANK_TABLE_FILE_PATH" in hv.get("env", {}):
            need_overwrite_inv = True
            rank_table_save_path = hv.get("env", {})["RANKTABLE_SAVE_PATH"]
            if role == 'prefill':
                if len(pod_info["pod_hosts"]) > 1:
                    hv.get("env", {})["RANK_TABLE_FILE_PATH"] = f"$(ls {rank_table_save_path}/global/collect_files_p/{pod_info['master_ip']}/local_*merge.json)"
                else:
                    prefill_server_list = hv.get("ascend_rt_visible_devices", "").replace(',', '')
                    hv.get("env", {})["RANK_TABLE_FILE_PATH"] = f"$(ls {rank_table_save_path}/prefill_config/local_*{prefill_server_list}.json)"
            if role == 'decode':
                if len(pod_info["pod_hosts"]) > 1:
                    hv.get("env", {})["RANK_TABLE_FILE_PATH"] = f"$(ls {rank_table_save_path}/global/collect_files_d/local_*merge.json)"
                else:
                    hv.get("env", {})["RANK_TABLE_FILE_PATH"] = f"$(ls {rank_table_save_path}/decode_config/local_*.json)"
            print(f"{INFO} host={host} RANK_TABLE_FILE_PATH set to {hv.get('env', {})['RANK_TABLE_FILE_PATH']}")
        if "SERVER_IP_LIST" in hv.get("env", {}):
            if hv.get("env", {}).get("SERVER_IP_LIST") != server_ip_list:
                need_overwrite_inv = True
                hv.get("env", {})["SERVER_IP_LIST"] = server_ip_list
                print(f"{INFO} host={host} SERVER_IP_LIST set to {server_ip_list}")
        if "SERVER_OFFSET" in hv.get("env", {}):
            server_offset = pod_info.get("server_offset", {}).get(host, None)
            if server_offset is not None and hv.get("env", {}).get("SERVER_OFFSET") != server_offset:
                hv.get("env", {})["SERVER_OFFSET"] = server_offset
                need_overwrite_inv = True
                print(f"{INFO} host={host} SERVER_OFFSET set to {server_offset}")
        if "KV_RANK" in hv.get("env", {}):
            kv_rank = pod_info.get("kv_rank", None)
            if kv_rank is not None and hv.get("env", {}).get("KV_RANK") != kv_rank:
                hv.get("env", {})["KV_RANK"] = kv_rank
                need_overwrite_inv = True
                print(f"{INFO} host={host} KV_RANK set to {kv_rank}")
        if "HOST_IP" in hv.get("env", {}):
            host_ip = pod_info.get("master_ip", None)
            if host_ip is not None and hv.get("env", {}).get("HOST_IP") != host_ip:
                hv.get("env", {})["HOST_IP"] = host_ip
                need_overwrite_inv = True
                print(f"{INFO} host={host} HOST_IP set to {host_ip}")
        # set master port same as host_ip's master port
        if "MASTER_PORT" in hv.get("env", {}):
            if role == "prefill" or role == "decode":
                master_port = pod_info.get("master_port", None)
                if master_port is None:
                    print(f"{WARNING} host={host} with master node={master_host} can not find MASTER_PORT")
                if master_port is not None and hv.get("env", {}).get("MASTER_PORT") != master_port:
                    hv.get("env", {})["MASTER_PORT"] = master_port
                    need_overwrite_inv = True
                    print(f"{INFO} host={host} MASTER_PORT set to {master_port}")
        if "num-servers" in hv.get("args", {}):
            num_server = pod_info.get("num_servers", None)
            if num_server is not None and hv.get("args", {}).get("num-servers") != num_server:
                hv.get("args", {})["num-servers"] = num_server
                need_overwrite_inv = True
                print(f"{INFO} host={host} num-servers set to {num_server}")
        if "num-dp" in hv.get("args", {}):
            num_dp = pod_info.get("num_dp", None)
            if num_dp is not None and hv.get("args", {}).get("num-dp") != num_dp:
                hv.get("args", {})["num-dp"] = num_dp
                need_overwrite_inv = True
                print(f"{INFO} host={host} num-dp set to {num_dp}")
        if "tp" in hv.get("args", {}):
            tp = pod_info.get("tp", None)
            if tp is not None and hv.get("args", {}).get("tp") != tp:
                hv.get("args", {})["tp"] = tp
                need_overwrite_inv = True
                print(f"{INFO} host={host} tp set to {tp}")
        if "PREFILL_TENSOR_PARALLEL_SIZE" in hv.get("env", {}) and role == "prefill":
            tp = pod_info.get("tp", None)
            if tp is not None and hv.get("env", {}).get("PREFILL_TENSOR_PARALLEL_SIZE") != tp:
                hv.get("env", {})["PREFILL_TENSOR_PARALLEL_SIZE"] = tp
                need_overwrite_inv = True
                print(f"{INFO} host={host} PREFILL_TENSOR_PARALLEL_SIZE set to {tp}")
        if "MODEL_LEN_MAX_PREFILL" in hv.get("env", {}) and role == "prefill":
            model_len_max = hv.get("env", {}).get("MODEL_LEN_MAX_PREFILL")
            tp = pod_info.get("tp", None)
            if model_len_max % tp != 0:
                print(f"{WARNING} host={host} MODEL_LEN_MAX_PREFILL is not a multiple of TP size!")

    if need_overwrite_inv:
        with open(inventory_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(inventory, f, default_flow_style=False, sort_keys=False)
        print(f"{INFO} inventory written back to {inventory_path}")
    else:
        print(f"{INFO} inventory at {inventory_path} has passed verification")



def omni_ranktable(inventory):
    cur_dir = os.path.dirname(__file__)
    cmd = "ansible-playbook -i " + str(inventory) + " " + str(cur_dir) + "/configs/generate_ranktable.yml"
    os.system(cmd)

def maybe_start_ray(is_master, pod_info, role, log_path):

    if role == "decode":
        return False, ""

    ## if >1 servers for Prefill, ray is needed
    num_server_in_pod = len(pod_info.get("pod_hosts", []))
    num_servers = pod_info.get("num_servers", 1)
    master_ip = pod_info.get("master_ip", None)
    if num_server_in_pod > 1 and role == "prefill":
        if is_master:
            ray_cmd = f"""
ray stop --force
ray start --head --num-gpus={num_servers} >> {log_path}/omni_cli.log 2>&1
sleep 10s
"""
        else:
             ray_cmd = f"""
sleep 5s
command="ray start --address='{master_ip}:6379' --num-gpus={num_servers} &> /dev/null"
echo $command >> {log_path}/omni_cli.log
cost_time=0
end_time=300
while true; do
    if [ $cost_time -ge $end_time ]; then
    echo "error, conneciton timeout" >> {log_path}/omni_cli.log
    exit 1
    fi

    eval $command
    if [ $? -eq 0 ]; then
    echo "succeed to connect to ray head node" >> {log_path}/omni_cli.log
    break
    else
    echo "failed to connect to ray head node, wait 5s....." >> {log_path}/omni_cli.log
    sleep 5
    cost_time=$((cost_time + 5))
    fi
done
"""
        return True, ray_cmd
    else:
        return False, ""

def omni_cli_start(
    inventory_path: str = "./server_profiles.yml",
    host_pattern: Optional[str] = None,   # e.g., "127.0.0.1"
    role_filter: Optional[str] = None,    # e.g., "P" or "D"
    python_bin: str = "python",
    entry_py: str = "start_api_servers.py",
    skip_verify_config: bool = False,
    dev: bool = False,
    proxy_only: bool = False
) -> None:
    """
    Read inventory YAML, generate a per-host bash script, and run it via:
      ansible <host> -i <inventory> -m script -a <script_path>
    """
    if not inventory_path:
        print(f"{ERROR} Inventory path is required.")
        return
    else:
        print(f"{INFO} Use inventory at:", inventory_path)
    if not dev:
        omni_ranktable(inventory_path)

    if proxy_only:
        omni_cli.proxy.omni_run_proxy(inventory_path)
        return

    inv_file = Path(inventory_path).expanduser().resolve()
    with open(inventory_path, "r", encoding="utf-8") as f:
        inv = yaml.safe_load(f)

    cluster_info = ClusterInfo(inv)
    if not skip_verify_config:
        _verify_and_fix_env_vars(cluster_info, inv, inv_file)

    if not dev:
        omni_cli.proxy.omni_run_proxy(inventory_path)

    all_hosts = _walk_hosts(inv.get("all", inv))
    if not role_filter:
        role_filter = ["prefill", "decode"]
    selected: List[Tuple[str, Dict[str, Any]]] = []
    for host, hv in all_hosts:
        if host_pattern and host != host_pattern:
            continue
        if role_filter and hv.get("env").get("ROLE") not in role_filter:
            continue
        selected.append((host, hv))

    if not selected:
        raise RuntimeError("No matching hosts found with given filters.")

    for host, hv in selected:
        env: Dict[str, Any] = hv.get("env", {}) or {}
        args: Dict[str, Any] = hv.get("args", {}) or {}
        container_name: str = hv.get("container_name")
        role = env.get("ROLE", None)
        master_node = hv.get("master_node", None)
        pod_info = cluster_info.p_pod_info if role == "prefill" else cluster_info.d_pod_info
        pod_info = pod_info.get(master_node, None)
        is_master = host == master_node

        code_path = str(env.get("CODE_PATH") or "").strip()
        log_path = str(env.get("LOG_PATH") or "").strip()
        log_path = f"{log_path}/{host.replace('.', '_')}"
        env["LOG_PATH"] = log_path
        # check if start ray
        need_start_ray, ray_cmd = maybe_start_ray(is_master, pod_info, role, log_path)

        if need_start_ray:
            args.get("extra-args", {})["distributed-executor-backend"] = "ray"
            env["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = 1
            env["RAY_CGRAPH_get_timeout"] = 7200
            if is_master:
                env["RAY_USAGE_STATS_ENABLED"] = 0

        export_block = _build_export_block(env)
        args_line = _build_args_line(args)


        start_server_cmd = f"""
# Exec the command
cd {_double_quotes(code_path)}/tools/scripts
echo "cd {_double_quotes(code_path)}/tools/scripts" >> {log_path}/omni_cli.log
{python_bin} {entry_py} {args_line} >> {log_path}/omni_cli.log 2>&1 &
echo "{python_bin} {entry_py} {args_line} >> {log_path}/omni_cli.log 2>&1 &" >> {log_path}/omni_cli.log
"""

        with tempfile.NamedTemporaryFile(
            "w", delete=False,
            dir="./",
            prefix=f"omni_start_{host.replace('.', '_')}_",
            suffix=".sh"
        ) as tf:
            script_path = Path(tf.name)
            tf.write("#!/usr/bin/env bash\n")
            tf.write("set -euo pipefail\n\n")

            tf.write(f"docker exec -i {shlex.quote(container_name)} bash -s <<'EOF'\n")
            tf.write("source ~/.bashrc\n\n")

            tf.write(f"if [ ! -d {shlex.quote(log_path)} ]; then\n")
            tf.write(f"  mkdir -p {shlex.quote(log_path)}\n")
            tf.write("fi\n")

            tf.write("# Export environment variables\n")
            tf.write(export_block + "\n\n")
            tf.write(f'echo "{export_block}\n" > {log_path}/omni_cli.log\n\n')

            if need_start_ray:
                tf.write(f"{ray_cmd}\n")
                if is_master:
                    tf.write(start_server_cmd)
            else:
                tf.write(start_server_cmd)
            tf.write("EOF\n")

        os.chmod(script_path, 0o755)

        cmd = (
            f"ansible {shlex.quote(host)} "
            f"-i {shlex.quote(str(inv_file))} "
            f"-m script "
            f"-a {shlex.quote(str(script_path))}"
        )

        try:
            return_code = execute_command(cmd) # 0 for success
            if return_code == 0:
                for host, vars in inv['all']['children']['C']['hosts'].items():
                    print("\nServer is available at: http://" + vars.get('ansible_host', '') \
                        + ":" + str(vars.get('env').get('API_PORT', '')))
                    print("\n\n")
            else:
                print(f"{ERROR} ansible command failed with return code {return_code}")
        finally:
            try:
                script_path.unlink(missing_ok=True)
            except Exception:
                pass

def omni_cli_stop(
    inventory_path: str = "./server_profiles.yml",
    host_pattern: Optional[str] = None,   # e.g., "127.0.0.1"
    role_filter: Optional[str] = None,    # e.g., "P" or "D"
) -> None:
    """kill python and vllm processes in the containers
    """
    inv_file = Path(inventory_path).expanduser().resolve()
    with open(inv_file, "r", encoding="utf-8") as f:
        inv = yaml.safe_load(f)

    all_hosts = _walk_hosts(inv.get("all", inv))

    if not role_filter:
        role_filter = ["prefill", "decode"]
    selected: List[Tuple[str, Dict[str, Any]]] = []
    for host, hv in all_hosts:
        if host_pattern and host != host_pattern:
            continue
        if role_filter and hv.get("env").get("ROLE") not in role_filter:
            continue
        selected.append((host, hv))

    if not selected:
        raise RuntimeError("No matching hosts found with given filters.")

    for host, hv in selected:
        env: Dict[str, Any] = hv.get("env", {}) or {}
        container_name: str = hv.get("container_name")

        with tempfile.NamedTemporaryFile(
            "w", delete=False,
            dir="./",
            prefix=f"omni_stop_{host.replace('.', '_')}_",
            suffix=".sh"
        ) as tf:
            script_path = Path(tf.name)

            tf.write("#!/usr/bin/env bash\n")
            tf.write(f"docker exec -i {shlex.quote(container_name)} bash -s <<'EOF'\n")
            tf.write("pkill -9 python || true\n")
            tf.write("pkill -9 vllm   || true\n")
            tf.write("pkill -9 -f ray || true\n")
            tf.write("EOF\n")
        os.chmod(script_path, 0o755)

        cmd = (
            f"ansible {shlex.quote(host)} "
            f"-i {shlex.quote(str(inv_file))} "
            f"-m script "
            f"-a {shlex.quote(str(script_path))}"
        )

        try:
            execute_command(cmd)
        finally:
            try:
                script_path.unlink(missing_ok=True)
                pass
            except Exception:
                pass

def get_host_groups(inv_data: dict) -> Dict[str, List[str]]:
    """Build host-to-group mappings"""
    host_groups = {}

    # Traverse the entire inventory structure
    def traverse(node, current_groups=None):
        if current_groups is None:
            current_groups = []

        # Dealing with the host of the current node
        hosts = node.get("hosts", {})
        for host, host_vars in hosts.items():
            # Merge current groups and host custom groups
            merged_groups = current_groups.copy()
            if "groups" in host_vars:
                merged_groups.extend(host_vars["groups"])
            host_groups[host] = list(set(merged_groups))

        # Handle child nodes
        children = node.get("children", {})
        for child_name, child_data in children.items():
            # Child nodes inherit the parent node's group and add their own group name
            child_groups = current_groups + [child_name]
            traverse(child_data, child_groups)

    # Traversing from the root node
    traverse(inv_data.get("all", inv_data))
    return host_groups

def sync_code(
    inventory_path,
    dry_run: bool = False,
    code_path: str = None
) -> None:
    """Sync code to all relevant hosts and containers with minimal output"""
    if not code_path:
        print(f"{ERROR} code_path is required")
        return

    # Read inventory file
    with open(inventory_path, 'r') as f:
        inventory_data = yaml.safe_load(f)

    # Get host-group mapping
    host_groups = get_host_groups(inventory_data)

    # Get all hosts and their variables using _walk_hosts
    all_hosts = dict(_walk_hosts(inventory_data.get("all", inventory_data)))

    # Identify all hosts that need processing
    p_hosts = set()
    d_hosts = set()
    c_hosts = set()
    p_d_ips = set()
    for host, groups in host_groups.items():
        host_vars = all_hosts.get(host, {})
        host_ip = host_vars.get("ansible_host", host)

        if 'P' in groups:
            p_hosts.add(host)
            p_d_ips.add(host_ip)
        if 'D' in groups:
            d_hosts.add(host)
            p_d_ips.add(host_ip)
        if 'C' in groups:
            c_hosts.add(host)

    # Find C nodes that need processing (not in P or D)
    c_hosts_to_process = set()
    for host in c_hosts:
        host_vars = all_hosts.get(host, {})
        host_ip = host_vars.get("ansible_host", host)

        if host_ip not in p_d_ips:
            c_hosts_to_process.add(host)
        else:
            print("{INFO} Node C is skipped because it has the same IP address as a P or D node")

    # All hosts that need processing
    all_target_hosts = p_hosts | d_hosts | c_hosts_to_process

    # Create temporary script file
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as tf:
        script_path = tf.name

        # Write script header
        tf.write("#!/bin/bash\n\n")
        tf.write(f"CODE_SRC=\"{code_path}\"\n\n")

        # Add progress indicator function
        tf.write("""
# Function to show progress spinner
show_spinner() {
    local pid=$!
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c] " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\\b\\b\\b\\b\\b"
    done
    printf "    \\b\\b\\b\\b"
}
""")

        # Create directories and sync code to all target hosts
        for host in all_target_hosts:
            host_vars = all_hosts.get(host, {})
            # Get actual connection address for the host
            host_addr = host_vars.get("ansible_host", host)

            # Get SSH key path (if exists)
            ssh_private_key = host_vars.get("ansible_ssh_private_key_file", "")
            ssh_user = host_vars.get("ansible_user", "")
            env = host_vars.get("env", {})

            # Build SSH command prefixes
            ssh_prefix = "ssh"
            rsync_prefix = "rsync -avz --quiet --delete"

            if ssh_private_key:
                ssh_prefix = f"ssh -i {ssh_private_key}"
                rsync_prefix = f"rsync -avz --quiet --delete -e 'ssh -i {ssh_private_key}'"

            if ssh_user:
                ssh_prefix = f"{ssh_prefix} -l {ssh_user}"
                rsync_prefix = f"{rsync_prefix} -e 'ssh -l {ssh_user}'"
                if ssh_private_key:
                    rsync_prefix = f"rsync -avz --quiet --delete -e 'ssh -i {ssh_private_key} -l {ssh_user}'"

            log_path = env.get("LOG_PATH")
            if log_path:
                tf.write(f"echo \"{INFO} Creating log directory \"{log_path}/{host}\" on {host}\"\n")
                tf.write(f"{ssh_prefix} {host_addr} \"mkdir -p {log_path}/{host}\" >/dev/null 2>&1\n\n")
            else:
                tf.write(f"echo \"{WARNING} LOG_PATH not defined for host {host}, skipping log directory creation\"\n\n")

            tf.write(f"echo \"{INFO} Creating code directory \'{code_path}\' on {host}\"\n")
            tf.write(f"{ssh_prefix} {host_addr} \"mkdir -p {code_path}\" >/dev/null 2>&1\n\n")

            tf.write(f"echo \"{INFO} Syncing code from executor from \'{code_path}/omniinfer/\' to \'{host}:{code_path}/omniinfer/\' \"\n")
            tf.write(f"echo -n \"{INFO} Progress: \"\n")
            tf.write(f"{rsync_prefix} {code_path}/omniinfer/ {host_addr}:{code_path}/omniinfer/ & show_spinner\n")
            tf.write(f"echo \"Done\"\n\n")

            # Handle docker cp for all hosts that need it
            container_name = host_vars.get("container_name", "")
            if container_name:
                tf.write(f"echo \"{INFO} Docker cp code to container on {host}, from {code_path}/omniinfer to {container_name}:/workspace/\"\n")
                tf.write(f"{ssh_prefix} {host_addr} \"docker cp {code_path}/omniinfer {container_name}:/workspace/\" >/dev/null 2>&1\n")
                tf.write(f"echo \"{INFO} Container copy completed\"\n\n")
            else:
                tf.write(f"echo \"{WARNING} Missing container_name for host {host}\"\n\n")

    # Set script execution permissions
    os.chmod(script_path, 0o755)

    # Execute script or display dry run information
    if dry_run:
        print("Dry run: would execute the following script:")
        with open(script_path, 'r') as f:
            print(f.read())
    else:
        print("Starting sync process...")
        try:
            # Execute the script and capture output
            process = subprocess.Popen(
                [script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Print output in real-time
            for line in process.stdout:
                print(line, end='')

            # Wait for process to complete
            process.wait()

            if process.returncode == 0:
                print("\nSync process completed successfully.")
            else:
                print(f"\nSync process failed with return code {process.returncode}")
                print("Error output:")
                for line in process.stderr:
                    print(line, end='')
        except Exception as e:
            print(f"Error during sync process: {e}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(script_path)
            except:
                pass

def install_code(
    inventory_path,
    dry_run: bool = False,
) -> None:
    """Install code inside container"""

    inv_file = Path(inventory_path).expanduser().resolve()

    # Load inventory
    with open(inv_file, "r", encoding="utf-8") as f:
        inv = yaml.safe_load(f)

    all_hosts = _walk_hosts(inv.get("all", inv))
    host_groups = get_host_groups(inv)

    if not all_hosts:
        print("Warning: No hosts found in inventory.")
        return

    # Update command templates
    docker_update_code_cmd = """
    /bin/bash -c '
    {
        . ~/.bashrc
        export http_proxy=http://10.155.96.5:8081
        export https_proxy=http://10.155.96.5:8081
        sed -i s#https://pypi.tuna.tsinghua.edu.cn/simple#https://mirrors.tools.huawei.com/pypi/simple#g /root/.config/pip/pip.conf
        sed -i s#pypi.tuna.tsinghua.cn#mirrors.tools.huawei.com#g /root/.config/pip/pip.conf
        pip install setuptools_scm
        cd /workspace/omniinfer/infer_engines
        git config --global --add safe.directory /workspace/omniinfer/infer_engines/vllm
        bash bash_install_code.sh
        pip uninstall vllm -y
        pip uninstall omni_infer -y
        cd vllm
        SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e . --no-deps --no-build-isolation
        cd ../../
        pip install -e . --no-deps --no-build-isolation
    } > ${LOG_PATH}/{{ inventory_hostname }}/pip.log 2>&1
    '
    """

    for host, hv in all_hosts:
        groups = host_groups.get(host, [])
        print(f"\nProcessing host: {host} (groups: {groups})")

        # Determine the role (P, D, or C)
        role = "C"  # Default
        if 'P' in groups:
            role = "P"
        elif 'D' in groups:
            role = "D"
        elif 'C' in groups:
            role = "C"

        # Only groups P and D are processed
        if role not in ['P', 'D']:
            print(f"Skipping host {host} (not in P or D group)")
            continue

        # Get environment variables and other host variables
        env = hv.get("env", {})
        log_path = env.get("LOG_PATH")
        container_name = hv.get("container_name")

        # Check the necessary variables
        if not log_path:
            print(f"{WARNING} LOG_PATH not defined for host {host}, skipping")
            continue
        if not container_name:
            print(f"{WARNING} container_name not defined for host {host}, skipping")
            continue

        # Create a temporary script file
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as tf:
            script_path = Path(tf.name)

            # Write scripted content
            tf.write("#!/bin/bash\n")
            tf.write("set -euo pipefail\n\n")

            # Export environment variables
            tf.write("# Export environment variables\n")
            for key, value in env.items():
                tf.write(f"export {key}={shlex.quote(str(value))}\n")

            # Add role-specific variables
            tf.write("\n# Role-specific variables\n")
            tf.write(f"export CONTAINER_NAME={shlex.quote(container_name)}\n")

            # Update code inside container - fix command formatting
            tf.write("\n# Update code inside container\n")
            # Replace the variable in the command
            update_cmd = docker_update_code_cmd.replace(
                "${LOG_PATH}/{{ inventory_hostname }}/pip.log",
                f"{log_path}/{host}/pip.log"
            )
            tf.write(f"echo \"Updating code inside container {container_name}\"\n")
            tf.write(f"docker exec {shlex.quote(container_name)} /bin/bash -c {shlex.quote(update_cmd)}\n")
            os.chmod(script_path, 0o755)

            # Print script content in dry run mode
            if dry_run:
                print(f"=== DRY RUN: Script for host {host} ===")
                with open(script_path, "r") as script_file:
                    print(script_file.read())
                print("===")

        if not dry_run:
            try:
                # Build ansible commands
                cmd = (
                    f"ansible {shlex.quote(host)} "
                    f"-i {shlex.quote(str(inv_file))} "
                    f"-m script "
                    f"-a {shlex.quote(str(script_path))}"
                )
                execute_command(cmd)
            finally:
                # Clean up temporary files
                try:
                    script_path.unlink(missing_ok=True)
                except Exception as e:
                    print(f"Warning: Failed to delete temp file: {e}")
        else:
            # In dry run mode, only the commands we will run are displayed
            print(f"DRY RUN: Would execute for host {host}:")
            print(f"  ansible {host} -i {inv_file} -m script -a {script_path}")

    print("\nInstall process completed.")

def get_default_deploy_path(current_cmd):
    """Get or create the default deployment path (file is created only on first call)"""
    # Create default file and save path
    current_dir = Path.cwd()
    deploy_path = current_dir / "server_profiles.yml"

    if not deploy_path.exists():
        # Create default inventory structure
        if current_cmd == 'add_node':
            default_inventory = {
                "all": {
                    "children": {
                        "P": {"hosts": {}},
                        "D": {"hosts": {}},
                        "C": {"hosts": {}}
                    }
                }
            }

            # Write to file
            with open(deploy_path, "w") as f:
                yaml.dump(default_inventory, f)

            print(f"{INFO} Created default inventory file at: {deploy_path}")
        else:
            raise FileNotFoundError("server_profiles.yml not found, please confirm the workspace or reinitialize using add_node")
    # Save as absolute path
    deploy_path = deploy_path.resolve()
    return deploy_path

def _walk_hosts(node: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Flatten an Ansible-style inventory dict into [(host, hostvars)]."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    if not isinstance(node, dict):
        return out
    for h, vars_ in (node.get("hosts") or {}).items():
        out.append((h, vars_ or {}))
    for _g, child in (node.get("children") or {}).items():
        out.extend(_walk_hosts(child))
    return out

def print_node_list(inventory_path: str) -> None:
    """
    Print the current node list including role, name, and IP address.

    Args:
        inventory_path: Path to the inventory YAML file
    """
    inv_file = Path(inventory_path).expanduser().resolve()
    with open(inv_file, "r", encoding="utf-8") as f:
        inv = yaml.safe_load(f)

    children = inv.get("all", {}).get("children", {})

    print(f"{'Role':<5} | {'Name':<5} | {'IP Address':<15}")
    print("-" * 30)

    for role, role_data in children.items():
        if role not in ["C", "D", "P"]:
            continue

        hosts = role_data.get("hosts", {})

        for host_name, host_data in hosts.items():
            ip_address = host_data.get("ansible_host", "N/A")

            print(f"{role:<5} | {host_name:<5} | {ip_address:<15}")

    print("-" * 30)

def run_docker_containers(
    inventory_path,
    dry_run: bool = False,
) -> None:
    inv_file = Path(inventory_path).expanduser().resolve()

    # Load inventory
    with open(inv_file, "r", encoding="utf-8") as f:
        inv = yaml.safe_load(f)

    all_hosts = _walk_hosts(inv.get("all", inv))
    host_groups = get_host_groups(inv)

    if not all_hosts:
        print(f"{WARNING} No hosts found in inventory.")
        return

    # Base Docker command template without LOG_PATH or MODEL_PATH
    base_docker_run_cmd = """docker run -it --shm-size=500g \\
        --net=host \\
        --privileged=true \\
        -u root \\
        -w /data \\
        --device=/dev/davinci_manager \\
        --device=/dev/hisi_hdc \\
        --device=/dev/devmm_svm \\
        --entrypoint=bash \\
        -v /tmp:/tmp \\
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \\
        -v /usr/local/dcmi:/usr/local/dcmi \\
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \\
        -v /etc/ascend_install.info:/etc/ascend_install.info \\
        -v /usr/local/sbin:/usr/local/sbin \\
        -v /etc/hccn.conf:/etc/hccn.conf \\
        -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \\
        -v /tmp/ranktable_save_path:/tmp/ranktable_save_path \\
        -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime"""

    for host, hv in all_hosts:
        print(f"\nProcessing host: {host}")

        # Determine role
        groups = host_groups.get(host, [])
        role = "C"
        if 'P' in groups:
            role = "P"
        elif 'D' in groups:
            role = "D"
        elif 'C' in groups:
            role = "C"

        # Get environment variables and host variables
        env = hv.get("env", {})
        log_path = env.get("LOG_PATH")
        model_path = env.get("MODEL_PATH")
        docker_image_id = hv.get("DOCKER_IMAGE_ID")
        container_name = hv.get("container_name", f"omni_container_{host}")

        # Check required variables
        if not log_path:
            print(f"{WARNING} LOG_PATH not defined for host {host}, skipping")
            continue
        if not docker_image_id:
            print(f"{WARNING} DOCKER_IMAGE_ID not defined for host {host}, skipping")
            continue

        # For P and D roles, MODEL_PATH is required
        if role in ['P', 'D'] and not model_path:
            print(f"{WARNING} MODEL_PATH not defined for host {host} (role {role}), skipping")
            continue

        # Create temporary script file
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as tf:
            script_path = Path(tf.name)

            # Write script content
            tf.write("#!/bin/bash\n")
            tf.write("set -euo pipefail\n\n")

            # Export environment variables
            tf.write("# Export environment variables\n")
            for key, value in env.items():
                tf.write(f"export {key}={shlex.quote(str(value))}\n")

            # Clean up existing container
            tf.write("\n# Cleanup existing container\n")
            tf.write(f"if docker inspect --format='{{{{.Name}}}}' {shlex.quote(container_name)} &>/dev/null; then\n")
            tf.write(f"    docker stop {shlex.quote(container_name)}\n")
            tf.write(f"    docker rm -f {shlex.quote(container_name)}\n")
            tf.write("fi\n")

            # Build Docker run command with actual paths
            tf.write("\n# Build Docker run command\n")
            tf.write(f"docker_cmd={shlex.quote(base_docker_run_cmd)}\n")

            # Add LOG_PATH mount with actual path
            tf.write(f"docker_cmd+=\" -v {shlex.quote(log_path)}:{shlex.quote(log_path)}\"\n")

            # Add MODEL_PATH mount for P and D roles
            if role in ['P', 'D'] and model_path:
                tf.write(f"docker_cmd+=\" -v {shlex.quote(model_path)}:{shlex.quote(model_path)}\"\n")

            tf.write(f"docker_cmd+=\" -d --name {shlex.quote(container_name)} {shlex.quote(docker_image_id)}\"\n")

            # Execute command
            tf.write("\n# Execute Docker run command\n")
            tf.write("echo \"Starting container with command: $docker_cmd\"\n")
            tf.write("eval \"$docker_cmd\"\n")

            # Print container status
            tf.write("\n# Verify container status\n")
            tf.write(f"docker ps -a --filter \"name={shlex.quote(container_name)}\"\n")

            # Set execution permissions
            os.chmod(script_path, 0o755)

            # Print script content in dry-run mode
            if dry_run:
                print(f"=== DRY RUN: Script for host {host} ===")
                with open(script_path, "r") as script_file:
                    print(script_file.read())
                print("===")

        if not dry_run:
            try:
                # Build ansible command
                cmd = (
                    f"ansible {shlex.quote(host)} "
                    f"-i {shlex.quote(str(inv_file))} "
                    f"-m script "
                    f"-a {shlex.quote(str(script_path))}"
                )

                print(f"Executing script on host {host}:")
                execute_command(cmd)
            finally:
                # Clean up temporary file
                try:
                    script_path.unlink(missing_ok=True)
                except Exception as e:
                    print(f"{WARNING} Failed to delete temp file: {e}")
        else:
            # In dry-run mode, just show what command would be executed
            print(f"DRY RUN: Would execute for host {host}:")
            print(f"  ansible {host} -i {inv_file} -m script -a {script_path}")

    print("\nAll hosts processed.")

def upgrade_packages():
    """Install the latest wheel package"""
    print("Under development")

def collect_logs():
    """Fetch logs"""
    print("Under development")

def main():
    # Create main argument parser with description
    parser = argparse.ArgumentParser(description="Omni Inference Service Management")
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_deploy_path = ''

    # START command configuration
    start_parser = subparsers.add_parser("start", help="Start the omni services")
    start_parser.add_argument(
        "--config_path",
        nargs='?',
        default=None,
        help='Start in normal mode with config file'
    )
    start_parser.add_argument("--skip-verify-config", action="store_true", help="Skip verification of config")
    start_parser.add_argument("--proxy-only", action="store_true", help="Start the proxy only")
    start_group = start_parser.add_mutually_exclusive_group()
    start_group.add_argument(
        "--normal",
        nargs=1,
        metavar='config_path',
        help="Start in normal mode (default) with config file"
    )
    start_group.add_argument("--run_dev", action="store_true", help="Start in developer mode: Start the service, without ranktable and proxy")

    # STOP command configuration
    subparsers.add_parser("stop", help="Stop the omni service")

    # CFG command configuration
    cfg_parser = subparsers.add_parser("cfg", help="Modify configuration")
    cfg_group = cfg_parser.add_mutually_exclusive_group(required=True)
    cfg_group.add_argument("--set", action='store_true', help="Set configuration key-value pairs (e.g., --key value)")
    cfg_parser.add_argument('name', nargs=1, help='Node name (e.g., prefill_0)')
    cfg_parser.add_argument('remaining_args', nargs=argparse.REMAINDER, help='Additional optional parameters')
    cfg_group.add_argument("--delete", action='store_true', help="Delete configuration keys (e.g., --key)")

    # INSPECT command configuration
    inspect_parser = subparsers.add_parser("inspect", help="Inspect Configuration")
    inspect_parser.add_argument('name', nargs=1, help='Node name (e.g., prefill_0)')

    # UPGRADE command configuration
    subparsers.add_parser("upgrade", help="Upgrade packages")

    # FETCH_LOG command configuration
    subparsers.add_parser("collect_log", help="Collect logs")

    # LS command configuration
    ls_parser = subparsers.add_parser("ls", help="Print the node list")
    ls_parser.add_argument(
        "--deploy_path",
        default=str(default_deploy_path),
        help=f"Path to server_profiles.yml (default: {default_deploy_path})"
    )
    ls_parser.set_defaults(func=lambda args:print_node_list(
        inventory_path=args.deploy_path
    ))
    # ADD_NODE command configuration
    addnode_parser = subparsers.add_parser("add_node", help="Add a node to server_profiles.yml")
    addnode_parser.add_argument(
        "--deploy_path",
        default=str(default_deploy_path),
        help=f"Path to server_profiles.yml (default: {default_deploy_path})"
    )
    addnode_parser.add_argument("--role", required=True, choices=['P', 'D', 'C'], help="Node role")
    addnode_parser.add_argument("--name", required=True, help="Node name")
    addnode_parser.add_argument("--host_ip", required=True, help="host_ip")
    addnode_parser.add_argument("--user", default="root", help="ansible_user")
    addnode_parser.add_argument("--ssh_common_args", default="-o StrictHostKeyChecking=no -o IdentitiesOnly=yes", help="ssh_common_args")
    addnode_parser.add_argument("--ssh_private_key_file", required=True, help="ssh_private_key_file")
    addnode_parser.add_argument("--master-node",
                                metavar="<master-node>, e.g. d0",
                                default=None,
                                help="The default value is set to current node name.")
    addnode_parser.add_argument("--docker_image_id", required=True, help="docker_image_id")
    addnode_parser.set_defaults(func=add_node)

    # RM_NODE command configuration
    rmnode_parser = subparsers.add_parser("rm_node", help="Remove a node from server_profiles.yml")
    rmnode_parser.add_argument("--role", required=True, choices=['P', 'D', 'C'], help="Node role")
    rmnode_parser.add_argument(
        "--deploy_path",
        default=str(default_deploy_path),
        help=f"Path to server_profiles.yml (default: {default_deploy_path})"
    )
    rmnode_parser.add_argument("--name", required=True, help="Node name to remove")
    rmnode_parser.set_defaults(func=rm_node)

    # RUN_DOCKER command configuration
    run_docker_parser = subparsers.add_parser("run_docker", help="Run Docker containers based on inventory")
    run_docker_parser.add_argument(
        "--config_path", "-i",
        default=None,
        help=f"Path to server_profiles.yml (default: {default_deploy_path})"
    )
    run_docker_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - show what would be done without executing"
    )
    run_docker_parser.set_defaults(func=lambda args: run_docker_containers(
        inventory_path=str(default_deploy_path),
        dry_run=args.dry_run
    ))
    # SYNC_CODE command configuration
    sync_parser = subparsers.add_parser("sync_code", help="Developer mode: Synchronize the code")
    sync_parser.add_argument(
        "--deploy_path",
        default=str(default_deploy_path),
        help=f"Path to server_profiles.yml (default: {default_deploy_path})"
    )
    sync_parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without making any changes"
    )
    sync_parser.add_argument("--code_path", required=True, help="code_path")

    sync_parser.set_defaults(func=lambda args:sync_code(
        inventory_path=args.deploy_path,
        dry_run=args.dry_run,
        code_path=args.code_path
    ))

    # INSTALL_CODE command configuration
    install_parser = subparsers.add_parser("install_code", help="Developer mode: Install packages")
    install_parser.add_argument(
        "--deploy_path",
        default=str(default_deploy_path),
        help=f"Path to server_profiles.yml (default: {default_deploy_path})"
    )
    install_parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without making any changes"
    )
    install_parser.set_defaults(func=lambda args:install_code(
        inventory_path=args.deploy_path,
        dry_run=args.dry_run
    ))

    args = parser.parse_args()

    if hasattr(args, 'func'):
        if  hasattr(args, 'config_path') and args.config_path is not None:
            default_deploy_path = args.config_path
        else:
            args.deploy_path = get_default_deploy_path(args.command)
            default_deploy_path = args.deploy_path
        args.func(args)
        return
    else:
        if not hasattr(args, 'config_path') or (args.config_path is None and args.normal is None):
            args.deploy_path = get_default_deploy_path(args.command)
            default_deploy_path = args.deploy_path
        else:
            if args.config_path is None:
                default_deploy_path = args.normal
            if args.normal is None:
                default_deploy_path = args.config_path

    if args.command == "start" and not any([args.normal, args.run_dev]):
        args.normal = True

    if args.command == "start":
        if args.config_path is None:
            args.config_path = default_deploy_path
        if args.normal:
            print(f"{INFO} Starting omni service in Normal mode...")
            omni_cli_start(inventory_path=args.config_path,
                           skip_verify_config=args.skip_verify_config,
                           dev=False,
                           proxy_only=args.proxy_only)
        elif args.run_dev:
            print(f"{INFO} Starting omni service in Developer mode...")
            omni_cli_start(inventory_path=args.config_path,
                           skip_verify_config=args.skip_verify_config,
                           dev=True,
                           proxy_only=args.proxy_only)
    elif args.command == "stop":
        print(f"{INFO} Stopping omni service...")
        omni_cli_stop(inventory_path=default_deploy_path)
    elif args.command == "cfg":
        node_type, node_name = parse_node_name(args.name[0])
        sections = parse_remaining_args(node_type, node_name, args.set, args.remaining_args, default_deploy_path)
        if args.set:
            print(f"{INFO} Set configuration.")
            cfg_set_process(node_type, node_name, args, sections, default_deploy_path)
        elif args.delete:
            print(f"{INFO} Delete configuration.")
            cfg_delete_process(node_type, node_name, args, sections, default_deploy_path)
    elif args.command == "inspect":
        print(f"{INFO} Inspect configuration.")
        print_node_config(default_deploy_path, args.name[0])
    elif args.command == "upgrade":
        print(f"{INFO} Upgrade packages")
        upgrade_packages()
    elif args.command == "collect_log":
        print(f"{INFO} Fetch logs")
        collect_logs()

if __name__ == "__main__":
    main()
