import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import tempfile
import os
import shlex

def register_values(inventory):
    # calculate PREFILL_API_SERVER_LIST
    prefill_api_server_list = []
    
    for host, vars in inventory['all']['children']['P']['hosts'].items():
        ansible_host_val = vars.get('ansible_host', '')
        host_ip_val = vars.get('host_ip', '')
        api_port_val = vars.get('env', '').get('API_PORT', '')
        
        if ansible_host_val and host_ip_val and ansible_host_val == host_ip_val:
            entry = f"{ansible_host_val}:{api_port_val}"
            if entry not in prefill_api_server_list:
                prefill_api_server_list.append(entry)

    prefill_api_server_list_result = ','.join(prefill_api_server_list)

    # calculate DECODE_API_SERVER_LIST
    decode_api_server_list = []
    
    for host, vars in inventory['all']['children']['D']['hosts'].items():
        ip = vars.get('ansible_host', '')
        api_port_val = vars.get('env', {}).get('API_PORT', '')

        tp_str = vars.get('args', {}).get('tp', '1')
        try:
            tp = int(tp_str)
        except (ValueError, TypeError):
            tp = 1

        devices_str = vars.get('ascend_rt_visible_devices', '')
        device_count = devices_str.count(',') + 1

        num = int(device_count / tp)
        
        if ip:
            entry = f"{ip}:{api_port_val}@{num}"
            if entry not in decode_api_server_list:
                decode_api_server_list.append(entry)

    decode_api_server_list_result = ','.join(decode_api_server_list)

    return {
        'PREFILL_API_SERVER_LIST': prefill_api_server_list_result,
        'DECODE_API_SERVER_LIST': decode_api_server_list_result
    }

def process_results(results, inventory, inv_file):

    prefill_result = results['PREFILL_API_SERVER_LIST']
    decode_api_servers = results['DECODE_API_SERVER_LIST']

    prefill_result = ' '.join(prefill_result.split())

    decode_result = ""

    decode_array = decode_api_servers.split(',')

    for var in decode_array:
        address = var.split('@')[0]
        num = int(var.split('@')[1])

        ip = address.split(':')[0]
        port = int(address.split(':')[1])

        for i in range(num):
            if not decode_result:
                decode_result = f"{ip}:{port}"
            else:
                decode_result += f",{ip}:{port}"
            port += 1

    for host, vars in inventory['all']['children']['C']['hosts'].items():
        host_ip_val = vars.get('ansible_host', '')
        api_port_val = vars.get('env', '').get('API_PORT', '')
        container_name = vars.get('container_name', '')

        env: Dict[str, Any] = vars.get("env", {}) or {}
        log_path = str(env.get("LOG_PATH") or "").strip()

        args: Dict[str, Any] = vars.get("args", {}) or {}
        prefill_lb_sdk = args.get('prefill-lb-sdk', 'pd_score_balance')
        decode_lb_sdk = args.get('decode-lb-sdk', 'pd_score_balance')

    with tempfile.NamedTemporaryFile(
        "w", delete=False,
        dir="./",
        prefix=f"omni_proxy_",
        suffix=".sh") as tf:
        script_path = Path(tf.name)
        tf.write("#!/usr/bin/env bash\n")
        tf.write(f"docker exec -i {shlex.quote(container_name)} bash -s <<'EOF'\n")
        tf.write(f"ps aux | grep 'nginx' | grep -v 'grep' | awk '{{print $2}}' | xargs kill -9; cd /workspace/omniinfer/tools/scripts; bash global_proxy.sh \\\n\
          --listen-port {api_port_val} \\\n\
          --prefill-servers-list {prefill_result} \\\n\
          --decode-servers-list {decode_result} \\\n\
          --log-file {log_path}/nginx/nginx_error.log \\\n\
          --log-level notice \\\n\
          --core-num 4 \\\n\
          --start-core-index 16 \\\n\
          --prefill-lb-sdk {prefill_lb_sdk} \\\n\
          --decode-lb-sdk {decode_lb_sdk}\n\n")
        tf.write("EOF\n")

    os.chmod(script_path, 0o755)

    cmd = (
        f"ansible {shlex.quote(host)} "
        f"-i {shlex.quote(str(inv_file))} "
        f"-m script "
        f"-a {shlex.quote(str(script_path))}"
    )

    try:
        os.system(cmd)
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except Exception:
            pass

def omni_run_proxy(inventory):
    inv_file = Path(inventory).expanduser().resolve()
    inv = None
    with open(inv_file, "r", encoding="utf-8") as f:
        inv = yaml.safe_load(f)
    
    result = register_values(inv)
    process_results(result, inv, inv_file)
