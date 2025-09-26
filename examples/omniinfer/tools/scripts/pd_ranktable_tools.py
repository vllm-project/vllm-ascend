import argparse
import json
import logging
import os
import socket
import stat
from collections import defaultdict
from copy import deepcopy

SCHEDULER_GROUP = "0"
PREFILL_GROUP = "1"
DECODE_GROUP = "2"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s-%(levelname)s: %(message)s')


def str2list(input_str):
    str_list = input_str.split(",")
    try:
        str_list = [str(s) for s in str_list]
    except Exception as e:
        raise ValueError("Input format error.") from e

    return str_list


def dump_json(save_path, json_dict):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    if os.path.exists(save_path):
        os.remove(save_path)
    with os.fdopen(os.open(save_path, flags, modes), 'w') as f:
        json.dump(json_dict, f, indent=4)
    logging.info(f"Save %s.", save_path)


def load_json(load_path):
    with open(load_path, 'r') as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="PD separate need generate ranktable.")
    parser.add_argument('--mode', default='gen', choices=['gen', 'merge', 'merge-local', 'merge-all'],
                        help="The mode of tools, `gen` for generating the global ranktable, "
                             "`merge` for merging the global ranktable.")

    # For gen mode.
    parser.add_argument('--api-server', action='store_true',
                        help="Use API server for receiving the request.")
    parser.add_argument('--prefill-server-list', nargs='*', default=[], type=str2list,
                        help="List of prefill servers (default is an empty list).")
    parser.add_argument('--decode-server-list', nargs='*', default=[], type=str2list,
                        help="List of decode servers (default is an empty list).")
    parser.add_argument('--save-dir', default='./', type=str,
                        help="Directory to save files (default is './').")
    parser.add_argument('--ip', default='', type=str,
                        help="local ip.")

    # For merge mode.
    parser.add_argument('--global-ranktable-list', nargs='*', default=[], type=str,
                        help="List of global rank tables (default is an empty list).")
    parser.add_argument('--local-ranktable-list', nargs='*', default=[], type=str,
                        help="List of local rank tables (default is an empty list).")
    parser.add_argument('--api-server-list', nargs='*', default=[], type=str,
                        help="List of api server.")

    args = parser.parse_args()
    verify_server_args(args)
    return args


def verify_server_args(args):
    # Tensor-parallel-size should be same.
    if args.prefill_server_list:
        all_prefill_server_device_num = [len(server) for server in args.prefill_server_list]
    else:
        all_prefill_server_device_num = None
    if args.decode_server_list:
        all_decode_server_device_num = [len(server) for server in args.prefill_server_list]
    else:
        all_decode_server_device_num = None

    if all_prefill_server_device_num and \
            max(all_prefill_server_device_num) != min(all_prefill_server_device_num):
        raise ValueError("All the tensor-parallel-size of the prefill server must be same.")
    if all_decode_server_device_num and \
            max(all_decode_server_device_num) != min(all_decode_server_device_num):
        raise ValueError("All the tensor-parallel-size of the decode server must be same.")
    if all_prefill_server_device_num and all_decode_server_device_num and \
            all_decode_server_device_num[0] != all_prefill_server_device_num[0]:
        raise ValueError("The tensor-parallel-size of prefill server and decode server must be same.")

    # Unique device_id.
    server_set = set()
    device_count = 0
    for server in args.prefill_server_list:
        server_set.update(server)
        device_count += len(server)

    for server in args.decode_server_list:
        server_set.update(server)
        device_count += len(server)

    if len(server_set) != device_count:
        raise ValueError('A device_id can be used only once in prefill server and decode server.')


def generate_global_ranktable(args):
    global_ranktable = {
        "version": "1.0",
        "status": "completed",
        "server_group_list": []
    }
    if args.api_server:
        global_ranktable["server_group_list"].append(generate_group(args, SCHEDULER_GROUP))
    if args.prefill_server_list:
        global_ranktable["server_group_list"].append(generate_group(args, PREFILL_GROUP, args.prefill_server_list))
    if args.decode_server_list:
        global_ranktable["server_group_list"].append(generate_group(args, DECODE_GROUP, args.decode_server_list))

    local_ip = args.ip if args.ip else get_host_ip()
    dump_json(os.path.join(args.save_dir, f"global_ranktable_{local_ip}.json"), global_ranktable)


def generate_local_ranktable(args):
    # Only support all device in one server.
    local_ranktable_base = {
        "version": "1.0",
        "status": "completed",
        "group_id": "0",
        "server_count": "1",
        "server_list": [
            {
                "server_id": None,
                "server_ip": None,
            }
        ]
    }

    for device_list in args.prefill_server_list + args.decode_server_list + (['host'] if args.api_server else []):
        local_ranktable = deepcopy(local_ranktable_base)
        local_ip = args.ip if args.ip else get_host_ip()
        local_ranktable['server_list'][0]["server_id"] = local_ip
        local_ranktable['server_list'][0]["server_ip"] = local_ip
        if device_list != 'host':
            local_ranktable['server_list'][0]["device"] = get_device(device_list)

        dump_json(os.path.join(args.save_dir, f"local_ranktable_{local_ip}_{''.join(device_list)}.json"),
                  local_ranktable)


def generate_group(args, group_id, server_list=None):
    group_info = {
        "group_id": group_id,
        "server_count": str(len(server_list) if server_list else 1),
        "server_list": []
    }
    local_ip = args.ip if args.ip else get_host_ip()
    default_server = {
        "server_id": local_ip,
        "server_ip": local_ip,
    }
    if server_list is None:
        group_info["server_list"].append(default_server)
        return group_info
    for device_list in server_list:
        server = deepcopy(default_server)
        server['device'] = get_device(device_list)
        group_info["server_list"].append(server)

    return group_info


def merge_global_ranktable(args):
    if len(args.global_ranktable_list) <= 1:
        raise ValueError("Ensure that there are two global ranktable for merge mode.")
    whole_server_group = defaultdict(list)
    for ranktable_path in args.global_ranktable_list:
        ranktable = load_json(ranktable_path)
        for server_group in ranktable.get("server_group_list", []):
            group_id = server_group.get("group_id")
            server_list = server_group.get("server_list")
            if group_id and server_list:
                whole_server_group[group_id].extend(server_list)

    if not whole_server_group.get(SCHEDULER_GROUP) or len(whole_server_group.get(SCHEDULER_GROUP)) != 1:
        raise ValueError("All global ranktable can contain only one group_id 0 field.")

    global_ranktable = {
        "version": "1.0",
        "status": "completed",
        "server_group_list": []
    }
    for group_id in [SCHEDULER_GROUP, PREFILL_GROUP, DECODE_GROUP]:
        server_group = whole_server_group.get(group_id)
        if server_group:
            group_info = {
                "group_id": group_id,
                "server_count": str(len(server_group)),
                "server_list": server_group
            }
            global_ranktable["server_group_list"].append(group_info)

    dump_json(os.path.join(args.save_dir, f"global_ranktable_merge.json"), global_ranktable)


def merge_local_ranktable(args):
    if len(args.local_ranktable_list) <= 1:
        raise ValueError("Ensure that there are two local ranktable for merge mode.")

    device_id = 0
    new_server_list = []
    for ranktable_path in args.local_ranktable_list:
        ranktable = load_json(ranktable_path)
        server_list = ranktable.get("server_list")[0]
        device_list = server_list.get("device")
        new_device_list = []
        for ip_info in device_list:
            ip_info.update({'rank_id': str(device_id)})
            device_id += 1
            new_device_list.append(ip_info)
        single_device = {
            "server_id": server_list.get("server_id"),
            "server_ip": server_list.get("server_ip"),
            "device": new_device_list
        }
        new_server_list.append(single_device)

    local_ranktable = {
        "version": "1.0",
        "status": "completed",
        "group_id": "0",
        "server_count": len(new_server_list),
        "server_list": new_server_list
    }

    dump_json(os.path.join(args.save_dir, f"local_ranktable_merge.json"), local_ranktable)


def merge_all(args):
    if not args.prefill_server_list or not args.decode_server_list:
        raise ValueError("Ensure provide prefill and decode server.")

    api_server_list = args.api_server_list
    prefill_server_list = args.prefill_server_list
    decode_server_list = args.decode_server_list

    global_ranktable = {
        "version": "1.0",
        "status": "completed",
        "server_group_list": []
    }

    server_list = []
    for api_server_path in api_server_list:
        temp_api_server = load_json(api_server_path)
        server_list.append(temp_api_server.get("server_list")[0])

    group_id = 0
    for _, server_list in enumerate([prefill_server_list, decode_server_list]):
        for server_info in server_list:
            group_info = {
                "group_id": str(group_id),
                "server_count": 1,
                "server_list": []
            }
            group_id += 1

            server_path = server_info[0]
            server = load_json(server_path)
            server_count = 0
            for single_server_list in server.get("server_list"):
                server_count += 1
                group_info["server_list"].append(single_server_list)
            group_info["server_count"] = server_count

            global_ranktable["server_group_list"].append(group_info)

    dump_json(os.path.join(args.save_dir, f"global_ranktable_merge.json"), global_ranktable)


def get_device(device_list):
    device_info = []
    for rank_id, device_id in enumerate(device_list):
        device = {
            "device_id": str(device_id),
            "device_ip": get_device_ip(device_id),
            "rank_id": str(rank_id)
        }
        device_info.append(device)
    return device_info


def get_device_ip(device_id):
    return os.popen(f"hccn_tool -i {device_id} -ip -g").readlines()[0].split(":")[1].replace('\n', '')


def get_host_ip():
    ip = None

    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
    except EOFError:
        pass

    return ip


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    if args.mode == 'gen':
        generate_global_ranktable(args)
        local_ip = args.ip if args.ip else get_host_ip()
        logging.info(f'Generate global ranktable successful, host ip is %s', local_ip)
        generate_local_ranktable(args)
        logging.info(f'Generate local ranktable successful.')
    elif args.mode == 'merge':
        merge_global_ranktable(args)
        logging.info(f'Merge %d global ranktable successful.', len(args.global_ranktable_list))
    elif args.mode == 'merge-local':
        merge_local_ranktable(args)
        logging.info(f'Merge %d local ranktable successful.', len(args.local_ranktable_list))
    elif args.mode == 'merge-all':
        merge_all(args)
        logging.info(f'Merge %d api server %d prefill server %d decode server successful.', len(args.api_server_list),
                     len(args.prefill_server_list), len(args.decode_server_list))


if __name__ == '__main__':
    main()
