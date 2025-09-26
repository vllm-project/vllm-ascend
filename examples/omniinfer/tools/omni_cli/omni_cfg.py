import yaml
import os
import re
import shlex

INFO    = "\033[92m[INFO]\033[0m"      # green
WARNING = "\033[93m[WARNING]\033[0m"   # yellow
ERROR   = "\033[91m[ERROR]\033[0m"     # red

def get_data_from_yaml(yml_file_path):
    try:
        with open(yml_file_path, 'r') as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: The %s file is not exist." % yml_file_path)
        return None

    return data

def parse_node_name(name):
    """Parse node name, return node type and index"""
    if not name:
        return None, None

    match = re.match(r'^([pdc](?:0|[1-9]\d*)?|all)$', name)
    if match:
        full_match = match.group(0)
        if full_match == 'all':
            return full_match, None
        else:
            node_type = full_match[0].upper()
            return node_type, full_match
    return None, None

def convert_to_dict(s):
    if ':' not in s:
        return s

    parts = s.split(":", 1)
    key = parts[0].strip()
    value_str = parts[1].strip()
    if not key or not value_str:
        return s

    return {key: value_str}

def parse_remaining_args_for_set(arg, remaining_args, sections, i):
    if remaining_args[i+1] in ['--additional-config', '--extra-args', '--kv-transfer-config']:
        if i + 2 >= len(remaining_args):
            raise ValueError(f"Missing value for key: '{remaining_args[i+1]}'")
        extra_args_list = shlex.split(remaining_args[i+2])
        j = 0
        while j < len(extra_args_list):
            extra_arg = extra_args_list[j]
            if not extra_arg.startswith('--'):
                raise ValueError(f"Invalid key format: '{extra_arg}'. Keys must start with '--'")
            elif (j + 1 < len(extra_args_list) and extra_args_list[j+1].startswith('--')) or \
                j + 1 == len(extra_args_list):
                sections.setdefault(arg, {}).setdefault(remaining_args[i+1][2:], {})[extra_arg[2:]] = ''
                j += 1
            elif j + 1 < len(extra_args_list) and not extra_args_list[j+1].startswith('--'):
                extra_args_dict = convert_to_dict(extra_args_list[j+1])
                sections.setdefault(arg, {}).setdefault(remaining_args[i+1][2:], {})[extra_arg[2:]] = extra_args_dict
                j += 2
    elif i + 2 >= len(remaining_args) or remaining_args[i+2].startswith('--'):
        raise ValueError(f"Missing value for key: '{remaining_args[i+1]}'")
    else:
        sections[arg][remaining_args[i+1][2:]] = remaining_args[i+2]

def parse_remaining_args_for_delete(arg, remaining_args, sections, i):
    if remaining_args[i+1] in ['--additional-config', '--extra-args', '--kv-transfer-config']:
        if i + 2 >= len(remaining_args):
            raise ValueError(f"Missing value for key: '{remaining_args[i+1]}'")
        additional_config_list = shlex.split(remaining_args[i+2])
        j = 0
        while j < len(additional_config_list):
            additional_config = additional_config_list[j]
            if not additional_config.startswith('--'):
                raise ValueError(f"Invalid key format: '{additional_config}'. Keys must start with '--'")
            else:
                sections[remaining_args[i+1][2:]].append(additional_config[2:])
                j += 1
        i += 1
    else:
        sections[arg].append(remaining_args[i+1][2:])

    return i

def update_container_name(node_type, node_name, container_name_prefix, yml_file_path):
    data = get_data_from_yaml(yml_file_path)
    if data:
        if node_type == 'all':
            for n_type in data['all']['children']:
                for n_name in data['all']['children'][n_type]['hosts']:
                    data['all']['children'][n_type]['hosts'][n_name]['container_name'] = \
                        f'{container_name_prefix}_{n_name}'
        elif node_type == 'P' or node_type == 'D' or node_type == 'C':
            for n_name in data['all']['children'][node_type]['hosts']:
                data['all']['children'][node_type]['hosts'][n_name]['container_name'] = \
                    f'{container_name_prefix}_{n_name}'
        else:
            data['all']['children'][node_type]['hosts'][node_name]['container_name'] = \
                    f'{container_name_prefix}_{node_name}'

        with open(yml_file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    else:
        raise ValueError(f"There is no data in {yml_file_path}")
        return

def parse_remaining_args(node_type, node_name, is_set, remaining_args, yml_file_path):
    """Resolve the remaining parameters."""
    if is_set:
        sections = {'env': {}, 'args': {}, 'DOCKER_IMAGE_ID': '', 'ascend_rt_visible_devices': '', 'container_name': ''}
    else:
        sections = {'env': [], 'args': [], 'DOCKER_IMAGE_ID': '', 'ascend_rt_visible_devices': '', \
            'container_name': '', 'extra-args': [], 'additional-config': [], 'kv-transfer-config': []}
    seen_sections = set()

    i = 0
    arg = remaining_args[0]
    while i < len(remaining_args):
        if remaining_args[i] in list(sections.keys())[:2]:
            arg = remaining_args[i]
            if remaining_args[i] in seen_sections:
                raise ValueError(f"Duplicate section keyword '{remaining_args[i]}'")
            seen_sections.add(remaining_args[i])
            if i + 1 >= len(remaining_args) or not remaining_args[i+1].startswith('--'):
                raise ValueError(f"Missing value for key: '{remaining_args[i]}'")
            elif remaining_args[i+1][2:] not in list(sections.keys())[2:5]:
                if is_set:
                    parse_remaining_args_for_set(arg, remaining_args, sections, i)
                    i += 3
                else:
                    i = parse_remaining_args_for_delete(arg, remaining_args, sections, i)
                    i += 2
            else:
                raise ValueError(f"Unexpected argument '{remaining_args[i+1]}' after (env/arg)")
        elif remaining_args[i][2:] in list(sections.keys())[2:5]:
            arg = remaining_args[i]
            if arg in seen_sections:
                raise ValueError(f"Duplicate section keyword '{arg}'")
            seen_sections.add(arg)
            if is_set:
                if i + 1 >= len(remaining_args) or remaining_args[i+1].startswith('--'):
                    raise ValueError(f"Missing value for key: '{arg}'")
                sections[arg[2:]] = remaining_args[i+1]
                i += 2
            else:
                sections[arg[2:]] = True
                i += 1
        elif remaining_args[i][2:] == 'container_name_prefix':
            print("Please note that you are setting the container name using the container name prefix.")
            if is_set:
                if i + 1 >= len(remaining_args) or remaining_args[i+1].startswith('--'):
                    raise ValueError(f"Missing value for key: '{remaining_args[i]}'")
                update_container_name(node_type, node_name, remaining_args[i+1], yml_file_path)
                i += 2
            else:
                raise ValueError(f"Unexpected key {arg}")
        else:
            if arg in list(sections.keys())[:2]:
                if i + 1 >= len(remaining_args) or not remaining_args[i+1].startswith('--'):
                    raise ValueError(f"Missing value for key: '{remaining_args[i]}'")
                elif remaining_args[i+1][2:] not in list(sections.keys())[2:]:
                    if is_set:
                        parse_remaining_args_for_set(arg, remaining_args, sections, i)
                        i += 3
                    else:
                        i = parse_remaining_args_for_delete(arg, remaining_args, sections, i)
                        i += 2
                else:
                    raise ValueError(f"Unexpected argument '{remaining_args[i+1]}' after (env/arg)")
            else:
                raise ValueError(f"Unexpected key {remaining_args[i]}")

    return sections

def check_model_path(sections, data, node_type, node_name):
    if node_type == 'C':
        return True

    model_path_is_none = 'MODEL_PATH' in data['all']['children'][node_type]['hosts'][node_name]['env'] and \
        (data['all']['children'][node_type]['hosts'][node_name]['env']['MODEL_PATH'] == '' or \
        data['all']['children'][node_type]['hosts'][node_name]['env']['MODEL_PATH'] == None)
    if 'MODEL_PATH' not in data['all']['children'][node_type]['hosts'][node_name]['env'] or model_path_is_none:
        if 'MODEL_PATH' in sections['env']:
            if 'deepseek' in sections['env']['MODEL_PATH'].lower() or 'qwen' in sections['env']['MODEL_PATH'].lower():
                return True
            else:
                print("Error: This model is currently not supported.")
                return False
        else:
            print("Error: This model is not configured.")
            return False
    else:
        model_path = data['all']['children'][node_type]['hosts'][node_name]['env']['MODEL_PATH']
        if 'deepseek' in model_path.lower() or 'qwen' in model_path.lower():
            return True
        else:
            print("Error: This model is not configured or This model is currently not supported.")
            return False

def updata_dict(sections, data):
    for modify_key, modify_values in sections.items():
        if isinstance(modify_values, dict) and modify_key in data:
            updata_dict(modify_values, data[modify_key])
        elif isinstance(modify_values, dict) and modify_key not in data:
            data[modify_key] = {}
            updata_dict(modify_values, data[modify_key])
        else:
            data[modify_key] = modify_values

def update_cfg_yml(node_type, node_name, sections, yml_file_path):
    data = get_data_from_yaml(yml_file_path)
    filtered_sections = {k: v for k, v in sections.items() if v not in ['', None, {}]}
    if data:
        if node_type == 'all':
            for n_type in data['all']['children']:
                for n_name in data['all']['children'][n_type]['hosts']:
                    updata_dict(filtered_sections, data['all']['children'][n_type]['hosts'][n_name])
            print(f"{INFO} You have modified the configuration of all nodes")
        elif node_name == 'p' or node_name == 'd' or node_name == 'c':
            for n_name in data['all']['children'][node_type]['hosts']:
                updata_dict(filtered_sections, data['all']['children'][node_type]['hosts'][n_name])
            print(f"{INFO} You have modified the configuration of all nodes in the group {node_type}")
        else:
            updata_dict(filtered_sections, data['all']['children'][node_type]['hosts'][node_name])
            print(f"{INFO} You have modified the configuration of node {node_name}")

        with open(yml_file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    else:
        print(f"{ERROR} There is no data in {yml_file_path}.")
        return

def delete_cfg_yml_for_node(data, node_type, node_name, env_list, arg_list, DOCKER_IMAGE_ID, \
    ascend_rt_visible_devices, container_name, extra_args_list, additional_config_list, \
    kv_transfer_config_list):
    vars_dict = data['all']['children'][node_type]['hosts'][node_name]
    for key in env_list:
        if key in vars_dict['env']:
            del vars_dict['env'][key]
        else:
            print(f"{WARNING} No matching configuration {key} found in {node_name}.")

    for key in arg_list:
        if key in vars_dict['args']:
            del vars_dict['args'][key]
        else:
            print(f"{WARNING} No matching configuration {key} found in {node_name}.")

    if DOCKER_IMAGE_ID and 'DOCKER_IMAGE_ID' in vars_dict:
        del vars_dict['DOCKER_IMAGE_ID']

    if ascend_rt_visible_devices and 'ascend_rt_visible_devices' in vars_dict:
        del vars_dict['ascend_rt_visible_devices']

    if container_name and 'container_name' in vars_dict:
        del vars_dict['container_name']

    for key in extra_args_list:
        if 'extra-args' in vars_dict['args'] and key in vars_dict['args']['extra-args']:
            del vars_dict['args']['extra-args'][key]
        else:
            print(f"{WARNING} No matching configuration {key} found in {node_name}.")

    if 'extra-args' in vars_dict['args'] and vars_dict['args']['extra-args'] == {}:
        vars_dict['args']['extra-args'] = ''

    for key in additional_config_list:
        if 'additional-config' in vars_dict['args'] and key in vars_dict['args']['additional-config']:
            del vars_dict['args']['additional-config'][key]
        else:
            print(f"{WARNING} No matching configuration {key} found in {node_name}.")

    if 'additional-config' in vars_dict['args'] and vars_dict['args']['additional-config'] == {}:
        vars_dict['args']['additional-config'] = ''

    for key in kv_transfer_config_list:
        if key in vars_dict['args']['kv-transfer-config']:
            del vars_dict['args']['kv-transfer-config'][key]
        else:
            print(f"{WARNING} No matching configuration {key} found in {node_name}.")

    if 'kv-transfer-config' in vars_dict['args'] and vars_dict['args']['kv-transfer-config'] == {}:
        vars_dict['args']['kv-transfer-config'] = ''

def delete_model_path(sections):
    default_cfg_path = f'{os.path.dirname(__file__)}/configs/default_profiles.yml'
    default_cfg = get_data_from_yaml(default_cfg_path)

    if 'MODEL_PATH' in sections['env']:
        if 'model_path_used' not in default_cfg:
            print(f"{WARNING} The key 'MODEL_PATH' does not exist, there is no need to delete it")
            return

        del default_cfg['model_path_used']

    with open(default_cfg_path , 'w') as file:
        yaml.dump(default_cfg, file, default_flow_style=False, sort_keys=False)

def delete_cfg_yml(node_type, node_name, sections, yml_file_path):
    env_list = sections['env']
    arg_list = sections['args']
    DOCKER_IMAGE_ID = sections['DOCKER_IMAGE_ID']
    ascend_rt_visible_devices = sections['ascend_rt_visible_devices']
    container_name = sections['container_name']
    extra_args_list = sections['extra-args']
    additional_config_list = sections['additional-config']
    kv_transfer_config_list = sections['kv-transfer-config']
    delete_model_path(sections)
    data = get_data_from_yaml(yml_file_path)
    if data:
        if node_type == 'all':
            for n_type in data['all']['children']:
                for n_name in data['all']['children'][n_type]['hosts']:
                    delete_cfg_yml_for_node(data, n_type, n_name, env_list, arg_list, DOCKER_IMAGE_ID, \
                        ascend_rt_visible_devices, container_name, extra_args_list, \
                        additional_config_list, kv_transfer_config_list)
            print(f"{INFO} You have deleted the configuration of all nodes")
        elif node_name == 'p' or node_name == 'd' or node_name == 'c':
            for n_name in data['all']['children'][node_type]['hosts']:
                delete_cfg_yml_for_node(data, node_type, n_name, env_list, arg_list, DOCKER_IMAGE_ID, \
                    ascend_rt_visible_devices, container_name, extra_args_list, \
                    additional_config_list, kv_transfer_config_list)
            print(f"{INFO} You have deleted the configuration of all nodes in group {node_type}")
        else:
            delete_cfg_yml_for_node(data, node_type, node_name, env_list, arg_list, DOCKER_IMAGE_ID, \
                ascend_rt_visible_devices, container_name, extra_args_list, \
                additional_config_list, kv_transfer_config_list)
            print(f"{INFO} You have deleted the configuration of node {node_name}")

        with open(yml_file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    else:
        print(f"{ERROR} There is no data in {yml_file_path}.")
        return

def modify_by_use_default_file(sections, default_cfg, data, node_type, node_name):
    if check_model_path(sections, data, node_type, node_name) is False:
        return False
    if 'MODEL_PATH' in sections['env']:
        if 'deepseek' in sections['env']['MODEL_PATH'].lower():
            sections_bak = default_cfg['profiles']['vllm']['deepseek'][node_type]
        elif 'qwen' in sections['env']['MODEL_PATH'].lower():
            sections_bak = default_cfg['profiles']['vllm']['qwen'][node_type]
        else:
            print("Error: This model is currently not supported.")
            return False
        updata_dict(sections_bak, data['all']['children'][node_type]['hosts'][node_name])
    return True

def cfg_set_process(node_type, node_name, args, sections, deploy_path):
    if node_type is None and node_name is None:
        print(f"{ERROR} Invalid node name: '{args.name[0]}'。")
        print("The node name must conform to one of the following formats:")
        print("  - prefill_<number> (for example: p0, p1, p11)")
        print("  - decode_<number> (for example: d0, d1, d11)")
        return

    default_cfg_path = f'{os.path.dirname(__file__)}/configs/default_profiles.yml'
    default_cfg = get_data_from_yaml(default_cfg_path)
    data =  get_data_from_yaml(deploy_path)
    if data:
        if node_type == 'all':
            for n_type in default_cfg['profiles']['vllm']['deepseek']:
                for n_name in data['all']['children'][n_type]['hosts']:
                    if modify_by_use_default_file(sections, default_cfg, data, n_type, n_name) is False:
                        return
        elif node_type == 'P' or node_type == 'D' or node_type == 'C' and node_name is None:
            for n_name in data['all']['children'][node_type]['hosts']:
                if modify_by_use_default_file(sections, default_cfg, data, node_type, n_name) is False:
                    return
        else:
            if modify_by_use_default_file(sections, default_cfg, data, node_type, node_name) is False:
                return
        with open(deploy_path , 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    else:
        raise ValueError(f"There is no data in {deploy_path}")
        return

    update_cfg_yml(node_type, node_name, sections, deploy_path)

def cfg_delete_process(node_type, node_name, args, sections, deploy_path):
    if node_type is None and node_name is None:
        print(f"{ERROR} Invalid node name: '{args.name[0]}'。")
        print("The node name must conform to one of the following formats:")
        print("  - prefill_<number> (for example: p0, p1, p11)")
        print("  - decode_<number> (for example: d0, d1, d11)")
        return

    delete_cfg_yml(node_type, node_name, sections, deploy_path)