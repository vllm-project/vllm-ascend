import yaml

def print_node_config(inventory_file, node_id):
    with open(inventory_file, 'r') as file:
        inventory = yaml.safe_load(file)

    node_config = None
    for group in inventory['all']['children'].values():
        if node_id in group.get('hosts', {}):
            node_config = group['hosts'][node_id]
            break

    if not node_config:
        print(f"Node {node_id} not found in inventory")
        return
    
    env_config = node_config.get('env', {})
    args_config = node_config.get('args', {})

    print("env:")
    if env_config:
        max_key_len = max(len(key) for key in env_config.keys())
        for key, value in env_config.items():
            print(f"   {key}:{' ' * (max_key_len - len(key))} {value}")

    print("\nargs:")
    if args_config:
        max_key_len = max(len(key) for key in args_config.keys())
        for key, value in args_config.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                sub_max_len = max(len(k) for k in value.keys())
                for sub_key, sub_value in value.items():
                    print(f"      {sub_key}:{' ' * (sub_max_len - len(sub_key))} {sub_value}")
            else:
                print(f"   {key}:{' ' * (max_key_len - len(key))} {value}")