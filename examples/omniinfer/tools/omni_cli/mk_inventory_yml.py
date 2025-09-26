import yaml
import os

INFO    = "\033[92m[INFO]\033[0m"      # green
WARNING = "\033[93m[WARNING]\033[0m"   # yellow
ERROR   = "\033[91m[ERROR]\033[0m"     # red

def load_yaml(path):
    """Load YAML file content, return empty dict if file doesn't exist"""
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def save_yaml(path, data):
    """Save data to YAML file"""
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)

def add_node(args):
    """Add a node to server_profiles.yml under all.children.{role}.hosts"""
    # Locate default_profiles.yml and server_profiles.yml
    base_dir = os.path.dirname(__file__)
    default_path = os.path.join(base_dir, 'configs', 'default_profiles.yml')
    deploy_path = args.deploy_path
    default_profiles = load_yaml(default_path)

    # Validate default profiles structure
    if not default_profiles or 'profiles' not in default_profiles or 'vllm' not in default_profiles['profiles']:
        print(f"{ERROR} default_profiles.yml not found or invalid.")
        return

    # Load or create deployment file
    deployment = load_yaml(deploy_path)
    if 'all' not in deployment:
        # Initialize with default structure if empty
        deployment['all'] = {'children': {'P': {'hosts': {}}, 'D': {'hosts': {}}, 'C': {'hosts': {}}}}
    children = deployment['all']['children']

    # Ensure role exists in children
    if args.role not in children:
        children[args.role] = {'hosts': {}}
    hosts = children[args.role]['hosts']

    # Validate C node addition (only one allowed)
    if args.role == 'C' and hosts:
        print(f"{ERROR} Only one C node allowed.")
        return

    # Check for duplicate node name
    if args.name in hosts:
        print(f"{ERROR} Node name '{args.name}' already exists in role '{args.role}'.")
        return

    # Calculate port offset based on role and existing nodes
    role_count = len(hosts)
    if args.role == 'P':
        offset = 0
    elif args.role == 'D':
        offset = 100
    elif args.role == 'C':
        offset = 200
    else:
        print(f"{ERROR} role must be P, D, or C.")
        return


    # Set container name prefix based on role
    if args.role == 'P':
        container_name_prefix = "omni_infer_prefill"
    elif args.role == 'D':
        container_name_prefix = "omni_infer_decode"
    elif args.role == 'C':
        container_name_prefix = "omni_infer_proxy"

    # Create node information dictionary
    node = {
        'ansible_user': args.user,
        'ansible_ssh_common_args': args.ssh_common_args,
        'ansible_ssh_private_key_file': args.ssh_private_key_file,
        'ansible_host': args.host_ip,
        'container_name': f"{container_name_prefix}_{args.name}",  # Set container name
        'ascend_rt_visible_devices': '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
    }

    # Add DOCKER_IMAGE_ID if provided
    if hasattr(args, 'docker_image_id') and args.docker_image_id:
        node['DOCKER_IMAGE_ID'] = args.docker_image_id

    # Get role-specific configuration from default profiles
    role_config = default_profiles['profiles']['vllm']['deepseek'].get(args.role)
    env = role_config.get('env', {}).copy()

    # Set master IP if not provided
    if not args.master_node:
        args.master_node = args.name
    is_slave_node = args.master_node != args.name
    if is_slave_node:
        if args.master_node not in hosts:
            print(f"{ERROR} Master node '{args.master_node}' not found in role '{args.role}'. Please add the master node first.")
            return
        master_ip = hosts[args.master_node]['ansible_host']
        master_port = hosts[args.master_node]['env'].get('MASTER_PORT', None)
    else:
        master_ip = args.host_ip
        master_port = env.get('MASTER_PORT', None) # use default in default_profiles.yml

    # For P/D roles, add master_ip
    if args.role in ['P', 'D']:
        node['host_ip'] = master_ip
        node['master_node'] = args.master_node
        env['MASTER_PORT'] = master_port

    # Copy arguments if present
    if role_config.get('args', {}):
        args_dict = role_config.get('args', {}).copy()
        node['args'] = args_dict

    # Update PORT fields in environment variables
    for k in env:
        if k.endswith('PORT') and k != 'MASTER_PORT':
            user_port = None
            if user_port is not None:
                env[k] = user_port
            else:
                env[k] = env[k] + 16 * role_count + offset

    node['env'] = env

    # Add node to hosts and save deployment
    hosts[args.name] = node
    save_yaml(deploy_path, deployment)
    print(f"{INFO} Node '{args.name}' added successfully to role '{args.role}'.")

def rm_node(args):
    """Remove node from server_profiles.yml and reassign ports for the role"""
    base_dir = os.path.dirname(__file__)
    deploy_path = args.deploy_path
    deployment = load_yaml(deploy_path)

    # Validate deployment structure
    if 'all' not in deployment or 'children' not in deployment['all'] or args.role not in deployment['all']['children']:
        print(f"{ERROR} Role '{args.role}' not found.")
        return

    hosts = deployment['all']['children'][args.role]['hosts']
    if args.name not in hosts:
        print(f"{ERROR} Node '{args.name}' not found in role '{args.role}'.")
        return

    # Remove node
    del hosts[args.name]

    # Reassign ports for remaining nodes in this role
    default_path = os.path.join(base_dir, 'configs', 'default_profiles.yml')
    default_profiles = load_yaml(default_path)
    role_config = default_profiles['profiles']['vllm']['deepseek'].get(args.role)
    env_template = role_config.get('env', {}).copy()

    # Set offset based on role
    if args.role == 'P':
        offset = 0
    elif args.role == 'D':
        offset = 100
    elif args.role == 'C':
        offset = 200
    else:
        offset = 0

    # Update environment PORT fields for P/D roles
    if args.role in ['P', 'D']:
        sorted_items = list(hosts.items())
        for idx, (n, node) in enumerate(sorted_items):
            env = node.get('env', {})
            for k in env:
                if k.endswith('PORT'):
                    env[k] = env_template[k] + 16 * idx + offset
            node['env'] = env
            hosts[n] = node

    # Save updated deployment
    save_yaml(deploy_path, deployment)
    print(f"{INFO} Node '{args.name}' removed from role '{args.role}'. Ports reassigned.")