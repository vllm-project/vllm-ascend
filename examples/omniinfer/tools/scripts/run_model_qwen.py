# import 
import os
import sys
import time
import json
import fcntl
import socket
import struct
import argparse
import warnings
import subprocess

def get_path_before_omniinfer():
    """Get the base path before the 'omniinfer' directory in the current script's path.
    
    Returns:
        str: The path segment before 'omniinfer' directory.
    Raises:
        ValueError: If 'omniinfer' directory is not found in the path.
    """
    # Get absolute path of the currently executing script
    script_path = os.path.abspath(sys.argv[0])
    
    # Split path into components using OS-specific separator
    path_parts = script_path.split(os.sep)
    
    # Find the index of 'omniinfer' in the path components
    try:
        omni_index = path_parts.index('omniinfer')
    except ValueError:
        raise ValueError("'omniinfer' directory not found in path")
    
    # Reconstruct path up to (but not including) 'omniinfer'
    before_omni = os.sep.join(path_parts[:omni_index])
    
    return before_omni

def get_network_interfaces():
    """
    Retrieves primary network interface information excluding loopback.
    Returns a dictionary with interface name and its IP address.
    Falls back to 'eth0' if no interfaces found.
    """
    # List all network interfaces except loopback (lo)
    if_names = [name for name in os.listdir('/sys/class/net') if name != 'lo']
    
    # Select first available interface or default to 'eth0'
    if_name = if_names[0] if if_names else 'eth0'

    try:
        # Get IP address for selected interface
        ip = get_ip_address(if_name)

        # Compose result dictionary
        interfaces = {
            'if_name': if_name,  # Network interface name
            'ip': ip             # IPv4 address of the interface
        }   
    except Exception as e:
        print(f"Error getting network interfaces: {if_name}:{e}")
        interfaces = {}  # Return empty dict on error
    
    return interfaces

def get_ip_address(if_name):
    """
    Retrieves the IPv4 address of a network interface using ioctl.
    Args:
        if_name: Name of the network interface (e.g., 'eth0')
    Returns:
        IPv4 address as string
    Raises:
        RuntimeError on failure
    """
    # Create UDP socket for ioctl operations
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # SIOCGIFADDR = 0x8915 (get interface address)
        # Pack interface name into byte structure (max 15 chars)
        packed_ifname = struct.pack('256s', if_name[:15].encode('utf-8'))
        
        # Perform ioctl call to get interface info
        # [20:24] slices the IP address from the returned structure
        ip_bytes = fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR constant
            packed_ifname
        )[20:24]
        
        # Convert packed binary IP to dotted-quad string
        return socket.inet_ntoa(ip_bytes)
    except Exception as e:
        raise RuntimeError(f"Failed to get IP address for interface {if_name}: {e}")


def run_default_mode(args):
    """Run in mixed deployment mode"""

    if (args.network_interface is not None and args.host_ip is None) or \
        (args.network_interface is None and args.host_ip is not None):
        warnings.warn(
            "For best results, please specify both --network-interface AND --host-ip "
            "together. Falling back to auto-detection for missing values.",
            RuntimeWarning
        )
    # Get network interface
    if args.network_interface:
        intf = {'if_name': args.network_interface, 'ip': get_ip_address(args.network_interface)}
    else:
        intf = get_network_interfaces()
        if not intf:
            raise RuntimeError("No network interface found and none specified")

    # Override IP if host-ip was specified
    if args.host_ip:
        intf['ip'] = args.host_ip


    env = os.environ.copy()
    # Network config for distributed training
    env['GLOO_SOCKET_IFNAME'] = intf['if_name']
    env['TP_SOCKET_IFNAME'] = intf['if_name']

    # Hardware and framework settings
    env['ASCEND_RT_VISIBLE_DEVICES'] = args.server_list  # Use first 8 NPUs
    env['VLLM_WORKER_MULTIPROC_METHOD'] = 'fork'   # Process spawning method
    env['OMNI_USE_QWEN'] = '1'  # Enable custom model support
    env['VLLM_USE_V1'] = '1'
    env['ASCEND_GLOBAL_LOG_LEVEL'] = '3'
    env['VLLM_LOGGING_LEVEL'] = 'INFO'

    env['HCCL_OP_EXPANSION_MODE'] = 'AIV'
    env['TNG_HOST_COPY'] = '1'
    env['TASK_QUEUE_ENABLE'] = '2'
    env['CPU_AFFINITY_CONF'] = '2'

    if args.graph_true.lower() == 'false':
          # Base command for API server
        cmd = [
            'python',  os.path.join(args.code_path, 'omniinfer/tools/scripts/start_api_servers.py'),
            '--num-servers', '1',
            '--model-path', args.model_path,
            '--master-ip', intf['ip'],       # Coordinator IP
            '--master-port', args.master_port,         # Coordinator port
            '--tp', str(len(args.server_list.split(','))) ,   # tensor parallelism
            '--served-model-name', args.model_name,
            '--base-api-port', args.https_port,        # HTTP service port
            '--log-dir', args.log_path,  # Log directory
            '--extra-args', '--enforce-eager '   # Disable graph execution
        ]

        if hasattr(args, 'additional_config') and args.additional_config:
            cmd.extend(['--additional-config', args.additional_config])

    # Graph mode specific optimizations
    elif args.graph_true.lower() == 'true':
        # Base command for API server
        additional_config = args.additional_config if args.additional_config else \
                '{"graph_model_compile_config": {"level":1, "use_ge_graph_cached":false, "block_num_floating_range":50}, "enable_hybrid_graph_mode": false}'
        cmd = [
            'python',  os.path.join(args.code_path,'omniinfer/tools/scripts/start_api_servers.py'),
            '--num-servers', '1',
            '--model-path', args.model_path,
            '--master-ip', intf['ip'],       # Coordinator IP
            '--master-port', args.master_port,    # Coordinator port
            '--tp', str(len(args.server_list.split(','))),     # tensor parallelism
            '--served-model-name', args.model_name,
            '--base-api-port', args.https_port,        # HTTP service port
            '--log-dir', args.log_path,  # Log directory
            '--gpu-util', '0.9',  # Target NPU utilization
            '--max-model-len', args.max_model_len,
            '--extra-args', f'--max-num-batched-tokens {args.max_num_batched_tokens} --max-num-seqs {args.max_num_seqs} ',
            '--additional-config', additional_config
        ]
    
    print(f'Starting with NIC: {intf["if_name"]}, IP: {intf["ip"]}')

    subprocess.run(cmd, env=env)


def set_common_env_vars(intf, env):
    """Set common environment variables for all servers"""
    
    # Ascend NPU library path
    env['PYTHONPATH'] = '/usr/local/Ascend:' + env.get('PYTHONPATH', '')
    
    # Network configuration for distributed communication
    env['LOCAL_DECODE_SERVER_IP_LIST'] = intf['ip']  # Local decoder IP
    env['GLOBAL_DECODE_SERVER_IP_LIST'] = intf['ip'] # Global decoder IP
    env['GLOO_SOCKET_IFNAME'] = intf['if_name']      # Gloo communication interface
    env['TP_SOCKET_IFNAME'] = intf['if_name']        # Tensor parallelism interface
    
    # Framework configuration
    env['VLLM_USE_V1'] = '1'                       # Use vLLM v1 API
    env['VLLM_WORKER_MULTIPROC_METHOD'] = 'fork'    # Process spawning method
    env['VLLM_LOGGING_LEVEL'] = 'INFO'             # Log verbosity level
    env['ASCEND_GLOBAL_LOG_LEVEL'] = '3'
    
    # Custom model support
    env['OMNI_USE_QWEN'] = '1'  # Enable QWEN model optimizations

    # Pod configuration
    env['PREFILL_POD_NUM'] = '1'       # Prefill pod count
    env['DECODE_POD_NUM'] = '1'        # Decoder pod count



def start_perfill_api_servers(intf, args):
    """Start prefill API servers with specialized configuration"""
    ip = intf['ip']

    env = os.environ.copy()
    set_common_env_vars(intf, env)  # Apply common network settings
    
    # Specialized environment for prefill servers
    env['VLLM_LLMDATADIST_ZMQ_PORT'] = '5570'  # ZeroMQ port for data distribution
    env['ASCEND_RT_VISIBLE_DEVICES'] = args.prefill_server_list  # NPUs for prefill
    
    prefill_server_list_list = args.prefill_server_list.split(',')
    prefill_rank_table_suffix = ''.join(prefill_server_list_list)

    # Ranktable paths for distributed training
    env['RANK_TABLE_PATH'] = os.path.join(args.code_path, 'omniinfer/tools/scripts/global_path/')
    env['GLOBAL_RANK_TABLE_FILE_PATH'] = os.path.join(args.code_path, 'omniinfer/tools/scripts/global_path/global_ranktable_merge.json')
    env['RANK_TABLE_FILE_PATH'] = os.path.join(args.code_path, f'omniinfer/tools/scripts/perfill-ranktable/local_ranktable_{ip}_{prefill_rank_table_suffix}.json')
    env['ROLE'] = 'prefill'  # Server role identifier

    # HCCL communication settings
    env['HCCL_INTRA_ROCE_ENABLE'] = '1'  # Enable RoCE communication
    env['HCCL_INTRA_PCIE_ENABLE'] = '0'  # Disable PCIe communication
    env['HCCL_DETERMINISTIC'] = 'true'  # Enable deterministic behavior
    env['CLOSE_MATMUL_K_SHIFT'] = '1'  # Optimization flag
    env['HCCL_OP_EXPANSION_MODE'] = 'AIV'

    # Prefill-specific optimizations
    env['ENABLE_PREFILL_TND'] = '1'

    # KV transfer configuration for attention
    kv_transfer_config = {
        "kv_connector": "AscendHcclConnectorV1",
        "kv_buffer_device": "npu",
        "kv_role": "kv_producer",  # Prefill produces KV cache
        "kv_rank": 0,
        "engine_id": 0,
        "kv_parallel_size": 2
    }

    # Command to start prefill API servers
    cmd = [
        'python', os.path.join(args.code_path,'omniinfer/tools/scripts/start_api_servers.py'),
        '--num-servers', '1',
        '--model-path', args.model_path,
        '--master-ip', intf['ip'],  # Coordinator IP
        '--master-port', args.master_port,    # Coordinator port
        '--base-api-port', args.service_port,  # API service port
        '--tp', str(len(args.prefill_server_list.split(','))),                # 8-way tensor parallelism
        '--served-model-name', args.model_name,
        '--max-model-len', args.max_model_len, # Max context length
        '--log-dir', args.log_path + '/prefill/',  # Log directory
        '--no-enable-prefix-caching',  # Disable caching
        '--gpu-util', '0.9',        # Target NPU utilization
        '--extra-args', f'--max-num-batched-tokens {args.max_num_batched_tokens} --max-num-seqs {args.max_num_seqs} --enforce-eager',  # Perf mance flags
        '--kv-transfer-config', json.dumps(kv_transfer_config)  # KV transfer settings
    ]

    subprocess.Popen(cmd, env=env)  # Start as background process

def start_decoder_api_servers(intf, args):
    """Start decoder API servers with specialized configuration"""
    ip = intf['ip']

    env = os.environ.copy()
    set_common_env_vars(intf, env)  # Apply common network settings
    
    # Specialized environment for decoder servers
    env['VLLM_LLMDATADIST_ZMQ_PORT'] = '5569'  # Different ZeroMQ port
    env['ASCEND_RT_VISIBLE_DEVICES'] = args.decode_server_list  # Different NPU set
    
    deocde_server_list_list = args.decode_server_list.split(',')
    decode_rank_table_suffix = ''.join(deocde_server_list_list)

    # Ranktable paths for distributed training
    env['RANK_TABLE_PATH'] = os.path.join(args.code_path, 'omniinfer/tools/scripts/global_path/')
    env['GLOBAL_RANK_TABLE_FILE_PATH'] = os.path.join(args.code_path, 'omniinfer/tools/scripts/global_path/global_ranktable_merge.json')
    env['RANK_TABLE_FILE_PATH'] = os.path.join(args.code_path, f'omniinfer/tools/scripts/decode-ranktable/local_ranktable_{ip}_{decode_rank_table_suffix}.json')
    env['ROLE'] = 'decode'  # Server role identifier

    # Advanced HCCL settings
    env['HCCL_INTRA_ROCE_ENABLE'] = '1'  # Enable RoCE communication
    env['HCCL_INTRA_PCIE_ENABLE'] = '0'  # Disable PCIe communication
    env['HCCL_BUFFSIZE'] = '1000'       # Communication buffer size
    env['HCCL_OP_EXPANSION_MODE'] = 'AIV'  # Operation expansion mode
    env['VLLM_ENABLE_MC2'] = '1'        # Memory optimization

    # Debugging and profiling flags
    env['DUMP_GE_GRAPH'] = '2'
    env['DUMP_GRAPH_LEVEL'] = '3'

    # Decoder-specific optimizations
    env['DECODE_DP_SIZE'] = '1'         # Data parallelism size
    env['MOE_DISPATCH_COMBINE'] = '1'   # Mixture-of-Experts optimization
    env['HCCL_DETERMINISTIC'] = 'true'  # Enable deterministic behavior
    env['CLOSE_MATMUL_K_SHIFT'] = '1'   # Optimization flag

    
    # Server offset handling
    try:
        server_offset = env['SERVER_OFFSET']
    except KeyError:
        server_offset = '0'

    # KV transfer configuration for attention
    kv_transfer_config = {
        "kv_connector": "AscendHcclConnectorV1",
        "kv_buffer_device": "npu",
        "kv_role": "kv_consumer",  # Decoder consumes KV cache
        "kv_rank": 1,
        "engine_id": 0,
        "kv_parallel_size": 2
    }

    if args.graph_true.lower() == 'false':
          # Base command for API server
        # Command to start decoder API servers
        cmd = [
            'python', os.path.join(args.code_path,'omniinfer/tools/scripts/start_api_servers.py'),
            '--num-servers', '1',
            '--server-offset', server_offset,  # Server offset parameter
            '--num-dp', env['DECODE_DP_SIZE'],  # Data parallelism degree
            '--model-path', args.model_path,
            '--master-ip', intf['ip'],      # Coordinator IP
            '--master-port', args.master_port,        # Coordinator port
            '--base-api-port', str(int(args.service_port) + 100),      # API service port
            '--tp', str(len(args.decode_server_list.split(','))),                    # 8-way tensor parallelism
            '--served-model-name', args.model_name,
            '--max-model-len', args.max_model_len,    # Max context length
            '--log-dir', args.log_path + '/decode/',  # Log directory
            '--no-enable-prefix-caching',  # Disable caching
            '--extra-args', f'--max-num-batched-tokens {args.max_num_batched_tokens} --max-num-seqs {args.max_num_seqs}',  # Performance flag
            '--kv-transfer-config', json.dumps(kv_transfer_config)  # KV transfer settings
        ]

        if hasattr(args, 'additional_config') and args.additional_config:
            cmd.extend(['--additional-config', args.additional_config])

    # Graph mode specific optimizations
    elif args.graph_true.lower() == 'true':
        additional_config = args.additional_config if args.additional_config else \
                '{"graph_model_compile_config":{"level":1,"use_ge_graph_cached":false, "block_num_floating_range":50}, "decode_gear_list": [64]}'
        # Command to start decoder API servers
        cmd = [
            'python', os.path.join(args.code_path,'omniinfer/tools/scripts/start_api_servers.py'),
            '--num-servers', '1',
            '--server-offset', server_offset,  # Server offset parameter
            '--num-dp', env['DECODE_DP_SIZE'],  # Data parallelism degree
            '--model-path', args.model_path,
            '--master-ip', intf['ip'],      # Coordinator IP
            '--master-port', args.master_port,        # Coordinator port
            '--base-api-port', str(int(args.service_port) + 100),      # API service port
            '--tp', str(len(args.decode_server_list.split(','))),           # 8-way tensor parallelism
            '--served-model-name', args.model_name,
            '--max-model-len', args.max_model_len,    # Max context length
            '--log-dir', args.log_path + '/decode/',  # Log directory
            '--no-enable-prefix-caching',  # Disable caching
            '--extra-args', f'--max-num-batched-tokens {args.max_num_batched_tokens} --max-num-seqs {args.max_num_seqs} ',  # Performance flag
            '--additional-config', additional_config,  # Graph mode config
            '--kv-transfer-config', json.dumps(kv_transfer_config)  # KV transfer settings
        ]

    subprocess.Popen(cmd, env=env)  # Start as background process

def start_global_proxy(intf, args):
    """Start global proxy for routing requests"""
    env = os.environ.copy()
    env['PATH'] = '/usr/local/nginx:' + env.get('PATH', '')  # Ensure nginx in PATH

    # Start proxy script
    cmd = [
        'bash', os.path.join(args.code_path, 'omniinfer/omni/accelerators/sched/global_proxy/global_proxy.sh'),
        '--listen-port', args.https_port,          # Proxy listening port
        '--prefill-servers-list', intf['ip'] + ':' + args.service_port,  # Prefill server endpoints
        '--decode-servers-list', intf['ip'] + ':' + str(int(args.service_port) + 100),    # Decoder server endpoints
    ]

    subprocess.run(cmd, env=env)

def kill_all_processes():
    """Terminate all related processes"""
    # Kill processes by pattern matching
    subprocess.run("kill -9 $(ps aux | grep 'start_decode.sh' | awk '{print $2}')", shell=True)
    subprocess.run("kill -9 $(ps aux | grep 'start_prefill.sh' | awk '{print $2}')", shell=True)
    subprocess.run("kill -9 $(ps aux | grep 'run_benchmark.sh' | awk '{print $2}')", shell=True)
    subprocess.run("kill -9 $(ps aux | grep 'python' | awk '{print $2}')", shell=True)
    subprocess.run("kill -9 $(ps aux | grep 'python3.11' | awk '{print $2}')", shell=True)
    subprocess.run("kill -9 $(ps aux | grep 'python3.10' | awk '{print $2}')", shell=True)
    subprocess.run("kill -9 $(ps aux | grep 'start_do_16.sh' | awk '{print $2}')", shell=True)

def pd_ranktable(intf, args):
    """Generate ranktable configuration for distributed training"""
    ip = intf['ip']
    
    # Prefill ranktable generation
    # target_path_perfill = os.path.join(args.code_path, 'omniinfer/tools/scripts/perfill-ranktable/')
    # if not os.path.exists(target_path_perfill):
    #     print(f'Path {target_path_perfill} does not exist, creating it...')
    cmd_p = [
        'python', os.path.join(args.code_path, 'omniinfer/tools/scripts/pd_ranktable_tools.py'),
        '--mode', 'gen',
        '--prefill-server-list', args.prefill_server_list,  # NPU IDs for prefill
        '--api-server',             # API server flag
        '--save-dir', './perfill-ranktable',
    ]
    subprocess.run(cmd_p)
    # else:
    #     print(f'Path {target_path_perfill} already exists, skipping creation...')

    # Decoder ranktable generation
    # target_path_decode = os.path.join(args.code_path,'omniinfer/tools/scripts/decode-ranktable/')
    # if not os.path.exists(target_path_decode):
    #     print(f'Path {target_path_decode} does not exist, creating it...')
    cmd_d = [
        'python', os.path.join(args.code_path, 'omniinfer/tools/scripts/pd_ranktable_tools.py'),
        '--mode', 'gen',
        '--decode-server-list', args.decode_server_list,  # NPU IDs for decoder
        '--save-dir', './decode-ranktable',
    ]
    subprocess.run(cmd_d)
    # else:
    #     print(f'Path {target_path_decode} already exists, skipping creation...')

    # Global ranktable merge
    # target_path_global = os.path.join(args.code_path, 'omniinfer/tools/scripts/global_path/')
    # if not os.path.exists(target_path_global):
    #     print(f'Path {target_path_global} does not exist, creating it...')

    prefill_server_list_list = args.prefill_server_list.split(',')
    prefill_rank_table_suffix = ''.join(prefill_server_list_list)

    deocde_server_list_list = args.decode_server_list.split(',')
    decode_rank_table_suffix = ''.join(deocde_server_list_list)

    cmd_global = [
        'python', os.path.join(args.code_path, 'omniinfer/tools/scripts/pd_ranktable_tools.py'),
        '--mode', 'merge-all',  # Merge all ranktables
        '--api-server-list', f'perfill-ranktable/local_ranktable_{ip}_host.json',
        '--prefill-server-list', f'perfill-ranktable/local_ranktable_{ip}_{prefill_rank_table_suffix}.json',
        '--decode-server-list', f'decode-ranktable/local_ranktable_{ip}_{decode_rank_table_suffix}.json',
        '--save-dir', 'global_path'
    ]
    subprocess.run(cmd_global)
    # else:
    #     print(f'Path {target_path_global} already exists, skipping creation...')

def run_pd_separate_mode(args):
    """Run pipeline parallel (prefill/decoder separate) mode"""

    if (args.network_interface is not None and args.host_ip is None) or \
        (args.network_interface is None and args.host_ip is not None):
        warnings.warn(
            "For best results, please specify both --network-interface AND --host-ip "
            "together. Falling back to auto-detection for missing values.",
            RuntimeWarning
        )
    # Get network interface
    if args.network_interface:
        intf = {'if_name': args.network_interface, 'ip': get_ip_address(args.network_interface)}
    else:
        intf = get_network_interfaces()
        if not intf:
            raise RuntimeError("No network interface found and none specified")

    # Override IP if host-ip was specified
    if args.host_ip:
        intf['ip'] = args.host_ip

    # Setup distributed training configuration
    pd_ranktable(intf, args)

    # Start both server types
    start_perfill_api_servers(intf, args)
    time.sleep(1)  # Brief pause for servers to initialize
    start_decoder_api_servers(intf, args)

    time.sleep(2)  # Brief pause for servers to initialize

    # User control loop
    while True:
        user_input = input("\nEnter 'yes' to start global proxy, 'q' to quit: ").strip().lower()
        
        if user_input == 'yes' or user_input == 'y' or user_input == 'Y' or user_input == 'YES':
            start_global_proxy(intf, args)  # Start proxy after servers
            print("Global proxy started successfully!")
            break
        elif user_input == 'q' or user_input == 'Q' or user_input == 'quit' or user_input == 'exit':
            kill_all_processes()  # Cleanup before exit
            print("All processes terminated. Exiting program.")
            break
        else:
            print("Invalid input. Please enter 'yes' to proceed or 'q' to quit.")
           
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="OmniInfer Deployment Script - Launches model servers in different deployment modes"
        )
    parser.add_argument('--model-path', type=str, required=True,
                        help="Absolute path to the model checkpoint directory (required)")
    parser.add_argument('--deploy-mode', type=str, default='default',
                        help="Deployment strategy: 'default'  or 'pd_separate'  (default: default)")
    parser.add_argument('--graph-true', type=str, default='false',
                        help="Enable graph optimization mode: 'true' for optimized execution, 'false' for standard mode (default: false)")
    
    parser.add_argument('--network-interface', type=str, default=None,
                        help="Network interface name for distributed communication (default: auto-detect)")
    parser.add_argument('--host-ip', type=str, default=None,
                        help="Local machine's IP address for service binding (default: auto-detect from network interface)")
    parser.add_argument('--model-name', type=str, default='default_model',
                        help="Model identifier used for API endpoints (default: default_model)")
    parser.add_argument('--max-model-len', type=str, default='65536',
                        help="Maximum context length supported by the model in tokens (default: 65536)")
    parser.add_argument('--max-num-batched-tokens', type=str, default='32768',
                        help="Maximum context length supported by the model in tokens (default: 32768)")
    parser.add_argument('--max-num-seqs', type=str, default='64',
                        help="Maximum number of sequences supported by the model in tokens (default: 64)")
    parser.add_argument('--log-path', type=str, default='./apiserverlog',
                        help="Directory path for storing service logs (default: ./apiserverlog)")

    parser.add_argument('--server-list', type=str, default='0,1,2,3,4,5,6,7',
                        help="default mode: NPU device IDs for parallel processing (default: 0-7)")
    parser.add_argument('--prefill-server-list', type=str, default='0,1,2,3,4,5,6,7',
                        help="pd-separated:NPU device IDs dedicated to prompt prefill processing (default: '0,1,2,3,4,5,6,7')")
    parser.add_argument('--decode-server-list', type=str, default='8,9,10,11,12,13,14,15',
                        help="pd-separated:NPU device IDs dedicated to token decoding processing (default: '8,9,10,11,12,13,14,15')")
    
    parser.add_argument('--service-port', type=str, default='6660',
                        help="- In 'pd' mode: Prefill service port (Decoder uses this port + offset)\n"
                            "Global proxy will connect to these ports (default: 6660)")
    parser.add_argument('--master-port', type=str, default='8888',
                        help="The --master-port parameter in your command specifies the central coordination port used" \
                            " for distributed communication between different components of the inference system.")
    parser.add_argument('--https-port', type=str, default='8001',
                        help="Port for accepting HTTPS requests (default: 8001)")

    parser.add_argument('--additional-config', type=str, default=None,
                        help="JSON format advanced config, e.g. '{\"enable_graph_mode\":true}'")


    args = parser.parse_args()
    args.code_path = get_path_before_omniinfer()  # Get base path before 'omniinfer'
    
    # Validate critical paths
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")
     # Validate execution mode
    if args.deploy_mode not in ['pd_separate', 'default']:
        raise ValueError(f"Invalid operations mode: {args.deploy_mode}")

    # Deployment mode routing
    if args.deploy_mode == 'default':
        run_default_mode(args)
    elif args.deploy_mode == 'pd_separate':
        run_pd_separate_mode(args)

    
