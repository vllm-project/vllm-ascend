#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Generate endpoint configuration files from HCCL topology, rootinfo, and route.conf.

Input files (supports POD and SERVER topologies):
- hccl_rootinfo.json (contains topo_file_path reference)
- atlas_xxx.json (topology file)
- route.conf (CPU-device pair EID mappings)

Output files: ub_endpoint_npu_*.json (one file per NPU device_id)

CLI Arguments:
- --local, -l: Use local testing paths (default, supports --pod or --server modes)
- --pod, -p: POD mode: 1D PoD topology
- --server, -s: Server mode: 0+8 server topology or 2+4 server topology
- --dry-run, -n: Parse files but do not write output
- --rootinfo-path: Path to hccl_rootinfo.json file (only valid with --local)
- --topo-path: Path to topology JSON file (only valid with --local)
- --route-path: Path to route.conf file (only valid with --local)

Example topologies:
- POD mode: 1D PoD topology with devices 8-15
- SERVER mode: 0+8 server topology with devices 0-7

Example usage:
python generate_endpoint_configs.py --local --pod
python generate_endpoint_configs.py --local --server
python generate_endpoint_configs.py --local --pod --rootinfo-path pod06_cpu5/hccl_rootinfo.json
     --topo-path pod06_cpu5/atlas_950_1.json --route-path pod06_cpu5/route.conf
python generate_endpoint_configs.py --local --server --rootinfo-path server/hccl_rootinfo_08server.json
     --topo-path server/atlas_850_1.json --route-path server/route.conf
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_route_conf(route_conf_path: Path) -> Dict[int, Dict[str, str]]:
    """
    Parse route.conf file to extract device-to-EID mappings.

    Returns: {device_id: {'local_eid': '...', 'remote_eid': '...'}}
    """
    pairs = {}
    current_device_id = None
    current_pair = {}

    with open(route_conf_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if '_dev_id=' in line:
                # Extract device_id: pairX_dev_id=32 -> device_id=32
                parts = line.split('=')
                current_device_id = int(parts[1])
                current_pair = {'dev_id': current_device_id, 'local_eid': None, 'remote_eid': None}
                pairs[current_device_id] = current_pair
                print(f"current_device_local_id: {current_device_id}")
            elif '_local_eid=' in line:
                # Extract local EID: pairX_chan0_local_eid=0x...
                eid = line.split('=')[1].strip().replace('0x', '')
                if current_device_id is not None:
                    current_pair['local_eid'] = eid
            elif '_remote_eid=' in line:
                # Extract remote EID: pairX_chan0_remote_eid=0x...
                eid = line.split('=')[1].strip().replace('0x', '')
                if current_device_id is not None:
                    current_pair['remote_eid'] = eid

    return pairs


def build_eid_to_device_map(hccl_rootinfo: Dict) -> Dict[str, int]:
    """
    Build a mapping from EID to device_id using hccl_rootinfo.

    Returns: {eid: device_id}
    """
    eid_to_device = {}
    for rank in hccl_rootinfo.get('rank_list', []):
        device_id = rank['device_id']
        for level in rank.get('level_list', []):
            for addr in level.get('rank_addr_list', []):
                eid = addr['addr']
                eid_to_device[eid] = device_id
    return eid_to_device


def build_device_id_to_local_id_map(hccl_rootinfo: Dict) -> Dict[int, int]:
    """
    Build mapping from device_id to local_id using hccl_rootinfo.

    Returns: {device_id: local_id}
    """
    return {rank['device_id']: rank['local_id'] for rank in hccl_rootinfo.get('rank_list', [])}


def build_local_id_to_device_id_map(hccl_rootinfo: Dict) -> Dict[int, int]:
    """
    Build mapping from local_id to device_id using hccl_rootinfo.

    Returns: {local_id: device_id}
    """
    return {rank['local_id']: rank['device_id'] for rank in hccl_rootinfo.get('rank_list', [])}


def find_peer_eid_from_1dmesh(
    local_id: int,
    device_local_id: int,
    device_eid: str,
    device_eid_ports: List[str],
    topo_data: Dict,
    rootinfo: Dict
) -> str:
    """
    Find the connected peer's EID for a given device from 1DM mesh topology.

    For PEER2PEER (direct connections) in net_layer 0.

    Returns: peer EID string, or empty string if not found.
    """
    # Get all 1DM mesh PEER2PEER edges
    p2p_edges = [
        edge for edge in topo_data.get('edge_list', [])
        if edge.get('topo_type') == '1DMESH'
        and edge.get('link_type') == 'PEER2PEER'
        and edge.get('net_layer') == 0
    ]

    # Build EID+port to device_id mapping from rootinfo
    eid_port_to_device = {}
    for rank in rootinfo.get('rank_list', []):
        dev_id = rank['device_id']
        for level in rank.get('level_list', []):
            if level.get('net_layer') != 0:
                continue
            for addr in level.get('rank_addr_list', []):
                eid = addr['addr']
                ports = addr.get('ports', [])
                for port in ports:
                    eid_port_to_device[(eid, port)] = dev_id

    # Use local_id for topology edge matching (topo file uses local_id, not device_id)
    topo_my_index = local_id


    # Find edges involving our device and identify the peer
    for edge in p2p_edges:
        local_a = edge['local_a']
        local_b = edge['local_b']
        local_a_ports = edge.get('local_a_ports', [])
        local_b_ports = edge.get('local_b_ports', [])

        # Determine which end is our device and which is the peer
        if local_a == topo_my_index:
            my_ports = local_a_ports
            peer_ports = local_b_ports
            peer_topo_index = local_b
        elif local_b == topo_my_index:
            my_ports = local_b_ports
            peer_ports = local_a_ports
            peer_topo_index = local_a
        else:
            continue

        # Find intersection of our EID's ports and the connected ports
        for my_port in device_eid_ports:
            if my_port in my_ports:
                # Find which peer device connects through this port
                # The peer port is the corresponding one on the other side
                # For ring topology there's typically a 1:1 port mapping
                for peer_port in peer_ports:
                    # Look up EID for the peer device using local_id (peer_topo_index)
                    # Topology uses local_id, so we need to find the rank where local_id matches
                    for rank in rootinfo.get('rank_list', []):
                        if rank['local_id'] != peer_topo_index:
                            continue
                        for level in rank.get('level_list', []):
                            if level.get('net_layer') != 0:
                                continue
                            for addr in level.get('rank_addr_list', []):
                                if peer_port in addr.get('ports', []):
                                    return addr['addr']

    return ""


def get_protocol_from_eid(
    eid: str,
    net_layer: int,
    topo_data: Dict,
    device_local_id: int,
    device_eid_ports: List[str]
) -> str:
    """
    Determine protocol for an EID based on topology and net layer.

    Returns: 'ub_ctp' or 'ub_tp'.

    todo: 'roce', 'uboe'
    """
    # For net_layer 0 (1DMESH), always use ub_ctp
    if net_layer == 0:
        return 'ub_ctp'

    # For CLOS layer (net_layer 1+), check topology
    if net_layer >= 1:
        for edge in topo_data.get('edge_list', []):
            if edge.get('topo_type') == 'CLOS' and edge.get('net_layer') == net_layer:
                protocols = edge.get('protocols', [])
                # Check if any of the device's EID ports match this edge's ports
                edge_a_ports = edge.get('local_a_ports', [])
                edge_b_ports = edge.get('local_b_ports', [])
                all_edge_ports = set(edge_a_ports) | set(edge_b_ports)
                device_port_set = set(device_eid_ports)

                # If there's any port overlap, this edge applies to this device
                if all_edge_ports & device_port_set:
                    if 'UB_CTP' in protocols:
                        return 'ub_ctp'
                    elif 'UB_TP' in protocols:
                        return 'ub_tp'

    return 'ub_ctp'


def get_h2d_plane_id(device_id: int, rootinfo: Dict, mode: str) -> str:
    """
    Find the host endpoint and return its plane_id.

    POD mode: H2D has 6 ports (1DPoD topology)
    SERVER mode: H2D has 8 ports (0+8 server topology) or 4 ports (2+4 topo)

    Args:
        device_id: The NPU device ID (e.g., 32, 33, 34, etc.)
        rootinfo: Parsed hccl_rootinfo.json data
        mode: Topology mode ('pod' or 'server')

    Returns:
        The plane_id from the H2D endpoint, or 'plane_x' if not found
    """
    port_count = 4 if mode == 'server' else 6
    for rank in rootinfo.get('rank_list', []):
        if rank['device_id'] == device_id:
            for level in rank.get('level_list', []):
                if level.get('net_layer') == 1:  # CLOS layer
                    for addr_entry in level.get('rank_addr_list', []):
                        if len(addr_entry.get('ports', [])) >= port_count:
                            return addr_entry.get('plane_id', 'plane_x')
    return 'plane_x'


def generate_endpoint_list(
    local_id: int,
    device_info: Dict,
    topo_data: Dict,
    rootinfo: Dict
) -> List[Dict]:
    """
    Generate endpoint list for a single NPU device.

    Args:
        local_id: The logical device index (used for topology edge lookups)
        device_info: The device info from hccl_rootinfo (contains device_id, local_id, level_list)
        topo_data: Topology data from atlas_*.json
        rootinfo: HCCL rootinfo data

    Returns:
        List of endpoint configurations
    """
    endpoint_list = []
    device_local_id = device_info['local_id']
    seen_eids = set()  # Track unique EIDs

    # Iterate through all layers in level_list
    for level in device_info['level_list']:
        net_layer = level['net_layer']
        net_type = level['net_type']

        for addr_entry in level['rank_addr_list']:
            eid = addr_entry['addr']
            device_eid_ports = addr_entry.get('ports', [])
            plane_id = addr_entry.get('plane_id', 'plane_0')

            # Skip duplicate EIDs
            if eid in seen_eids:
                continue
            seen_eids.add(eid)

            # Determine protocol
            protocol = get_protocol_from_eid(eid, net_layer, topo_data, device_local_id, device_eid_ports)

            # Create endpoint
            endpoint = {
                "protocol": protocol,
                "comm_id": eid,
                "placement": "device"
            }

            # Add plane field for ub_tp protocol or for net_layer >= 1 (CLOS/P2N)
            # P2P (net_layer 0) direct connections should NOT have plane field
            if (net_layer >= 1 or protocol == 'ub_tp') and plane_id:
                endpoint["plane"] = plane_id

            # Add dst_eid for 1DMESH (net_layer 0) direct connections
            if net_layer == 0:
                peer_eid = find_peer_eid_from_1dmesh(
                    local_id, device_local_id, eid, device_eid_ports, topo_data, rootinfo
                )
                if peer_eid:
                    endpoint["dst_eid"] = peer_eid

            endpoint_list.append(endpoint)

    return endpoint_list


if __name__ == "__main__":
    """Main entry point for endpoint config generation."""
    parser = argparse.ArgumentParser(
        description="Generate NPU endpoint configuration files from HCCL topology and route.conf."
    )
    parser.add_argument("--local", "-l", action="store_true", default=False,
                       help="Use local testing paths (supports --pod or --server modes)")
    parser.add_argument("--pod", "-p", action="store_true",
                       help="POD mode: 1D PoD topology")
    parser.add_argument("--server", "-s", action="store_true",
                       help="Server mode: 0+8 server topology")
    parser.add_argument("--dry-run", "-n", action="store_true",
                       help="Parse files but do not write output")
    parser.add_argument("--rootinfo-path", type=str, default=None,
                       help="Path to hccl_rootinfo.json file (only valid with --local)")
    parser.add_argument("--topo-path", type=str, default=None,
                       help="Path to topology JSON file (only valid with --local)")
    parser.add_argument("--route-path", type=str, default=None,
                       help="Path to route.conf file (only valid with --local)")

    args = parser.parse_args()

    # Determine mode: pod or server (server takes priority if both specified)
    use_local = args.local  # local testing mode or production mode
    mode = 'server' if args.server else 'pod' if args.pod else 'pod'

    # Set default paths for local mode
    if use_local and not args.rootinfo_path:
        if mode == 'pod':
            args.rootinfo_path = "./pod/hccl_rootinfo.json"
            args.topo_path = "./pod/atlas_950_1.json"
            args.route_path = "./pod/route.conf"
        else:  # server mode
            args.rootinfo_path = "./server/hccl_rootinfo_08server.json"
            args.topo_path = "./server/atlas_850_1.json"
            args.route_path = "./server/route.conf"

    mode_str = f"local {mode}" if use_local else "production"
    print(f"Running in {mode_str} mode")
    
    # Production mode: use /etc and /lib paths (same for both modes)
    if not use_local:
        args.rootinfo_path = "/etc/hccl_rootinfo.json"
        args.route_path = "/lib/route.conf"

    if args.dry_run:
        print("Dry run mode: parsing only, no output files will be written")

    # Parse route.conf
    print(f"Loading: {args.route_path}")
    route_pairs = parse_route_conf(args.route_path)
    print(f"Found {len(route_pairs)} device pairs (local_id: {sorted(route_pairs.keys())})")

    # Load hccl_rootinfo.json
    print(f"Loading: {args.rootinfo_path}")
    with open(args.rootinfo_path) as f:
        hccl_rootinfo = json.load(f)

    if not use_local:
        args.topo_path = Path(hccl_rootinfo['topo_file_path'])

    # Load topology file
    print(f"Loading topology: {args.topo_path}")
    with open(args.topo_path) as f:
        topo_data = json.load(f)

    # Build local_id to device_id mapping (route.conf uses local_id, hccl_rootinfo uses device_id)
    local_id_to_device_id = build_local_id_to_device_id_map(hccl_rootinfo)
    print(f"Built local_id to device_id mapping: {local_id_to_device_id}")

    # Generate endpoint files for each device
    print(f"Generating endpoints for {len(route_pairs)} NPUs...")
    for local_id, route_pair_info in route_pairs.items():
        # route.conf uses local_id (named dev_id), convert to device_id for hccl_rootinfo lookup
        device_id = local_id_to_device_id.get(local_id)
        if device_id is None:
            print(f"Warning: device_id not found for local_id {local_id}")
            continue

        # Find device info in hccl_rootinfo by device_id
        device_info = None
        for rank in hccl_rootinfo.get('rank_list', []):
            if rank['device_id'] == device_id:
                device_info = rank
                break

        if not device_info:
            print(f"Warning: device_id {device_id} not found in hccl_rootinfo")
            continue

        # Generate endpoint list using local_id for topology edge lookups
        endpoint_list = generate_endpoint_list(
            local_id, device_info, topo_data, hccl_rootinfo
        )

        # Add CPU host endpoint from route.conf
        if 'local_eid' in route_pair_info:
            h2d_plane_id = get_h2d_plane_id(device_id, hccl_rootinfo, mode)
            h2d_device_eid = route_pair_info['remote_eid']
            if h2d_plane_id == 'plane_x':
                print(f"Warning: host plane not found in hccl_rootinfo")
                ub_host_endpoint = {
                    "protocol": "ub_ctp",
                    "comm_id": route_pair_info['local_eid'],
                    "placement": "host",
                    "dst_eid": "None"
                }
            else:
                ub_host_endpoint = {
                    "protocol": "ub_ctp",
                    "comm_id": route_pair_info['local_eid'],
                    "placement": "host",
                    "plane": h2d_plane_id
                }
            ub_host_endpoint_d = {
                "protocol": "ub_ctp",
                "comm_id": route_pair_info['local_eid'],
                "placement": "host",
                "dst_eid": h2d_device_eid
            }
            # Insert host endpoint at the end
            # endpoint_list.append(ub_host_endpoint_d)
            # endpoint_list.append(ub_host_endpoint)

        net_instance_id = next(
            (item.get('net_instance_id')
            for item in device_info.get('level_list', [])
            if item.get('net_layer') == 1),
            None  # 默认值：没找到就返回 None
        )
        output = {
            "version": "1.3",
            "net_instance_id": net_instance_id,
            "endpoint_list": endpoint_list
        }

        if use_local:
            output_path = Path(f"./hixlep/ub_endpoint_npu_{local_id}.json")
        else:
            output_path = Path(f"/etc/hixlep/ub_endpoint_npu_{local_id}.json")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not args.dry_run:
            with open(output_path, "w", encoding='utf-8') as f:
                json.dump(output, f, indent=2)
            print(f"Generated: {output_path}")
        else:
            print(f"[Dry run] Would generate: {output_path}")

    print(f"{'Dry run: Would generate' if args.dry_run else 'Generated'} {len(route_pairs)} endpoint configuration files.")
