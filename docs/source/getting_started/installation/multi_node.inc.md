### 5.2 Multi-node Deployment {: #multi-node-deployment }

Before a multi-node deployment, verify:

1. physical links;
2. the NPU network on each node;
3. communication between nodes;
4. consistent containers and software stacks across all nodes.

#### 5.2.1 Physical Layer Requirements

- The physical hosts are on mutually reachable networks;
- optical modules and network links between NPUs are working;
- switch, NIC, and NPU network configurations match the deployment topology.

#### 5.2.2 Checks on Each Node

Example for Atlas A2 / a typical node with eight NPU indices:

```bash
for i in {0..7}; do
    hccn_tool -i "$i" -lldp -g | grep Ifname
done

for i in {0..7}; do
    hccn_tool -i "$i" -link -g
done

for i in {0..7}; do
    hccn_tool -i "$i" -net_health -g
done

for i in {0..7}; do
    hccn_tool -i "$i" -netdetect -g
done

for i in {0..7}; do
    hccn_tool -i "$i" -gateway -g
done

cat /etc/hccn.conf
```

??? note "Atlas A3"

    If the current topology exposes 16 NPU indices, replace `{0..7}` with `{0..15}`.

#### 5.2.3 Obtain NPU IP Addresses

```bash
for i in {0..7}; do
    hccn_tool -i "$i" -ip -g | grep ipaddr
done
```

For an Atlas A3 topology with 16 indices, use `{0..15}`.

#### 5.2.4 Ping Across Nodes

Run the following command on the source node:

```bash
hccn_tool -i 0 -ping -g address <peer-npu-ip>
```

Verify every NPU link that will participate in communication.

#### 5.2.5 Additional Ascend 950DT Pre-checks

Before deploying the Ascend 950 series, confirm that the following files or directories exist and match the actual UB topology:

```text
/lib/route.conf
/etc/hccl_rootinfo.json
/etc/hixlep/
```

If the contents are missing or do not match the actual topology, generate and validate them first according to the corresponding HiXLEP configuration documentation.

#### 5.2.6 Software Environment on Each Node

All nodes must use the same:

- vLLM Ascend release;
- vLLM version;
- CANN/TorchNPU versions;
- model weights;
- container image;
- runtime configuration.

Continue to follow the hardware- and model-specific deployment documentation for container device mappings.

Multi-node model serving commands are outside the scope of Installation. After completing the environment checks, proceed to the corresponding [Feature Tutorials](../tutorials/features/index.md) or model tutorial.
