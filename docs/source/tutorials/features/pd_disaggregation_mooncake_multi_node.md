# Prefill-Decode Disaggregation (DeepSeek)

## Getting Started

vLLM-Ascend now supports prefill-decode (PD) disaggregation with EP (Expert Parallelism) options. This guide takes one-by-one steps to verify these features with constrained resources.

Take the DeepSeek-r1-w8a8 model as an example, use 4 Atlas 800T A3 servers to deploy the "2P1D" architecture. Assume the IP of the prefiller server is 192.0.0.1 (prefill 1) and 192.0.0.2 (prefill 2), and the decoder servers are 192.0.0.3 (decoder 1) and 192.0.0.4 (decoder 2). On each server, use 8 NPUs and 16 chips to deploy one service instance.

## Verify Multi-Node Communication Environment

### Physical Layer Requirements

- The physical machines must be located on the same LAN, with network connectivity.
- All NPUs must be interconnected. On A2/A3, intra-node connectivity is via HCCS, and inter-node connectivity is via RDMA. For Atlas 850/850E/950 SuperPod (950PR/950DT), see [Atlas 850/850E/950 SuperPod Communication Configuration](#atlas-850850e950-superpod-communication-configuration).

### Verification Process

The following checks apply to A2/A3 deployments.

Execute the following commands on each node in sequence. The results must all be `success` and the status must be `UP`:

=== "A3"

    1. Single Node Verification:

        Execute the following commands on each node in sequence. The results must all be `success` and the status must be `UP`:

        ```bash
        # Check the remote switch ports
        for i in {0..15}; do hccn_tool -i $i -lldp -g | grep Ifname; done
        # Get the link status of the Ethernet ports (UP or DOWN)
        for i in {0..15}; do hccn_tool -i $i -link -g ; done
        # Check the network health status
        for i in {0..15}; do hccn_tool -i $i -net_health -g ; done
        # View the network detected IP configuration
        for i in {0..15}; do hccn_tool -i $i -netdetect -g ; done
        # View gateway configuration
        for i in {0..15}; do hccn_tool -i $i -gateway -g ; done
        ```

    2. Check NPU HCCN Configuration:

        Ensure that the hccn.conf file exists in the environment. If using Docker, mount it into the container.

        ```bash
        cat /etc/hccn.conf
        ```

    3. Get NPU IP Addresses

        ```bash
        # Get virtual NPU IP.
        for i in {0..15}; do hccn_tool -i $i -vnic -g;done
        ```

    4. Get superpodid and SDID

        ```bash
        for i in {0..15}; do npu-smi info -t spod-info -i $i -c 0;npu-smi info -t spod-info -i $i -c 1;done
        ```

    5. Cross-Node PING Test

        ```bash
        # Execute on the target node (replace 'x.x.x.x' with virtual NPU IP address).
        for i in {0..15}; do hccn_tool -i $i -hccs_ping -g address x.x.x.x;done
        ```

    6. Check NPU TLS Configuration

        ```bash
        # The TLS settings should be consistent across all nodes
        for i in {0..15}; do hccn_tool -i $i -tls -g ; done | grep switch
        ```

=== "A2"

    1. Single Node Verification:

        Execute the following commands on each node in sequence. The results must all be `success` and the status must be `UP`:

        ```bash
        # Check the remote switch ports
        for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done
        # Get the link status of the Ethernet ports (UP or DOWN)
        for i in {0..7}; do hccn_tool -i $i -link -g ; done
        # Check the network health status
        for i in {0..7}; do hccn_tool -i $i -net_health -g ; done
        # View the network detected IP configuration
        for i in {0..7}; do hccn_tool -i $i -netdetect -g ; done
        # View gateway configuration
        for i in {0..7}; do hccn_tool -i $i -gateway -g ; done
        ```

    2. Check NPU HCCN Configuration:

        Ensure that the hccn.conf file exists in the environment. If using Docker, mount it into the container.

        ```bash
        cat /etc/hccn.conf
        ```

    3. Get NPU IP Addresses

        ```bash
        for i in {0..7}; do hccn_tool -i $i -ip -g;done
        ```

    4. Cross-Node PING Test

        ```bash
        # Execute on the target node (replace 'x.x.x.x' with actual npu ip address)
        for i in {0..7}; do hccn_tool -i $i -ping -g address x.x.x.x;done
        ```

    5. Check NPU TLS Configuration

        ```bash
        # The TLS settings should be consistent across all nodes
        for i in {0..7}; do hccn_tool -i $i -tls -g ; done | grep switch
        ```

### Atlas 850/850E/950 SuperPod Communication Configuration

For Atlas 850/850E/950 SuperPod (950PR/950DT), PD disaggregation uses the super plane (based on UB port and UBC protocol) by default. No additional network configuration is required for this default path.

Automatic UB configuration requires HIXL 9.1.0 or later. Upgrade HIXL before using the configuration below. Set the following environment variable in both the prefiller and decoder serving environments to generate the communication configuration automatically:

```bash
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
```

The parameter plane supports UBG, UBoE, and RoCE network protocols. Each protocol requires compatible hardware and network configuration completed in advance. Consult your environment provider or administrator for the required environment and configuration.

For UBG or UBoE network protocols, keep the automatic configuration setting above and select the corresponding protocol before starting the serving processes:

=== "UBG"

    ```bash
    export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["ub_rtp:device"]}'
    ```

=== "UBoE"

    ```bash
    export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'
    ```

=== "RoCE"

    Host RoCE parameter-plane configuration for Atlas 850/850E/950 SuperPod (950PR/950DT) currently requires manual endpoint files and checking the mapping between physical NPUs and host RoCE IP addresses. Prepare the endpoint configuration files under `/etc/hixlep` and set `ascend_local_comm_res_path` in `kv_connector_extra_config` on both sides. The directory is configurable; `/etc/hixlep` is only an example. For example, with `DP=4` and `TP=1` on each side:

    ```json
    {
        "kv_connector_extra_config": {
            "prefill": {
                "dp_size": 4,
                "tp_size": 1
            },
            "decode": {
                "dp_size": 4,
                "tp_size": 1
            },
            "ascend_local_comm_res_path": "/etc/hixlep"
        }
    }
    ```

    Merge these fields into your existing `--kv-transfer-config` and adjust the parallelism settings to your deployment.

    When combining PD disaggregation with KV pooling through `MultiConnector`, place `ascend_local_comm_res_path` in the **outer** `MultiConnector`'s `kv_connector_extra_config`, alongside the `connectors` list. Do not place it in a child connector's `kv_connector_extra_config`: the worker loads the endpoint files from the outer KV transfer configuration. Keep the `prefill` and `decode` parallelism settings in the Mooncake P/D child connector's configuration.

    Create an endpoint file for each physical NPU used on the host, named `/etc/hixlep/ub_endpoint_npu_<physical_npu_id>.json`. The suffix is the host physical NPU ID, not the DP rank or the ordinal within the visible devices.

    Check the host's NPU-to-RoCE-IP mapping before filling in these files. For a host whose RoCE IP addresses correspond to physical NPU IDs in ascending order, populate `ub_endpoint_npu_0.json`, `ub_endpoint_npu_1.json`, and so on in that order. **RoCE IPs can be populated in ascending numerical order**, after verifying the actual host mapping. Each file's `comm_id` must match the RoCE IP associated with that physical NPU; do not use the DP rank or visible-device order to select the address. The JSON content of each endpoint file follows this format:

    ```json
    {
        "version": "1.3",
        "net_instance_id": "superpod_0",
        "endpoint_list": [
            {
                "protocol": "roce",
                "comm_id": "<host_roce_ip>",
                "placement": "host"
            }
        ]
    }
    ```

    Replace `<host_roce_ip>` with the actual host RoCE IP address. Ensure the endpoint files are available at the configured path inside each container.

    !!! warning "Simplified Host RoCE configuration is NOT YET SUPPORTED"

        A future configuration is intended to select Host RoCE with only the following environment variable:

        ```bash
        # NOT YET SUPPORTED: do not use this in current deployments.
        export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":"roce:host"}'
        ```

        **This configuration is not yet supported and does not replace the manual endpoint files above.** Continue to configure the per-NPU endpoint files and `ascend_local_comm_res_path` for current deployments.

## Run with Docker

Start a Docker container on each node. The following example maps the 16 device nodes used by the A3 deployment above. For Atlas 850/850E/950 SuperPod (950PR/950DT), use the complete container command below.

```bash
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
export NAME=vllm-ascend

# Run the container using the defined variables
# Note: If you are running bridge network with Docker, please expose available ports for multiple nodes communication in advance.
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci8 \
--device /dev/davinci9 \
--device /dev/davinci10 \
--device /dev/davinci11 \
--device /dev/davinci12 \
--device /dev/davinci13 \
--device /dev/davinci14 \
--device /dev/davinci15 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /mnt/sfs_turbo/.cache:/root/.cache \
-it $IMAGE bash
```

### Atlas 850/850E/950 SuperPod (950PR/950DT) Container Requirements

For an eight-device 950PR/950DT host, use the following command. Choose an image compatible with the host driver; for openEuler, use the `{{ vllm_ascend_version }}-a5-openeuler` tag.

```bash
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a5
export NAME=vllm-ascend-950

docker run --rm \
    --name "$NAME" \
    --net=host \
    --privileged \
    --shm-size=1g \
    --ulimit memlock=-1:-1 \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/hisi_hdc \
    --device /dev/ummu \
    --device /dev/uburma \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/bin/urma_admin:/usr/bin/urma_admin:ro \
    -it "$IMAGE" bash
```

The complete driver mount includes `driver/topo/950`. All host mount sources must exist and have the expected file/directory type; `-v` creates a directory for a missing source path.

For automatic UB configuration, set `ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'` in both serving environments. The `/lib/route.conf` mount depends on the CANN version:

- **CANN 9.2.0 and later:** This file is no longer a dependency; omit the mount.
- **CANN 9.1.0 and 9.1.1:** The default UB P/D configuration above uses device-only endpoints and does not require this mount. If you explicitly enable Host UB (`ub_ctp:host`), mount the file unless the driver can provide the routing information through procfs.

If needed, generate the file on the host first (see the [LocalCommRes Configuration Guide](https://gitcode.com/cann/hixl/wiki/A5%20LocalCommRes%E9%85%8D%E7%BD%AE%E6%8C%87%E5%8D%97.md)) and add `-v /lib/route.conf:/lib/route.conf:ro`. Host RoCE uses the separate endpoint files described above.

For Host RoCE, insert the following options before `-it "$IMAGE" bash` to expose the RDMA devices, provide the HRN5 shared library, and mount the endpoint directory configured by `ascend_local_comm_res_path`:

```bash
--device /dev/infiniband \
-v /usr/lib64/libhrn5-rdmav34.so:/usr/lib64/libhrn5-rdmav34.so:ro \
-v /etc/hixlep:/etc/hixlep:ro \
```

Both paths in the endpoint-directory mount are configurable: the left side is the host directory containing the endpoint JSON files, and the right side is their directory inside the container. Set `ascend_local_comm_res_path` to the **container-side** directory. For example, `-v /srv/mooncake/endpoints:/opt/mooncake/endpoints:ro` requires `"ascend_local_comm_res_path": "/opt/mooncake/endpoints"`. Keep the `ub_endpoint_npu_<physical_npu_id>.json` filenames unchanged.

- Use the provider library path/version matching the host driver and container RDMA userspace. Mount `/etc/libibverbs.d/hrn5.driver` with `-v /etc/libibverbs.d/hrn5.driver:/etc/libibverbs.d/hrn5.driver:ro` if required by that userspace; this configuration file does not replace the shared library.
- `--privileged` exposes host devices. A non-privileged Host RoCE configuration must explicitly map the devices and provide `--cap-add=NET_ADMIN`; validate it for your driver and workload.
- Mounting the `hiroce5` diagnostic tool is **optional**, for example `-v /usr/local/bin/hiroce5:/usr/local/bin/hiroce5:ro` if that is its host path.

## Install Mooncake

The vLLM-Ascend container images already include a matching Mooncake version. You can use the bundled version directly; **downloading and compiling Mooncake is optional and is only needed for specific workload requirements**. Follow the source-build steps below only if your workload requires a custom Mooncake build, and ensure that it is compatible with the image's vLLM-Ascend and CANN versions.

Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI. Installation and Compilation Guide: <https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#build-and-use-binaries>
For an optional source build, first obtain the Mooncake project:

```shell
git clone -b v0.3.9 --depth 1 https://github.com/kvcache-ai/Mooncake.git
```

(Optional) Replace go install url if the network is poor

```shell
cd Mooncake
sed -i 's|https://go.dev/dl/|https://golang.google.cn/dl/|g' dependencies.sh
```

Install mpi

```shell
apt-get install mpich libmpich-dev -y
```

Install the relevant dependencies. The installation of Go is not required.

```shell
bash dependencies.sh -y
```

Compile and install

```shell
mkdir build
cd build
cmake .. -DUSE_ASCEND_DIRECT=ON
make -j
make install
```

Set environment variables

**Note:**

- Adjust the Python path according to your specific Python installation
- Ensure `/usr/local/lib` and `/usr/local/lib64` are in your `LD_LIBRARY_PATH`

```shell
export LD_LIBRARY_PATH=/usr/local/lib64/python3.12/site-packages/mooncake:$LD_LIBRARY_PATH
```

## Prefiller/Decoder Deployment

!!! note "Timeout for long sequences or high concurrency"

    For long sequences or high concurrency, the default KV-cache release timeout of 480 seconds may be insufficient. Increase it as needed in the prefiller startup script. For versions that use `VLLM_MOONCAKE_CONNECTOR_TIMEOUT` for this timeout, add:

    ```shell
    export VLLM_MOONCAKE_CONNECTOR_TIMEOUT=1800
    ```

    The value is in seconds: `1800` allows 30 minutes before timeout. In current versions, this setting is named `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`; use `export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=1800` instead. Check the variable name supported by your installed version.

    Increasing this timeout generally does not affect inference accuracy or performance during normal operation. If a decoder (D) node fails while requests still have KV caches on the prefiller (P) node that the decoder has not received, a longer timeout delays releasing those caches. They continue to occupy HBM, reducing the memory available for other requests on the P node until they are released.

We can run the following scripts to launch a server on the prefiller/decoder node, respectively. Please note that each P/D node will occupy ports ranging from kv_port to kv_port + num_chips to initialize socket listeners. To avoid any issues, port conflicts should be prevented.

### QoS Configuration

Set `qos_priority` in the transfer initiator's `kv_connector_extra_config`.
This option is supported by `MooncakeConnectorV1`, `MooncakeHybridConnector`,
and `MooncakeLayerwiseConnector`. The default P/D QoS is **1**; accepted values
are integers in **[0, 4]** (booleans are not accepted).

With the HIXL Client/Server transport, the endpoint initiating the connection
supplies the channel QoS. The peer creates its corresponding channel using
that received value:

| Transfer mode | Initiator | Operation | QoS configuration used |
| :--- | :--- | :--- | :--- |
| Pull | Decoder (D) | READ KV from the prefiller | D-side `qos_priority` |
| Push | Prefiller (P) | WRITE KV to the decoder | P-side `qos_priority` |

Although KV data flows from P to D in both modes, pull uses D's QoS and push
uses P's QoS. Configuring only the passive endpoint does not override the
initiator's channel QoS. Both endpoints support the option, but the effective
value for a connection comes from its initiator.

For example, add `"qos_priority": 1` alongside your existing parallelism settings:

```json
{
  "kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_consumer",
  "kv_connector_extra_config": {
    "qos_priority": 1,
    "prefill": {"dp_size": 1, "tp_size": 2},
    "decode": {"dp_size": 1, "tp_size": 2}
  }
}
```

Use `kv_producer` for the prefiller and retain the other deployment-specific
fields in your `--kv-transfer-config`.

No manual QoS environment export is required. Before initializing the transfer
engine, the connector merges its QoS into the literal top-level JSON key
`"comm_resource_config.qos"` in `ASCEND_GLOBAL_RESOURCE_CONFIG`, creating the
environment variable if absent. Other resource settings and the `store`
configuration are preserved. An explicit QoS value, or the default **1** when
omitted, replaces the existing top-level QoS. Invalid QoS values or malformed
resource JSON fail before the environment is modified.

The `Injected comm_resource_config.qos=...` log confirms that the local
environment configuration was written successfully. It does not by itself
prove that this value was used for a transfer channel. In pull mode, a passive
P-side HIXL server may not print a local QoS parsing log; in push mode, the
same applies to the passive D side. Check the initiator's configuration and
HIXL channel logs to verify the channel QoS.

With `MultiConnector`, set `qos_priority` in the Mooncake P/D child connector's
`kv_connector_extra_config`. P/D injection only updates the top-level QoS key;
it does not configure or validate pooling backends or modify their `store`
settings. Restart the serving processes after changing P/D QoS.

### kv_port Configuration Guide

On Ascend NPU, Mooncake uses AscendDirectTransport for RDMA data transfer, which randomly allocates ports within range `[20000, 20000 + npu_per_node × 1000)`. If `kv_port` overlaps with this range, intermittent port conflicts may occur. To avoid this, configure `kv_port` according to the table below:

| NPUs per Node | Reserved Port Range | Recommended kv_port |
|---------------|---------------------|---------------------|
| 8             | 20000 - 27999       | >= 28000            |
| 16            | 20000 - 35999       | >= 36000            |

!!! warning

    If you occasionally see `zmq.error.ZMQError: Address already in use` during startup, it may be caused by kv_port conflicting with randomly allocated AscendDirectTransport ports. Increase your kv_port value to avoid the reserved range.

### launch_online_dp.py

Use `launch_online_dp.py` to launch external dp vllm servers.
[launch_online_dp.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/launch_online_dp.py)

### run_dp_template.sh

Modify `run_dp_template.sh` on each node.
[run_dp_template.sh](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/run_dp_template.sh)

`launch_online_dp.py` starts `dp-size-local` vLLM instances and passes seven positional arguments to `run_dp_template.sh`. The examples below correspond to [Start the service](#start-the-service).

| Launcher option | Template value | Meaning | Example value | Constraints |
| --- | --- | --- | --- | --- |
| `--dp-size` | `$3` / `--data-parallel-size` | Total DP ranks in the group. | P: `2`; D: `32` | Positive integer; identical within the same group. Total ranks must cover `[0, dp-size)` without overlap. |
| `--tp-size` | `$7` / `--tensor-parallel-size` | NPUs used by each instance. | P: `8`; D: `1` | Positive integer; `dp-size-local * tp-size` must not exceed available NPUs on the node.|
| `--dp-size-local` | Number of template invocations | DP instances on this node. | P: `2`; D: `16` | If set to `-1`, it defaults to `dp-size`. Otherwise, `1 <= dp-size-local <= dp-size`.|
| `--dp-rank-start` | `$4` / `--data-parallel-rank` = `dp-rank-start + i` | First DP rank on this node. | P: `0`; D: `0` or `16` | Non-negative integer; the assigned rank range must be within `dp-size` and not overlap. |
| `--dp-address` | `$5` / `--data-parallel-address` | DP master address. | P: `192.0.0.1` or `192.0.0.2`; D: `192.0.0.3` | The IP address must be reachable and identical within the same DP group. |
| `--dp-rpc-port` | `$6` / `--data-parallel-rpc-port` | DP master RPC port. | `12321` | Free, reachable port; identical within a DP group. |
| `--vllm-start-port` | `$2` / `--port` = `vllm-start-port + i` | First local API port. | `7100`, then `7101`, ... | Consecutive free ports; must match the proxy configuration. |
| Computed by the launcher | `$1` / `ASCEND_RT_VISIBLE_DEVICES` | NPU IDs assigned to each instance. | P: `0,...,7`, `8,...,15`; D: `0` through `15` | IDs must exist and not overlap; non-contiguous IDs require adapting the launcher. |

!!! note

    - The launcher validates only argument types and required options. Check the remaining constraints before deployment.
    - If speculative decoding is enabled, `num_speculative_tokens` should be subject to one of the following conditions:
        1. Hybrid Mamba models (e.g., Qwen-Next and Qwen3.5 series): `num_speculative_tokens` should be equal on P nodes and D nodes.
        2. Other models: `num_speculative_tokens` on P nodes should be 1, and `num_speculative_tokens` on D nodes should be greater or equal to 1.

#### Layerwise

=== "Prefiller node 1"

    ```shell
    nic_name="eth0"  # network card name
    local_ip="192.0.0.1"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=256
    export TASK_QUEUE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE="AIV"
    export VLLM_USE_V1=1
    export ASCEND_RT_VISIBLE_DEVICES=$1
    vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --enable-expert-parallel \
      --seed 1024 \
      --served-model-name ds_r1 \
      --max-model-len 40000 \
      --max-num-batched-tokens 16384 \
      --max-num-seqs 8 \
      --enforce-eager \
      --trust-remote-code \
      --gpu-memory-utilization 0.9  \
      --quantization ascend \
      --no-enable-prefix-caching \
      --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
      --additional-config '{"enable_shared_expert_dp": true}' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeLayerwiseConnector",
      "kv_role": "kv_producer",
      "kv_port": "36000",
      "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 2,
                        "tp_size": 8
                 },
                 "decode": {
                        "dp_size": 32,
                        "tp_size": 1
                 }
          }
      }'
    ```

=== "Prefiller node 2"

    ```shell
    nic_name="eth0"  # network card name
    local_ip="192.0.0.2"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=256
    export TASK_QUEUE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE="AIV"
    export VLLM_USE_V1=1
    export ASCEND_RT_VISIBLE_DEVICES=$1
    vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --enable-expert-parallel \
      --seed 1024 \
      --served-model-name ds_r1 \
      --max-model-len 40000 \
      --max-num-batched-tokens 16384 \
      --max-num-seqs 8 \
      --enforce-eager \
      --trust-remote-code \
      --gpu-memory-utilization 0.9  \
      --quantization ascend \
      --no-enable-prefix-caching \
      --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
      --additional-config '{"enable_shared_expert_dp": true}' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeLayerwiseConnector",
      "kv_role": "kv_producer",
      "kv_port": "36100",
      "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 2,
                        "tp_size": 8
                 },
                 "decode": {
                        "dp_size": 32,
                        "tp_size": 1
                 }
          }
      }'
    ```

=== "Decoder node 1"

    ```shell
    nic_name="eth0"  # network card name
    local_ip="192.0.0.3"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=600
    export TASK_QUEUE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE="AIV"
    export VLLM_USE_V1=1
    export ASCEND_RT_VISIBLE_DEVICES=$1
    vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --enable-expert-parallel \
      --seed 1024 \
      --served-model-name ds_r1 \
      --max-model-len 40000 \
      --max-num-batched-tokens 256 \
      --max-num-seqs 40 \
      --trust-remote-code \
      --gpu-memory-utilization 0.94  \
      --quantization ascend \
      --no-enable-prefix-caching \
      --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
      --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"finegrained_tp_config": {"lmhead_tensor_parallel_size":16}}' \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeLayerwiseConnector",
      "kv_role": "kv_consumer",
      "kv_port": "36200",
      "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 2,
                        "tp_size": 8
                 },
                 "decode": {
                        "dp_size": 32,
                        "tp_size": 1
                 }
          }
      }'
    ```

=== "Decoder node 2"

    ```shell
    nic_name="eth0"  # network card name
    local_ip="192.0.0.4"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=600
    export TASK_QUEUE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE="AIV"
    export VLLM_USE_V1=1
    export ASCEND_RT_VISIBLE_DEVICES=$1
    vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --enable-expert-parallel \
      --seed 1024 \
      --served-model-name ds_r1 \
      --max-model-len 40000 \
      --max-num-batched-tokens 256 \
      --max-num-seqs 40 \
      --trust-remote-code \
      --gpu-memory-utilization 0.94  \
      --quantization ascend \
      --no-enable-prefix-caching \
      --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
      --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"finegrained_tp_config": {"lmhead_tensor_parallel_size":16}}' \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeLayerwiseConnector",
      "kv_role": "kv_consumer",
      "kv_port": "36200",
      "kv_connector_extra_config": {

                "prefill": {
                        "dp_size": 2,
                        "tp_size": 8
                 },
                 "decode": {
                        "dp_size": 32,
                        "tp_size": 1
                 }
          }
      }'
    ```

#### Non-layerwise

=== "Prefiller node 1"

    ```shell
    nic_name="eth0"  # network card name
    local_ip="192.0.0.1"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=256
    export TASK_QUEUE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE="AIV"
    export VLLM_USE_V1=1
    export ASCEND_RT_VISIBLE_DEVICES=$1
    vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --enable-expert-parallel \
      --seed 1024 \
      --served-model-name ds_r1 \
      --max-model-len 40000 \
      --max-num-batched-tokens 16384 \
      --max-num-seqs 8 \
      --enforce-eager \
      --trust-remote-code \
      --gpu-memory-utilization 0.9  \
      --quantization ascend \
      --no-enable-prefix-caching \
      --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
      --additional-config '{"enable_shared_expert_dp": true}' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV1",
      "kv_role": "kv_producer",
      "kv_port": "36000",
      "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 2,
                        "tp_size": 8
                 },
                 "decode": {
                        "dp_size": 32,
                        "tp_size": 1
                 }
          }
      }'
    ```

=== "Prefiller node 2"

    ```shell
    nic_name="eth0"  # network card name
    local_ip="192.0.0.2"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=256
    export TASK_QUEUE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE="AIV"
    export VLLM_USE_V1=1
    export ASCEND_RT_VISIBLE_DEVICES=$1
    vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --enable-expert-parallel \
      --seed 1024 \
      --served-model-name ds_r1 \
      --max-model-len 40000 \
      --max-num-batched-tokens 16384 \
      --max-num-seqs 8 \
      --enforce-eager \
      --trust-remote-code \
      --gpu-memory-utilization 0.9  \
      --quantization ascend \
      --no-enable-prefix-caching \
      --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
      --additional-config '{"enable_shared_expert_dp": true}' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV1",
      "kv_role": "kv_producer",
      "kv_port": "36100",
      "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 2,
                        "tp_size": 8
                 },
                 "decode": {
                        "dp_size": 32,
                        "tp_size": 1
                 }
          }
      }'
    ```

=== "Decoder node 1"

    ```shell
    nic_name="eth0"  # network card name
    local_ip="192.0.0.3"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=600
    export TASK_QUEUE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE="AIV"
    export VLLM_USE_V1=1
    export ASCEND_RT_VISIBLE_DEVICES=$1
    vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --enable-expert-parallel \
      --seed 1024 \
      --served-model-name ds_r1 \
      --max-model-len 40000 \
      --max-num-batched-tokens 256 \
      --max-num-seqs 40 \
      --trust-remote-code \
      --gpu-memory-utilization 0.94  \
      --quantization ascend \
      --no-enable-prefix-caching \
      --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
      --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"finegrained_tp_config": {"lmhead_tensor_parallel_size":16}}' \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV1",
      "kv_role": "kv_consumer",
      "kv_port": "36200",
      "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 2,
                        "tp_size": 8
                 },
                 "decode": {
                        "dp_size": 32,
                        "tp_size": 1
                 }
          }
      }'
    ```

=== "Decoder node 2"

    ```shell
    nic_name="eth0"  # network card name
    local_ip="192.0.0.4"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=600
    export TASK_QUEUE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE="AIV"
    export VLLM_USE_V1=1
    export ASCEND_RT_VISIBLE_DEVICES=$1
    vllm serve /path_to_weight/DeepSeek-r1_w8a8_mtp \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --enable-expert-parallel \
      --seed 1024 \
      --served-model-name ds_r1 \
      --max-model-len 40000 \
      --max-num-batched-tokens 256 \
      --max-num-seqs 40 \
      --trust-remote-code \
      --gpu-memory-utilization 0.94  \
      --quantization ascend \
      --no-enable-prefix-caching \
      --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
      --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"finegrained_tp_config": {"lmhead_tensor_parallel_size":16}}' \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV1",
      "kv_role": "kv_consumer",
      "kv_port": "36200",
      "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 2,
                        "tp_size": 8
                 },
                 "decode": {
                        "dp_size": 32,
                        "tp_size": 1
                 }
          }
      }'
    ```

### Start the service

```bash
# on 192.0.0.1
python launch_online_dp.py --dp-size 2 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address 192.0.0.1 --dp-rpc-port 12321 --vllm-start-port 7100
# on 192.0.0.2
python launch_online_dp.py --dp-size 2 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address 192.0.0.2 --dp-rpc-port 12321 --vllm-start-port 7100
# on 192.0.0.3
python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 16 --dp-rank-start 0 --dp-address 192.0.0.3 --dp-rpc-port 12321 --vllm-start-port 7100
# on 192.0.0.4
python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 16 --dp-rank-start 16 --dp-address 192.0.0.3 --dp-rpc-port 12321 --vllm-start-port 7100
```

## D2RH Deployment on Atlas 850E (UBoE)

**Validation scope:** D2RH has only been validated with the Atlas 850E
UBoE configuration shown below, using DeepSeek-V4-Flash. Other hardware,
network types, and deployment configurations have not been validated.

D2RH stages KV cache in the Decode node's host memory before loading it into
HBM. See the [D2RH workflow](../../developer_guide/Design_Documents/disaggregated_prefill.md#mooncake-d2rh-connector)
for the transfer sequence and cache-hit behavior.

**Memory limitation:** D2RH currently registers a pinned host KV cache on each
Decode worker with the same capacity as its device KV cache, plus alignment
overhead. This mirrors the KV cache, not the device's total HBM capacity.
Ensure that available host RAM and container memory limits cover the combined
allocation of all local Decode ranks, with headroom for model loading and other
process memory.

**1. Start P and D in separate terminals on their respective nodes.**

Replace the example IPs, interface names, and model paths with your deployment
values. Run each script on its corresponding node.

On P:

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy
unset FTP_PROXY HTTPS_PROXY HTTP_PROXY
unset HCCL_INTRA_ROCE_ENABLE

nic_name="eth0" # change to your own NIC name
local_ip=192.0.2.10 # change to your own IP

export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_ALGO=level0:fullmesh
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_BUFFSIZE=512
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'

# Ensure the model path matches the directory recorded during download
vllm serve /path/to/DeepSeek-V4-Flash-0731-w4a4c8 \
    --host "$local_ip" \
    --port 8100 \
    --served-model-name dsv4 \
    --data-parallel-size 4 \
    --data-parallel-address "$local_ip" \
    --data-parallel-rpc-port 12321 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.9 \
    --max-model-len 133120 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 32 \
    --block-size 32 \
    --enable-prefix-caching \
    --no-disable-hybrid-kv-cache-manager \
    --api-server-count 1 \
    --async-scheduling \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --trust-remote-code \
    --quantization ascend \
    --attention_config.indexer_kv_dtype fp8 \
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
    --enforce-eager \
    --kv-transfer-config \
    '{
        "kv_connector": "MooncakeD2RHConnectorV1",
        "kv_role": "kv_producer",
        "kv_port": "36010",
        "kv_connector_extra_config": {
            "prefill": {
                "dp_size": 4,
                "tp_size": 1
            },
            "decode": {
                "dp_size": 4,
                "tp_size": 1
            }
        }
    }' \
    --additional-config '{
        "enable_cpu_binding": true,
        "enable_shared_expert_dp": true,
        "multistream_overlap_shared_expert": true,
        "enable_flashcomm1": false
    }'
```

On D:

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy
unset FTP_PROXY HTTPS_PROXY HTTP_PROXY
unset HCCL_INTRA_ROCE_ENABLE

nic_name="eth0" # change to your own NIC name
local_ip=192.0.2.20 # change to your own IP

export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_ALGO=level0:fullmesh
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_BUFFSIZE=512
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'

# Ensure the model path matches the directory recorded during download
vllm serve /path/to/DeepSeek-V4-Flash-0731-w4a4c8 \
    --host "$local_ip" \
    --port 8200 \
    --served-model-name dsv4 \
    --data-parallel-size 4 \
    --data-parallel-address "$local_ip" \
    --data-parallel-rpc-port 12322 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.9 \
    --max-model-len 133120 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 32 \
    --block-size 32 \
    --enable-prefix-caching \
    --no-disable-hybrid-kv-cache-manager \
    --api-server-count 1 \
    --async-scheduling \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --trust-remote-code \
    --quantization ascend \
    --attention_config.indexer_kv_dtype fp8 \
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{
        "kv_connector": "MooncakeD2RHConnectorV1",
        "kv_role": "kv_consumer",
        "kv_port": "36020",
        "kv_connector_extra_config": {
            "d2rh_zmq_port": 38100,
            "d2rh_scheduler_ready_port": 38200,
            "prefill": {
                "dp_size": 4,
                "tp_size": 1
            },
            "decode": {
                "dp_size": 4,
                "tp_size": 1
            }
        }
    }' \
    --additional-config '{
        "enable_cpu_binding": true,
        "enable_shared_expert_dp": true,
        "multistream_overlap_shared_expert": true,
        "enable_flashcomm1": false
    }'
```

The scripts enable prefix caching on both sides, keep the hybrid KV cache
manager enabled, and set
`ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'`.
They use FP8 indexer cache on Atlas 850E, not the INT8 indexer setting from the
A2/A3 examples; see the [W4A4C8 FAQ](../models/DeepSeek-V4-Flash.md#q-how-do-the-w4a4c8-ultra-low-bit-quantized-weights-preserve-accuracy).
P uses eager execution and D uses `FULL_DECODE_ONLY` graphs.

The context limit is 133120 tokens, enough for 128k input plus 1k output, including
any prompt-template tokens within that limit.
Adjust `--gpu-memory-utilization` directly if needed.
Reducing it reduces available KV capacity and can reduce D's staging allocation;
recheck both host memory and maximum request capacity after changing it.

HTTP ports are 8100 (P) and 8200 (D); DP RPC ports are 12321 and 12322, and
`kv_port` bases are 36010 and 36020. The Decode script explicitly sets two
additional TCP/ZMQ control-port bases in `kv_connector_extra_config`:

- `d2rh_zmq_port` (default: `38100`): Each D worker listens for `START_PULL`
  requests from its local scheduler. These requests trigger staging of the
  required KV cache into D host memory, reusing valid host-cache entries when
  available. This port carries control messages, not the KV data itself.
- `d2rh_scheduler_ready_port` (default: `38200`): Each D scheduler listens for
  worker notifications that host staging is ready. Once the required workers
  report readiness, the request can proceed to device-block allocation and
  H2D loading. This notification does not mean that H2D or decoding has finished.

These are base ports, not single ports shared by all ranks. For this
DP4/TP1/PP1 deployment, D uses `38100-38103` for pull requests and
`38200-38203` for scheduler-ready notifications, with one port per DP rank in
each range. Other parallelism layouts introduce additional rank offsets;
reserve the full ranges for the deployment. Both settings are D-side controls
and do not need to be duplicated in the P script.

These ports are independent of `kv_port`, which is still used for Mooncake
handshake metadata and transfer-completion signaling. Changing `kv_port` does
not change either D2RH control-port base. Mooncake TransferEngine RPC and
UBoE/HCCL communication also have separate network requirements.

For multiple D instances sharing a host network, assign non-overlapping ranges
to both settings and avoid conflicts with HTTP, DP RPC, `kv_port`, and other
services. For example, a second DP4/TP1/PP1 instance can use bases `39100` and
`39200` if `39100-39103` and `39200-39203` are free. Configure the bases once in
that instance's D launch command; rank offsets are added automatically. Ensure
that D schedulers and workers can reach the bound D-node IP and these TCP ports.

**2. Check both services, then start the proxy.**

On a host that can reach both nodes, from the repository root:

```shell
unset http_proxy https_proxy ftp_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY
export P_IP=192.0.2.10
export D_IP=192.0.2.20
curl --fail "http://${P_IP}:8100/health"
curl --fail "http://${D_IP}:8200/health"
python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host 0.0.0.0 --port 8000 \
    --prefiller-hosts "$P_IP" --prefiller-ports 8100 \
    --decoder-hosts "$D_IP" --decoder-ports 8200
```

The proxy points at each node's single API endpoint; the server distributes
requests across its local DP ranks. Do not also run `launch_online_dp.py` for
this example. Keep these unauthenticated example endpoints on a trusted network.

## Example Proxy for Deployment

Run a proxy server on the same node where your prefiller service instance is deployed. You can find the proxy implementation in the repository's examples directory.

We provide two different proxy implementations with distinct request routing behaviors:

- **`load_balance_proxy_layerwise_server_example.py`**: Requests are first routed to the D nodes, which then forward to the P nodes as needed.This proxy is designed for use with the MooncakeLayerwiseConnector.[load\_balance\_proxy\_layerwise\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py)

- **`load_balance_proxy_server_example.py`**: Requests are first routed to the P nodes, which then forward to the D nodes for subsequent processing.This proxy is designed for use with the MooncakeConnector.[load\_balance\_proxy\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

=== "Layerwise"

    ```shell
    python load_balance_proxy_layerwise_server_example.py \
      --port 1999 \
      --host 192.0.0.1 \
      --prefiller-hosts \
        192.0.0.1 \
        192.0.0.1 \
        192.0.0.2 \
        192.0.0.2 \
      --prefiller-ports  \
        7100 7101 7100 7101 \
      --decoder-hosts \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
      --decoder-ports  \
        7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110 7111 7112 7113 7114 7115\
        7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110 7111 7112 7113 7114 7115
    ```

=== "Non-layerwise"

    ```shell
    python load_balance_proxy_server_example.py \
      --port 1999 \
      --host 192.0.0.1 \
      --prefiller-hosts \
        192.0.0.1 \
        192.0.0.1 \
        192.0.0.2 \
        192.0.0.2 \
      --prefiller-ports  \
        7100 7101 7100 7101 \
      --decoder-hosts \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.3  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
        192.0.0.4  \
      --decoder-ports  \
        7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110 7111 7112 7113 7114 7115\
        7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110 7111 7112 7113 7114 7115
    ```

Both proxies share the following backend options.

| Parameter | Applies to | Default | Example value | Meaning and constraints |
| --- | --- | --- | --- | --- |
| `--port` | Both | `8000` | `1999` | Proxy listen port. Must be free, reachable, and between `1` and `65535`. |
| `--host` | Both | `localhost` | `192.0.0.1` | Proxy listen address. Layerwise mode requires a non-wildcard address reachable by D nodes. |
| `--prefiller-hosts` | Both | `localhost` | `192.0.0.1 192.0.0.1 192.0.0.2 192.0.0.2` | P-instance hosts. Count must equal the number of `--prefiller-ports`. |
| `--prefiller-ports` | Both | `8001` | `7100 7101 7100 7101` | P-instance ports. Each host-port pair must be unique and reachable. |
| `--decoder-hosts` | Both | `localhost` | Sixteen `192.0.0.3`, then sixteen `192.0.0.4` | D-instance hosts. Count must equal the number of `--decoder-ports`. |
| `--decoder-ports` | Both | `8002` | `7100`-`7115` for each D host | D-instance ports. Each host-port pair must be unique and reachable. |

!!! note

    The proxy scripts validate that each host list has the same length as its corresponding port list.

You can get the proxy program in the repository's examples, [load\_balance\_proxy\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

## Benchmark

We recommend use aisbench tool to assess performance. [aisbench](https://github.com/AISBench/benchmark) Execute the following commands to install aisbench

```shell
git clone https://github.com/AISBench/benchmark.git
cd benchmark/
pip3 install -e ./
```

You need to cancel the http proxy before assessing performance, as following

```shell
# unset proxy
unset http_proxy
unset https_proxy
```

- You can place your datasets in the dir: `benchmark/ais_bench/datasets`
- You can change the configuration in the dir :`benchmark/ais_bench/benchmark/configs/models/vllm_api` Take the `vllm_api_stream_chat.py` for example

```python
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="/root/.cache/ds_r1",
        model="dsr1",
        request_rate = 14,
        retry = 2,
        host_ip = "192.0.0.1", # Proxy service host IP
        host_port = 8000,  # Proxy service Port
        max_out_len = 10,
        batch_size=768,
        trust_remote_code=True,
        generation_kwargs = dict(
            temperature = 0,
            seed = 1024,
            ignore_eos=False,
        )
    )
]
```

- Take gsm8k dataset for example, execute the following commands to assess performance.

```shell
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf  --debug  --mode perf
```

- For more details for commands and parameters for aisbench, refer to  [aisbench](https://github.com/AISBench/benchmark)

## FAQ

### 1. Prefiller nodes need to warmup

Since the computation of some NPU operators requires several rounds of warm-up to achieve best performance, we recommend preheating the service with some requests before conducting performance tests to achieve the best end-to-end throughput.

## Verification

Check service health using the proxy server endpoint.

```shell
curl http://192.0.0.1:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "ds_r1",
        "prompt": "Who are you?",
        "max_completion_tokens": 100,
        "temperature": 0
    }'
```
