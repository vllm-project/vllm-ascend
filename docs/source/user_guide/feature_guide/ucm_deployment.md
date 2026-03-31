# UCM-Enhanced Prefix Caching Deployment Guide

## Overview

Unified Cache Management (UCM) provides an external KV-cache storage layer designed for prefix-caching scenarios in vLLM/vLLM-Ascend. Unlike KV Pooling, which expands prefix-cache capacity only by aggregating device memory and therefore remains limited by HBM/DRAM size and lacks persistence, UCM decouples compute from storage and adopts a tiered design. Each node uses local DRAM as a fast cache, while a shared backend—such as 3FS or enterprise-grade storage—serves as the persistent KV store. This approach removes the capacity ceiling imposed by device memory, enables durable and reliable prefix caching, and allows cache capacity to scale with the storage system rather than with compute resources.

## Prerequisites

* OS: Linux
* Hardware with Ascend NPUs. It's usually the Atlas 800 A2 series.
* **vLLM: main branch**
* **vLLM Ascend: main branch**

## UCM Installation

**Please refer to the [official UCM installation guide for Ascend NPU](https://ucm.readthedocs.io/en/latest/getting-started/quickstart_vllm_ascend.html)**

## Configure UCM for Prefix Caching

Modify the UCM configuration file to specify which UCM connector to use and where KV blocks should be stored.
You may directly edit the example file at:

`unified-cache-management/examples/ucm_config_example.yaml`

**For updated configuration options, please refer to the [official UCM documentation for prefix-caching](https://ucm.readthedocs.io/en/latest/user-guide/prefix-cache/nfs_store.html)**

A minimal configuration looks like this:

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmNfsStore"
    ucm_connector_config:
      storage_backends: "/mnt/test"
      use_direct: false

load_only_first_rank: false
```

Explanation:

* ucm_connector_name: "UcmNfsStore":
  Specifies `UcmNfsStore` as the UCM connector.

* storage_backends:
  Specify the directory used for storing KV blocks. It can be a local directory or an NFS-mounted path. UCM will store KV blocks here.
   **⚠️ Make sure to replace `"/mnt/test"` with your actual storage directory.**

* use_direct:
  Whether to enable direct I/O (optional). Default is `false`.

* load_only_first_rank:
  Controls whether only rank 0 loads KV cache and broadcasts it to other ranks.  
  This feature is currently not supported on Ascend, so it must be set to `false` (all ranks load/dump independently).

## Launching Inference

In this guide, we describe **online inference** using vLLM with the UCM connector, deployed as an OpenAI-compatible server. For best performance with UCM, it is recommended to set `block_size` to 128.

To start the vLLM server with the Qwen/Qwen2.5-14B-Instruct model, run:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 2 \
--gpu_memory_utilization 0.87 \
--block_size 128 \
--trust-remote-code \
--port 7800 \
--enforce-eager \
--no-enable-prefix-caching \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml"}
}'
```

**⚠️ Make sure to replace `"/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml"` with your actual config file path.**

If you see the log below:

```bash
INFO:     Started server process [1049932]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Congratulations, you have successfully started the vLLM server with UCM connector!

## Evaluating UCM Prefix Caching Performance

After launching the vLLM server with `UCMConnector` enabled, the easiest way to observe the prefix caching effect is to run the built-in `vllm bench` CLI. Executing the following command **twice** in a separate terminal shows the improvement clearly.

```bash
vllm bench serve \
--backend vllm \
--model Qwen/Qwen2.5-14B-Instruct \
--host 127.0.0.1 \
--port 7800 \
--dataset-name random \
--num-prompts 12 \
--random-input-len 16000 \
--random-output-len 2 \
--request-rate inf \
--seed 123456 \
--percentile-metrics "ttft,tpot,itl,e2el" \
--metric-percentiles "90,99" \
--ignore-eos
```

### After the first execution

The `vllm bench` terminal prints the benchmark result:

```shell
---------------Time to First Token----------------
Mean TTFT (ms):                           15323.87
```

Inspecting the vLLM server logs reveals entries like:

```shell
INFO ucm_connector.py:228: request_id: xxx, total_blocks_num: 125, hit hbm: 0, hit external: 0
```

This indicates that for the first inference request, UCM did not hit any cached KV blocks. As a result, the full 16K-token prefill must be computed, leading to a relatively large TTFT.

### After the second execution

Running the same benchmark again produces:

```shell
---------------Time to First Token----------------
Mean TTFT (ms):                            1920.68
```

The vLLM server logs now contain similar entries:

```shell
INFO ucm_connector.py:228: request_id: xxx, total_blocks_num: 125, hit hbm: 0, hit external: 125
```

This indicates that during the second request, UCM successfully retrieved all 125 cached KV blocks from the storage backend. Leveraging the fully cached prefix significantly reduces the initial latency observed by the model, yielding an approximate **8× improvement in TTFT** compared to the initial run.
