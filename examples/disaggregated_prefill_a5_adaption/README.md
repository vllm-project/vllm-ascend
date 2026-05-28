# A5 Server Disaggregated Prefill-Deploy (PD) Adaptation Guide

This guide describes the steps to configure an A5 server for vLLM disaggregated prefill (PD) deployment with NPU endpoint routing.

## Overview

Disaggregated prefill requires proper NPU endpoint configuration so that HCCL communication layers correctly match the hardware topology. The setup involves two scripts that must be run sequentially:

1. **`gen_route.sh`** — reads NPU device topology from `/proc` and generates a route configuration file.
2. **`generate_ep.py`** — consumes the route file, the HCCL topology, and rootinfo to produce per-NPU endpoint JSON files.

Both scripts are located in this directory.

## Prerequisites

### 1. Install the dependency tool

```bash
python3 -m pip install unofficial-ascend-tools==0.0.7rc4
```

### 2. Generate `/etc/hccl_rootinfo.json`

Use the installed tool to generate the rootinfo file and copy and rename it to `/etc/hccl_rootinfo.json`:

```bash
mindcluster-tools rootinfo --product_type server
# The output filename (e.g. new_hccl_rootinfo.json) may vary by machine or tool version.
# Check the actual generated file and copy it accordingly.
cp new_hccl_rootinfo.json /etc/hccl_rootinfo.json
```

> **Note:** This file is required by `generate_ep.py` and describes the HCCL rank list, device IDs, EIDs, and topology file path.

## Step 1 — Generate Route Configuration

Run `gen_route.sh` to produce the route configuration file (`/lib/route.conf`). This script:

- Detects davinci/NPU devices via `/proc/ascend_ub` (or `/proc/asdrv_ub`).
- Extracts device ID and EID (local/remote) mappings from the kernel pair_info interface.
- Writes the route.conf file consumed by the next step.

```bash
bash gen_route.sh
```

On success, the script prints the number of devices found and the contents of the generated `/lib/route.conf`.

> **Tip:** If `/lib/route.conf` already exists, the script backs it up as `/lib/route.conf.bak` before regenerating.

## Step 2 — Generate Endpoint Configuration Files

Run `generate_ep.py` to produce per-NPU endpoint JSON files under `/etc/hixlep/`.

| Scenario       | Flag | Description                                       |
|----------------|------|---------------------------------------------------|
| A5 Server      | `-s` | 0+8 server topology (tested, recommended for A5)  |
| POD            | `-p` | 1D PoD topology (untested)                        |

**For A5 server (recommended):**

```bash
python generate_ep.py -s
```

**For POD (not yet tested):**

```bash
python generate_ep.py -p
```

## Output

After both scripts run successfully, the following files are produced:

| Path                               | Description                                   |
|------------------------------------|-----------------------------------------------|
| `/lib/route.conf`                  | Route configuration (device-to-EID mappings)  |
| `/etc/hixlep/ub_endpoint_npu_*.json` | One endpoint file per NPU device              |

## Post-Configuration — vLLM Environment Variable

With the updated worker code in `vllm_ascend/worker/worker.py`, you must set the following environment variable when launching vLLM:

```bash
export ASCEND_LOCAL_COMM_RES_PATH=/etc/hixlep
```

This tells the PD-adapted worker where to find the NPU endpoint configuration files.

## Full Command Summary

```bash
# 0. Prerequisites (one-time setup)
python3 -m pip install unofficial-ascend-tools==0.0.7rc4
mindcluster-tools rootinfo --product_type server
cp new_hccl_rootinfo.json /etc/hccl_rootinfo.json  # actual filename may vary

# 1. Generate route config
bash gen_route.sh

# 2. Generate endpoint configs (A5 server)
python generate_ep.py -s

# 3. Set vLLM environment variable
export ASCEND_LOCAL_COMM_RES_PATH=/etc/hixlep
```
