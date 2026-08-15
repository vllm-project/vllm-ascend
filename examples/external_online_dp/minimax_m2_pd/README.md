# MiniMax-M2.5 PD Separation (1P1D) Online Deployment

Reference: [MiniMax-M2 tutorial, section 5.2 Multi-Node PD Separation Deployment](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/MiniMax-M2.html#52-multi-node-pd-separation-deployment)

This example brings up a **1P1D** (1 Prefill + 1 Decode) MiniMax-M2.5 w8a8 QuaRot
cluster with 200k context length:

- **Hardware**: 2x Atlas 800 A3 (64GB x 16 NPUs), one node for Prefill, one for Decode
- **Topology**: Prefill DP2/TP8/EP16, Decode DP2/TP8/EP16
- **Context**: `--max-model-len 200000`
- **KV transfer**: Mooncake (Ascend direct), prefill is `kv_producer`, decode is `kv_consumer`
- **Decode acceleration**: EAGLE3 speculative decoding + `FULL_DECODE_ONLY` graph

The same layout applies to MiniMax-M2.7 (swap the model/EAGLE3 weight paths).

## 0. Prerequisites

- Download the base weights and EAGLE3 weights (ModelScope) to a directory
  reachable from both nodes (e.g. `/root/.cache/`):
  - `MiniMax-M2.5-w8a8-QuaRot`
  - `MiniMax-M2.5-eagle-model-0318`
- Mooncake, jemalloc, and the CANN toolkit must be installed inside the container.

## 1. Prepare scripts on each node

`launch_online_dp.py` launches one `vllm serve` instance per local DP rank and
always invokes `./run_dp_template.sh` relative to the working directory. Put the
files together and copy the matching template to `run_dp_template.sh`:

```bash
cd examples/external_online_dp/minimax_m2_pd
cp ../launch_online_dp.py .

# Prefill node:
cp run_dp_template_prefill.sh run_dp_template.sh

# Decode node (different content!):
cp run_dp_template_decode.sh run_dp_template.sh
```

Edit `NIC_NAME` and `LOCAL_IP` (or export them) in `run_dp_template.sh` on each
node. Set `MODEL_PATH` / `EAGLE_MODEL_PATH` to your actual weight paths.

## 2. Start the servers

**Prefill node:**

```bash
python launch_online_dp.py \
    --dp-size 2 --tp-size 8 \
    --dp-size-local 2 --dp-rank-start 0 \
    --dp-address <prefill_ip> --dp-rpc-port 12321 \
    --vllm-start-port 7000
```

**Decode node:**

```bash
python launch_online_dp.py \
    --dp-size 2 --tp-size 8 \
    --dp-size-local 2 --dp-rank-start 0 \
    --dp-address <decode_ip> --dp-rpc-port 12321 \
    --vllm-start-port 7100
```

## 3. Start the load-balance proxy

Run on any machine that can reach both nodes:

```bash
python ../../disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --port 8009 \
    --host <prefill_ip> \
    --prefiller-hosts <prefill_ip> <prefill_ip> \
    --prefiller-ports 7000 7001 \
    --decoder-hosts <decode_ip> <decode_ip> \
    --decoder-ports 7100 7101
```

The service is accessible at `http://<proxy_ip>:8009`. All requests go to the
proxy; it forwards prefill to the Prefill node and decode to the Decode node.

## 4. Verification

The servers are launched with `--max-model-len 200000`, so the model context
length is capped at 200k. A simple curl with a short prompt is enough to verify
the PD pipeline end to end:

```bash
curl -sS http://<proxy_ip>:8009/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model": "minimax",
         "messages": [{"role": "user", "content": "What is deep learning?"}],
         "max_tokens": 128}'
```

Note: the served model name is `minimax` (`--served-model-name minimax`).

## 5. Automated nightly variant

The same 1P1D 200k scenario is registered as a nightly A3 multi-node case:
`tests/e2e/nightly/multi_node/external_dp/config/MiniMax-M2.5-w8a8-QuaRot-A3-PD-200k.yaml`
(see `nightly_config.yaml`, `a3.multi_node`).
