# RFork Guide

This guide explains how to use **RFork** as a fast elastic weight-loading plugin in **vLLM Ascend**.

---

## Overview

RFork reduces model startup latency by reusing already-loaded weights from a seed instance and transferring them through a transfer engine.

In vLLM Ascend, RFork is integrated through the model-loader plugin mechanism (`register_model_loader`).

### Components

- **Planner (scheduler)**: manages seed allocation with HTTP APIs:
  - `GET /get_seed`
  - `POST /add_seed`
  - `POST /put_seed`
- **Seed instance**: starts a local HTTP service and exposes transfer metadata.
- **Client instance**: requests a seed from planner, pulls metadata from seed, and performs transfer.
- **Transfer engine backend**: provided by `@yuanrong-datasystem/transfer_engine`.

> Note: RFork in vLLM Ascend uses `transfer_engine` API, not Mooncake API.

---

## Prerequisites

- Install vLLM and vLLM Ascend.
- Install transfer engine package from your environment for
  `from transfer_engine import TransferEngine`.
- Deploy planner service compatible with RFork seed HTTP headers.

Required environment variables:

- `MODEL_URL`
- `MODEL_DEPLOY_STRATEGY_NAME`
- `RFORK_SCHEDULER_URL`

Optional environment variables:

- `RFORK_SEED_KEY_SEPARATOR` (default: `$`)
- `VLLM_RFORK_ENABLED` (default: `0`)

---

## Usage Modes

### 1) Explicit mode (recommended first)

Set load format directly:

```shell
vllm serve <model> \
  --load-format rfork
```

### 2) Auto mode

Set environment variable to auto-switch to RFork loader:

```shell
export VLLM_RFORK_ENABLED=1
vllm serve <model>
```

If RFork initialization fails in auto mode, loader falls back to previous `load_format`.

---

## Planner HTTP Contract

RFork planner APIs use **HTTP headers** for parameters.

### `GET /get_seed`

Request headers:

- `SEED_KEY`

Success response headers:

- `SEED_IP`
- `SEED_PORT`
- `USER_ID`
- `SEED_RANK`

### `POST /add_seed`

Request headers:

- `SEED_KEY`
- `SEED_IP`
- `SEED_PORT`
- `SEED_RANK`
- `SEED_REFCNT`

### `POST /put_seed`

Request headers:

- `SEED_IP`
- `SEED_PORT`
- `SEED_RANK`
- `USER_ID`

---

## Operational Notes

- Seed service publishes transfer metadata through local FastAPI endpoints.
- Planner should support heartbeat-based seed liveness management.
- Ensure `RFORK_SCHEDULER_URL` is reachable from all worker nodes.
- Ensure transfer engine runtime dependencies are installed on every node.

