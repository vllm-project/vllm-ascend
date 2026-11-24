# store_candidate_pages

## Description

The Kimi MegaRecurrentKda adapter calls `store_candidate_pages` after an ordinary decode or speculative verification group. It writes each candidate prefix's recurrent state into the scheduler's physical page for that prefix. The caller can subsequently select the page for the accepted prefix.

For request `b` and head `h`, start with `S = initial_state[b,h]`. At each real token position `s`, update each value/key element in FP32:

```text
S[v,k] = S[v,k] * decay[b,h,s,k] + correction[b,h,s,v] * key[b,h,s,k]
```

One Triton launch distributes requests, heads and tiles of value rows across programs. Each program holds its state tile as it processes the prefix in order. It stores the tile when the destination is valid. A missing destination skips that store and the recurrence continues. The entry point supports Ascend eager execution and fixed-shape graph capture/replay. The model target is Ascend 950 (A5).

## Parameters

All arguments are required and reside on the same device. The function returns `None`.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `state_cache` | Output, in place | Physical page pool `[P,H,D,D]` in value/key order | FP32 | Dense within each page, page stride may include padding |
| `initial_state` | Input | Per-request initial checkpoint `[B,H,D,D]` | FP32 | Contiguous |
| `correction` | Input | Value-axis correction `[B,H,W,D]` | FP32 | Contiguous |
| `key` | Input | Key-axis factor `[B,H,W,D]` | FP16 | Contiguous |
| `decay` | Input | Key-axis decay `[B,H,W,D]` | FP32 | Contiguous |
| `destinations` | Input | Physical page ID for each prefix `[B,W]` | int32 or int64 | Strided |
| `lengths` | Input | Number of real tokens in each window `[B]` | int32 or int64 | Strided |
| `active` | Input | Requests with valid scheduler metadata `[B]` | bool | Strided |

## Constraints

Only active requests with a position below both their length and `W` can write. Page IDs must satisfy `0 < page < P` to receive a write. Page zero, unselected pages and the padding between pages retain their contents. Live requests own disjoint destination pages. The packed source tensors occupy separate storage from the cache. The cache may have a nonzero storage offset.

Graph replay reuses the captured shapes, strides and addresses. The kernel reads current lengths, activity flags and destinations from these buffers on each replay. Head count, head dimension, window and tile geometry specialize the kernel. A batch-size change affects the launch grid.

## Origin and Differences

This model-specific operator groups the adapter's per-position FP32 updates and page writes in a single launch per KDA layer. The upstream MegaRecurrentKda operator supplies the correction, key and decay factors. Convolution history and output tokens use their existing writeback paths.

## Test Cases

The single-card test compares the entire backing allocation and the read-only inputs against a PyTorch reference. It covers `H=12`, `D=128`, single-token decode (`W=1`) and DSpark7 verification (`W=8`). It also checks strided metadata, invalid slots between valid prefixes, varying lengths and changing request ownership on graph replay. The dyadic factors give exact FP32 reference values for these windows.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_kda_candidate_pages.py
```
