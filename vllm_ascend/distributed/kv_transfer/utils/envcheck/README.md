# vLLM-Ascend KV Pool Environment Checker v1.0.0

This tool performs pre-deployment environment checks for vLLM-Ascend KV Pool deployments. It identifies dependency issues, reports blocking conditions and remediation guidance, and executes supported local repairs only after explicit authorization.

The tool runs only inside a vLLM-Ascend Docker container. It operates offline and is read-only by default. The `rules.json` file in this directory is the source of truth for rules and supported configurations.

The directory also contains the runtime artifacts `artifacts/mem_scan.py` and `artifacts/hccs_test_a3.py`, plus the source manifest `artifacts/manifest.json`.

## Invocation

### Run from the repository package

Run the following command from the vLLM-Ascend repository root or from a Python environment with vLLM-Ascend installed:

```bash
python -m vllm_ascend.distributed.kv_transfer.utils.envcheck.kvpool_envcheck \
  --hardware A3 \
  --backend memcache \
  --managed-memory-mb 32768 \
  --deployment-mode single-node \
  --link-type hccs
```

### Run from a standalone directory

Copy the complete `envcheck` directory to any location, then run the following command from that directory:

```bash
cd /path/to/envcheck
python kvpool_envcheck.py \
  --hardware A3 \
  --backend memcache \
  --managed-memory-mb 32768 \
  --deployment-mode single-node \
  --link-type hccs
```

Standalone use requires both `rules.json` and `artifacts/`. The remaining examples use the package invocation. For standalone use, replace
`python -m vllm_ascend.distributed.kv_transfer.utils.envcheck.kvpool_envcheck` with
`python kvpool_envcheck.py`.

## Quick start

### Print the report

```bash
python -m vllm_ascend.distributed.kv_transfer.utils.envcheck.kvpool_envcheck \
  --hardware A3 \
  --backend memcache \
  --managed-memory-mb 32768 \
  --deployment-mode single-node \
  --link-type hccs
```

This mode does not create files. The report is written to standard output, and progress is written to standard error.

### Save the results

```bash
python -m vllm_ascend.distributed.kv_transfer.utils.envcheck.kvpool_envcheck \
  --hardware A3 \
  --backend memcache \
  --managed-memory-mb 32768 \
  --deployment-mode multi-node \
  --link-type hccs \
  --output-dir ./results
```

Each run creates a pair of files with the same `run_id`:

| File | Purpose |
|---|---|
| `kvpool_envcheck_<run_id>.result.json` | Complete machine-readable result, raw evidence, repair actions, and reference documents |
| `kvpool_envcheck_<run_id>.report.txt` | Human-readable environment report highlighting conclusions, blockers, and recommendations |

When `--output-dir` is omitted, no files are created. When it is specified, standard output contains only the final conclusion, `run_id`, and output paths.

## Parameters and combinations

Run `python -m vllm_ascend.distributed.kv_transfer.utils.envcheck.kvpool_envcheck -h` to view complete help generated from the current rule catalog. Check mode requires all of the following parameters:

| Parameter | Description |
|---|---|
| `--hardware` | Hardware platform. Currently supports A2 and A3, and must match the hardware detected in the current container |
| `--backend` | KV Pool backend. Currently supports `memcache` and `mooncake` |
| `--managed-memory-mb MB` | Total backend-managed memory in MB. Must be greater than 0 and is compared against global capacity |
| `--deployment-mode` | `single-node` or `multi-node` |
| `--link-type` | `roce` or `hccs`. A2 supports only `roce`; A3 supports both |

Configuration constraints:

- A2 supports RoCE checks only.
- On A2, static huge-page capacity is the unreserved allocatable capacity: `(HugePages_Free - HugePages_Rsvd) * Hugepagesize`. Missing or inconsistent reservation evidence blocks the check.
- A3 with `roce` runs static RoCE port checks and dynamic local NPU-to-NPU checks.
- A3 with `hccs` runs static and dynamic intra-node SIO/HCCS checks. Multi-node deployments also require manual confirmation of the Lingqu CC version.
- A3 Mooncake requires HDK `>=25.5.0` with Mooncake `>=0.3.11.post1`, or HDK `>=26.0.0` with no minimum Mooncake version. A2 Mooncake requires Mooncake `>=0.3.11.post1`.
- `KERNEL-001` applies only to multi-node deployments. Single-node deployments return `NOT_APPLICABLE` and are not blocked by this rule.
- The `--hardware` value must be supported and must exactly match the detected hardware.

## SSD Offload

Enabling SSD checks requires `--ssd-path`:

```bash
python -m vllm_ascend.distributed.kv_transfer.utils.envcheck.kvpool_envcheck \
  --hardware A3 \
  --backend memcache \
  --managed-memory-mb 32768 \
  --deployment-mode multi-node \
  --link-type hccs \
  --ssd-offload \
  --ssd-path /dev/nvme1n1 \
  --output-dir ./results
```

- For `memcache`, `--ssd-path` must be an unmounted dedicated block device, partition, or loop device without a filesystem signature. A directory is not accepted.
- For `mooncake`, `--ssd-path` may be a writable directory or block device with sufficient free space.
- When `--ssd-offload` is omitted, SSD dependency rules do not participate in the check.

## Repair authorization

`--fix` explicitly authorizes the tool to execute supported local repairs:

```bash
python -m vllm_ascend.distributed.kv_transfer.utils.envcheck.kvpool_envcheck \
  --hardware A3 \
  --backend memcache \
  --managed-memory-mb 32768 \
  --deployment-mode multi-node \
  --link-type hccs \
  --fix \
  --output-dir ./results
```

Currently, only `MEM-001` on A3 supports automatic repair. When both 1 GB and 2 MB scans report insufficient global capacity, the tool attempts to drop caches inside the container and rescans. This action requires root privileges and a writable `/proc/sys/vm/drop_caches`, and may temporarily affect I/O.

The tool does not drop caches when A2 static huge pages are insufficient. It also does not automatically upgrade HDK, CANN, the kernel, network devices, container mounts, or software packages. The result includes manual remediation guidance for these cases. Repair action states are `SUCCESS`, `NO_EFFECT`, `FAILED`, `SKIPPED`, and `MANUAL`.

## Output states and exit codes

An individual scenario can return:

- `PASS`: the issue was not detected.
- `BLOCK`: a blocking issue was detected.
- `WARN`: a risk was detected and requires confirmation.
- `MANUAL_REQUIRED`: manual confirmation is required.
- `NOT_APPLICABLE`: the current hardware, backend, or deployment mode is not applicable.
- `NOT_EXECUTED`: the check was not executed because prerequisites were not met.

Process exit codes:

| Exit code | Meaning |
|---:|---|
| `0` | Checks passed |
| `10` | Warnings or manual confirmations are present |
| `20` | Blocking issues are present |
| `2` | Input, rule, or output error |

Real-time progress uses `--progress auto` by default. Progress is displayed in interactive terminals and disabled for non-interactive invocations. It can also be selected explicitly:

```bash
--progress always   # Always display progress
--progress never    # Disable progress
```

Progress is written only to standard error and does not mix with TEXT or JSON results. Each check stage reports only its start, completion, and elapsed time.

## Rule maintenance

`rules.json` is the single runtime source of truth for scenarios, applicability, evaluation methods, and remediation guidance. Supported configurations are reflected automatically in `-h` output, and this README does not duplicate the rule list.

```bash
# Validate the rule catalog
python -m vllm_ascend.distributed.kv_transfer.utils.envcheck.kvpool_envcheck \
  --validate-catalog

# List all scenarios
python -m vllm_ascend.distributed.kv_transfer.utils.envcheck.kvpool_envcheck \
  --list-rules

# Describe one scenario
python -m vllm_ascend.distributed.kv_transfer.utils.envcheck.kvpool_envcheck \
  --describe-rule MEM-001
```

## References

The `reference_documents` section of the result records the official [vLLM-Ascend KV Cache Pool User Guide](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md), including its source commit and review date. The tool does not access the network at runtime.
