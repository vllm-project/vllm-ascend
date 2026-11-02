# Environment Variables

vllm-ascend uses the following environment variables to configure the system:

**Note:** Some environment variables are being migrated to `--additional-config` options. These environment variables are still supported during the migration period, and it is recommended to use `--additional-config` for new deployments. See [Additional Configuration](additional_config.md) for details.

{{ include_code('vllm_ascend/envs.py', start_after='begin-env-vars-definition', end_before='end-env-vars-definition') }}

## Temporary grammar bitmask synchronization

`VLLM_ASCEND_SYNCHRONIZE_GRAMMAR_BITMASK=1` synchronizes the current NPU stream
before and after the V2 Triton grammar bitmask kernel. The default is `0`, which
preserves asynchronous submission. This non-sensitive setting is intended for
investigating suspected lock-workspace reuse in Triton-Ascend launchers that
initialize cached lock memory on the host without stream ordering.

The first synchronization completes earlier work on the submission stream before
grammar's lock initialization. The second waits for grammar to finish before the
calling host thread can submit later operations that might reuse the workspace.
This is a temporary mitigation for that submission pattern, not a general fix for
the runtime allocator, concurrent host threads, or unrelated kernel timeouts.
It cannot recover a kernel that is already stuck, and it does not establish the
cause of a timeout.

Enable this setting consistently in the affected workers at startup. It can
reduce throughput and increase latency; measure the impact before production use.
Use eager execution for the affected path. If the grammar launch occurs during
graph capture, the workaround raises an error before launching the kernel because
host synchronization cannot be captured. Disabling the setting restores the
original submission behavior.
