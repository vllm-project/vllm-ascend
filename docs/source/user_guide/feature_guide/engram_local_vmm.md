# Local shared Engram VMM on A3

The release branch supports node-local Host VMM for DeepSeek-V4.1 shared
CPU-offloaded Engram. Each local rank maps the same host-backed tables through
a Fabric handle; codes and scales are separate allocations.

For the TP1/DP16 configuration, set both options explicitly:

```bash
--engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
--additional-config '{"engram_local_vmm":true}'
```

If `--additional-config` is already present, merge `"engram_local_vmm": true`
into that JSON object. Do not pass the option twice. The path requires INT8
codes, FP32 scales, a real checkpoint, a compatible A3/CANN VMM library and a
private shared-memory namespace for all local ranks. Use `false` to disable the
local VMM path.

The target release also supports external shared host pools and remote Engram
readers. An implicit local-VMM default yields to those existing backends; an
explicit request for local VMM together with either external backend is rejected.

The local VMM path has not been validated with the release's unchanged Mooncake
shutdown path. Recovery from a failed transfer or peer process death is also
unverified. A timeout or DONE notification does not prove that remote device
access has retired.
