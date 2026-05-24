# Running vLLM-Ascend as Non-Root User

This guide explains how to run vLLM-Ascend containers as a non-root user in Kubernetes environments, especially when using `--tensor-parallel-size > 1`.

## Background

When running vLLM-Ascend with tensor parallelism (TP > 1), HCCL (Huawei Collective Communication Library) requires a valid user entry in `/etc/passwd` for the UID running the container. If the container runs as a non-root user (e.g., UID 1000) but `/etc/passwd` does not contain an entry for that UID, HCCL initialization will fail with error code 19.

## Error Example

```text
HcclGetRootInfo(&hcclID), error code is 19
ra init failed, return[19] devicePhyId_[4], nicPosition[0]
```

## Solution

To run vLLM-Ascend as a non-root user with TP > 1, you need to ensure that the container has a valid user entry in `/etc/passwd` for the UID you want to use.

### Option 1: Build a Derived Image

Create a derived image that adds a user with UID 1000:

```dockerfile
FROM quay.io/ascend/vllm-ascend:v0.18.0rc1

USER root

# Create a user with UID 1000
RUN useradd \
    --uid 1000 \
    --gid 0 \
    --home-dir /tmp \
    --shell /usr/sbin/nologin \
    --no-create-home \
    vllm

# Set environment variables
ENV HOME=/tmp \
    USER=vllm \
    LOGNAME=vllm \
    USERNAME=vllm

# Switch to non-root user
USER 1000:0
```

Build and push the derived image:

```bash
docker build -t your-registry/vllm-ascend-nonroot:v0.18.0rc1 .
docker push your-registry/vllm-ascend-nonroot:v0.18.0rc1
```

### Option 2: Use Kubernetes Security Context

When deploying in Kubernetes, use the `securityContext` to run as a non-root user:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: vllm-ascend-service
spec:
  predictor:
    containers:
      - name: vllm
        image: your-registry/vllm-ascend-nonroot:v0.18.0rc1
        securityContext:
          runAsUser: 1000
          runAsNonRoot: true
          runAsGroup: 0
        # ... other configuration
```

## Why UID 1000?

UID 1000 is commonly used in Ascend/CANN environments because:

1. The default Ascend user `HwHiAiUser` typically uses UID/GID 1000
2. Many Ascend tools and libraries expect this UID
3. It's a common convention for first non-root user in Linux systems

## Required Mounts for Non-Root Mode

When running as non-root with TP > 1, you may need to mount additional files:

```yaml
volumeMounts:
  - name: hccn-config
    mountPath: /etc/hccn.conf
    readOnly: true
  - name: hccn-weak-dict
    mountPath: /etc/hccn_weak_dict.conf
    readOnly: true
  - name: ascend-driver
    mountPath: /usr/local/Ascend/driver
    readOnly: true
```

## Permissions Considerations

Running as non-root may require additional permissions for:

1. **CPU Binding**: Reading `/proc/interrupts` and writing `/proc/irq/*/smp_affinity`
2. **HCCL Configuration**: Reading `/etc/hccn.conf` and related files
3. **Device Access**: Access to `/dev/davinci*` devices (usually handled by Ascend device plugin)

## Environment Variables

Setting these environment variables may help, but an actual `/etc/passwd` entry is required for HCCL:

```yaml
env:
  - name: USER
    value: vllm
  - name: LOGNAME
    value: vllm
  - name: USERNAME
    value: vllm
  - name: HOME
    value: /tmp
```

## Troubleshooting

### HCCL Initialization Fails

If you see HCCL initialization errors:

1. Check that `/etc/passwd` contains an entry for the UID
2. Verify that the user has read access to `/etc/hccn.conf`
3. Ensure device files are accessible

### Permission Denied

If you see permission errors:

1. Check file permissions for mounted volumes
2. Verify the user's group membership (GID)
3. Consider using `runAsGroup: 0` for broader access

## References

- [Installation Guide](installation.md)
- [Kubernetes Deployment](using_volcano_kthena.md)
- [KV Pool Guide](../feature_guide/kv_pool.md)
- [CPU Binding Guide](../feature_guide/cpu_binding.md)
