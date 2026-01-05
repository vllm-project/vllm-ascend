# Using Volcano Kthena

This guide shows how to run **prefill–decode (PD) disaggregation** on Huawei Ascend NPUs using **vLLM-Ascend**, with [**Kthena**](https://kthena.volcano.sh/) handling orchestration on Kubernetes. About vLLM support with kthena, please refer to [Deploy vLLM with Kthena](https://docs.vllm.ai/en/latest/deployment/integrations/kthena/).

---

## 1. What is Prefill–Decode Disaggregation?

Large language model inference naturally splits into two phases:

- **Prefill**
  - Processes input tokens and builds the key–value (KV) cache.
  - Batch‑friendly, high throughput, well suited to parallel NPU execution.
- **Decode**
  - Consumes the KV cache to generate output tokens.
  - Latency‑sensitive, memory‑intensive, more sequential.

From the client’s perspective, this still looks like a single Chat / Completions endpoint.

---

## 2. Deploy on Kubernetes with Kthena

[Kthena](https://kthena.volcano.sh/) is a Kubernetes-native LLM inference platform that transforms how organizations deploy and manage Large Language Models in production. Built with declarative model lifecycle management and intelligent request routing, it provides high performance and enterprise-grade scalability for LLM inference workloads. In this example, we use three key Custom Resource Definitions (CRDs):

- `ModelServing` — defines the workloads (prefill and decode roles).
- `ModelServer` — manages PD groupings and internal routing.
- `ModelRoute` — exposes a stable model endpoint.

This section uses the `deepseek-ai/DeepSeek-V2-Lite` example, but you can swap in any model supported by vLLM-Ascend.

### 2.1 Prerequisites

- Kubernetes cluster with Ascend NPU nodes:

    The Resources corresponding to different NPU Drivers may vary slightly. For example:

    - If using [MindCluster](https://gitee.com/ascend/mind-cluster#https://gitee.com/link?target=https%3A%2F%2Fgitcode.com%2FAscend%2Fmind-cluster), please use `huawei.com/Ascend310P` or `huawei.com/Ascend910`.

    - If running on CCE (Cloud Container Engine) of Huawei Cloud and the [CCE AI Suite Plugin (Ascend NPU)](https://support.huaweicloud.com/intl/en-us/usermanual-cce/cce_10_0239.html) is installed, please use `huawei.com/ascend-310` or `huawei.com/ascend-1980`.

- Kthena installed. Please follow the [Kthena installation guide](https://kthena.volcano.sh/docs/getting-started/installation).

### 2.2 Deploy Prefill-Decode Disaggregated DeepSeek-V2-Lite on Kubernetes

A concrete example is provided in Kthena as https://github.com/volcano-sh/kthena/blob/main/examples/model-serving/prefill-decode-disaggregation.yaml

Deploy it with below command:

```bash
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/kthena/refs/heads/main/examples/model-serving/prefill-decode-disaggregation.yaml
```

or

```bash
cat << EOF | kubectl apply -f -
apiVersion: workload.serving.volcano.sh/v1alpha1
kind: ModelServing
metadata:
  name: deepseek-v2-lite
  namespace: dev
spec:
  schedulerName: volcano
  replicas: 1
  recoveryPolicy: ServingGroupRecreate
  template:
    restartGracePeriodSeconds: 60
    roles:
      - name: prefill
        replicas: 1
        entryTemplate:
          spec:
            initContainers:
              - name: downloader
                imagePullPolicy: Always
                image: ghcr.io/volcano-sh/downloader:latest
                args:
                  - --source
                  - deepseek-ai/DeepSeek-V2-Lite
                  - --output-dir
                  - /mnt/cache/deepseek-ai/DeepSeek-V2-Lite/
                volumeMounts:
                  - name: models
                    mountPath: /mnt/cache/deepseek-ai/DeepSeek-V2-Lite/
            containers:
              - name: runtime
                image: ghcr.io/volcano-sh/runtime:latest
                ports:
                  - containerPort: 8100
                args:
                  - --port
                  - "8100"
                  - --engine
                  - vllm
                  - --pod
                  - $(POD_NAME).$(NAMESPACE)
                  - --model
                  - deepseek-v2-lite
                  - --engine-base-url
                  - http://localhost:8000
              - name: vllm
                image: ghcr.io/volcano-sh/kthena-engine:vllm-ascend_v0.10.1rc1_mooncake_v0.3.5
                ports:
                  - containerPort: 8000
                env:
                  - name: HF_HUB_OFFLINE
                    value: "1"
                  - name: HCCL_IF_IP
                    valueFrom:
                      fieldRef:
                        fieldPath: status.podIP
                  - name: GLOO_SOCKET_IFNAME
                    value: eth0
                  - name: TP_SOCKET_IFNAME
                    value: eth0
                  - name: HCCL_SOCKET_IFNAME
                    value: eth0
                  - name: VLLM_LOGGING_LEVEL
                    value: DEBUG
                  - name: AscendRealDevices
                    valueFrom:
                      fieldRef:
                        fieldPath: metadata.annotations['huawei.com/AscendReal']
                args:
                  - "/mnt/cache/deepseek-ai/DeepSeek-V2-Lite/"
                  - "--served-model-name"
                  - "deepseek-ai/DeepSeekV2"
                  - "--tensor-parallel-size"
                  - "2"
                  - "--gpu-memory-utilization"
                  - "0.8"
                  - "--max-model-len"
                  - "8192"
                  - "--max-num-batched-tokens"
                  - "8192"
                  - "--trust-remote-code"
                  - "--enforce-eager"
                  - "--kv-transfer-config"
                  - '{"kv_connector":"MooncakeConnectorV1","kv_buffer_device":"npu","kv_role":"kv_producer","kv_parallel_size":1,"kv_port":"20001","engine_id":"0","kv_rank":0,"kv_connector_module_path":"vllm_ascend.distributed.mooncake_connector","kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}}'
                imagePullPolicy: Always
                resources:
                  limits:
                    cpu: "8"
                    memory: 64Gi
                    huawei.com/ascend-1980: "4"
                  requests:
                    cpu: "8"
                    memory: 64Gi
                    huawei.com/ascend-1980: "4"
                readinessProbe:
                  initialDelaySeconds: 5
                  periodSeconds: 5
                  failureThreshold: 3
                  httpGet:
                    path: /health
                    port: 8000
                livenessProbe:
                  initialDelaySeconds: 900
                  periodSeconds: 5
                  failureThreshold: 3
                  httpGet:
                    path: /health
                    port: 8000
                volumeMounts:
                  - name: models
                    mountPath: /mnt/cache/deepseek-ai/DeepSeek-V2-Lite/
                    readOnly: true
                  - name: hccn-config
                    mountPath: /etc/hccn.conf
                    readOnly: true
                  - name: shared-memory-volume
                    mountPath: /dev/shm
            volumes:
              - name: models
                hostPath:
                  path: /mnt/cache/deepseek-ai/DeepSeek-V2-Lite/
                  type: DirectoryOrCreate
              - name: hccn-config
                hostPath:
                  path: /etc/hccn.conf
                  type: File
              - name: shared-memory-volume
                emptyDir:
                  sizeLimit: 256Mi
                  medium: Memory
      - name: decode
        replicas: 1
        entryTemplate:
          spec:
            initContainers:
              - name: downloader
                imagePullPolicy: Always
                image: ghcr.io/volcano-sh/downloader:latest
                args:
                  - --source
                  - deepseek-ai/DeepSeek-V2-Lite
                  - --output-dir
                  - /mnt/cache/deepseek-ai/DeepSeek-V2-Lite/
                volumeMounts:
                  - name: models
                    mountPath: /mnt/cache/deepseek-ai/DeepSeek-V2-Lite/
            containers:
              - name: vllm
                image: ghcr.io/volcano-sh/kthena-engine:vllm-ascend_v0.10.1rc1_mooncake_v0.3.5
                ports:
                  - containerPort: 8000
                env:
                  - name: HF_HUB_OFFLINE
                    value: "1"
                  - name: HCCL_IF_IP
                    valueFrom:
                      fieldRef:
                        fieldPath: status.podIP
                  - name: GLOO_SOCKET_IFNAME
                    value: eth0
                  - name: TP_SOCKET_IFNAME
                    value: eth0
                  - name: HCCL_SOCKET_IFNAME
                    value: eth0
                  - name: VLLM_LOGGING_LEVEL
                    value: DEBUG
                  - name: AscendRealDevices
                    valueFrom:
                      fieldRef:
                        fieldPath: metadata.annotations['huawei.com/AscendReal']
                args:
                  - "/mnt/cache/deepseek-ai/DeepSeek-V2-Lite/"
                  - "--served-model-name"
                  - "deepseek-ai/DeepSeekV2"
                  - "--tensor-parallel-size"
                  - "2"
                  - "--gpu-memory-utilization"
                  - "0.8"
                  - "--max-model-len"
                  - "8192"
                  - "--max-num-batched-tokens"
                  - "16384"
                  - "--trust-remote-code"
                  - "--no-enable-prefix-caching"
                  - "--enforce-eager"
                  - "--kv-transfer-config"
                  - '{"kv_connector":"MooncakeConnectorV1","kv_buffer_device":"npu","kv_role":"kv_consumer","kv_parallel_size":1,"kv_port":"20002","engine_id":"1","kv_rank":1,"kv_connector_module_path":"vllm_ascend.distributed.mooncake_connector","kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}}'
                imagePullPolicy: Always
                resources:
                  limits:
                    cpu: "8"
                    memory: 64Gi
                    huawei.com/ascend-1980: "4"
                  requests:
                    cpu: "8"
                    memory: 64Gi
                    huawei.com/ascend-1980: "4"
                readinessProbe:
                  initialDelaySeconds: 5
                  periodSeconds: 5
                  failureThreshold: 3
                  httpGet:
                    path: /health
                    port: 8000
                livenessProbe:
                  initialDelaySeconds: 900
                  periodSeconds: 5
                  failureThreshold: 3
                  httpGet:
                    path: /health
                    port: 8000
                volumeMounts:
                  - name: models
                    mountPath: /mnt/cache/deepseek-ai/DeepSeek-V2-Lite/
                    readOnly: true
                  - name: hccn-config
                    mountPath: /etc/hccn.conf
                    readOnly: true
                  - name: shared-memory-volume
                    mountPath: /dev/shm
            volumes:
              - name: models
                hostPath:
                  path: /mnt/cache/deepseek-ai/DeepSeek-V2-Lite/
                  type: DirectoryOrCreate
              - name: hccn-config
                hostPath:
                  path: /etc/hccn.conf
                  type: File
              - name: shared-memory-volume
                emptyDir:
                  sizeLimit: 256Mi
                  medium: Memory
EOF
```

You should see Pods such as:

- `deepseek-v2-lite-0-prefill-0-0`
- `deepseek-v2-lite-0-decode-0-0`

To enable the llm access, we still need to configure the routing layer with `ModelServer` and `ModelRoute`.

### 2.3 ModelServer: PD Group Management

The `ModelServer` resource:

- Selects the `ModelServing` workloads via labels.
- Groups prefill and decode Pods into PD pairs.
- Configures KV connector details and timeouts.
- Exposes an internal gRPC/HTTP interface.

Create modelServer with below command:

```bash
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/kthena/refs/heads/main/examples/kthena-router/ModelServer-prefill-decode-disaggregation.yaml
```

or

```bash
cat << EOF | kubectl apply -f -
apiVersion: networking.serving.volcano.sh/v1alpha1
kind: ModelServer
metadata:
  name: deepseek-v2
  namespace: dev
spec:
  kvConnector:
    type: nixl
  workloadSelector:
    matchLabels:
      modelserving.volcano.sh/name: deepseek-v2-lite
    pdGroup:
      groupKey: "modelserving.volcano.sh/group-name"
      prefillLabels:
        modelserving.volcano.sh/role: prefill
      decodeLabels:
        modelserving.volcano.sh/role: decode
  workloadPort:
    port: 8000
  model: "deepseek-ai/DeepSeekV2"
  inferenceEngine: "vLLM"
  trafficPolicy:
    timeout: 10s
EOF
```

### 2.4 ModelRoute: User-Facing Endpoint

The `ModelRoute` resource maps a model name (e.g., `"deepseek-ai/DeepSeekV2"`) to the `ModelServer`.

Example manifest:

```bash
cat << EOF | kubectl apply -f -
apiVersion: networking.serving.volcano.sh/v1alpha1
kind: ModelRoute
metadata:
  name: deepseek-v2
  namespace: dev
spec:
  modelName: "deepseek-ai/DeepSeekV2"
  rules:
    - name: "default"
      targetModels:
        - modelServerName: "deepseek-v2"
EOF
```

---

## 3. Verification

### 3.1 Check Workloads

Confirm that prefill and decode Pods are up:

```bash
kubectl get modelserving deepseek-v2-lite -n dev -o yaml | grep status -A 10

kubectl get pod -n dev -owide \
  -l modelserving.volcano.sh/name=deepseek-v2-lite
```

You should see both roles in `Running` and `Ready` state.

### 3.2 Test the Chat Endpoint

Once routing is configured, you can send a test request to the Kthena-router:

```bash

export ENDPOINT=$(kubectl get svc kthena-router -n kthena-system --output=jsonpath='{.status.loadBalancer.ingress[0].ip}:{.spec.ports[0].port}')

curl --location "http://${ENDPOINT}/v1/chat/completions" \
  --header "Content-Type: application/json" \
  --data '{
    "model": "deepseek-ai/DeepSeekV2",
    "messages": [
      {
        "role": "user",
        "content": "Where is the capital of China?"
      }
    ],
    "stream": false
  }'
```

A successful JSON response confirms that:

- The prefill and decode services are both running on Ascend NPUs.
- KV transfer between them is working.
- The Kthena routing layer is correctly fronting the vLLM-Ascend plugin.

---

## 4. Cleanup

To remove the deployment:

```bash
# 1. Remove user-facing routing
kubectl delete modelroute deepseek-v2 -n dev

# 2. Remove internal server
kubectl delete modelserver deepseek-v2 -n dev

# 3. Remove workloads
kubectl delete modelserving deepseek-v2-lite -n dev
```

---

## 5. Summary

For more advanced features, please refer to the [Kthena website](https://kthena.volcano.sh/).
