# Multi Node Test

Multi-Node CI is designed to test distributed scenarios of very large models, eg: disaggregated_prefill multi DP across multi nodes and so on.

## How is works

The following picture shows the basic deployment view of the multi-node CI mechanism, It shows how the github action interact with [lws](https://lws.sigs.k8s.io/docs/overview/) (a kind of kubernetes crd resource)

![alt text](../../assets/deployment.png)

From the workflow perspective, we can see how the final test script is executed, The key point is that these two [lws.yaml and run.sh](https://github.com/vllm-project/vllm-ascend/tree/main/tests/e2e/nightly/multi_node/scripts), The former defines how our k8s cluster is pulled up, and the latter defines the entry script when the pod is started, Each node executes different logic according to the [LWS_WORKER_INDEX](https://lws.sigs.k8s.io/docs/reference/labels-annotations-and-environment-variables/) environment variable, so that multiple nodes can form a distributed cluster to perform tasks.

![alt text](../../assets/workflow.png)

## How to contribute

1. Upload custom weights

   If you need customized weights, for example, you quantized a w8a8 weight for DeepSeek-V3 and you want your weight to run on CI, Uploading weights to ModelScope's [vllm-ascend](https://www.modelscope.cn/organization/vllm-ascend) organization is welcome, If you do not have permission to upload, please contact @Potabk

2. Add config yaml

    As the entrypoint script [run.sh](https://github.com/vllm-project/vllm-ascend/blob/0bf3f21a987aede366ec4629ad0ffec8e32fe90d/tests/e2e/nightly/multi_node/scripts/run.sh#L106) shows, A k8s pod startup means traversing all *.yaml files in the [directory](https://github.com/vllm-project/vllm-ascend/tree/main/tests/e2e/nightly/multi_node/config/models), reading and executing according to different configurations, so what we need to do is just add "yamls" like [DeepSeek-V3.yaml](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/nightly/multi_node/config/models/DeepSeek-V3.yaml).

    Suppose you have **2 nodes** running a 1P1D setup (1 Prefillers + 1 Decoder):

    you may add a config file looks like:

    ```yaml
    test_name: "test DeepSeek-V3 disaggregated_prefill"
    # the model being tested
    model: "vllm-ascend/DeepSeek-V3-W8A8"
    # how large the cluster is
    num_nodes: 2
    npu_per_node: 16
    # All env vars you need should add it here
    env_common:
    VLLM_USE_MODELSCOPE: true
    OMP_PROC_BIND: false
    OMP_NUM_THREADS: 100
    HCCL_BUFFSIZE: 1024
    SERVER_PORT: 8080
    disaggregated_prefill:
    enabled: true
    # node index(a list) which meet all the conditions:
    #  - prefiller
    #  - no headless(have api server)
    prefiller_host_index: [0]
    # node index(a list) which meet all the conditions:
    #  - decoder
    #  - no headless(have api server)
    decoder_host_index: [1]

    # Add each node's vllm serve cli command just like you runs locally
    deployment:
    -
        server_cmd: >
            vllm serve ...
    -
        server_cmd: >
            vllm serve ...
    benchmarks:
    perf:
        # fill with performance test kwargs
    acc:
        # fill with accuracy test kwargs
    ```
