<h1 align="center">
Omni Proxy
</h1>

<!-- ## Omni Proxy is A Nginx Enforced Proxy for P/D Disaggregation LLM Inference 

This guide describes how to build and configure Nginx-based Omni Proxy dynamic modules. -->

<!-- ![design](./img/global_proxy_design.png) -->

## Key Advantages over Global Proxy

Omni Proxy represents a next-generation evolution of Global Proxy, specifically engineered for the unique demands of LLM serving.

-   **Next-Generation LLM Proxy:** A significant upgrade over Global Proxy, featuring ***APC-aware*** and ***metrics-aware*** scheduling for prefill and decode phases. The architecture is fundamentally designed around the ***periodic workload patterns*** of LLMs.
-   **Request Delayed Release:** Implements a novel ***request holding mechanism*** within the proxy layer (e.g., Nginx), enabling intelligent request queuing and delayed dispatch to backend instances.
-   **Flexible Framework Integration:** Supports ***different scheduling mode*** required by different inference frameworks (e.g., sequential scheduling in vLLM and parallel scheduling in sglang), while enabling ***tokenization result reuse*** in downstream inference frameworks.

![Omni Proxy Architecture](https://foruda.gitee.com/images/1758272742382062862/5c8fea19_14535041.png )

## Core Principles for Intelligent LLM Scheduling:

Omni Proxy leverages three key principles for intelligent LLM scheduling:
### 1. Endogenous Performance Metrics
Omni Proxy continuously collects real-time metrics at two levels:
-   **Request-level:** Prompt tokens, decoded tokens, prefix-match score, the timestamps of each stages in Omni Proxy.
-   **Instance-level:** the number of running requests, the num of batch tokens and the execution times for each batch.


### 2. Inference Periodicity
Leveraging the cycle-based periodicity of LLM inference, Omni Proxy records the historical execution states (i.e. endogenous performance metrics) of each upstream to maintaine an online prediction model and uses it to predict the expected available time of every instance. Based on these predictions, requests are accurately dispatched to the most suitable upstream, ensuring precise scheduling and reduced waiting time.

### 3. APC-Aware and Metrics-Aware Coordination
Omni Proxy integrates with the inference engine's Prefix Cache for cache matching and scheduling.

-   **Prefill Stage:** Requests are prioritized based on their prefix-match score to maximize cache hits and the prefix-match score is obtained by the radix trees maintained in Omni Proxy. Load balancing is achieved by redirecting requests from overloaded nodes to alternatives while still preserving cache benefits.
-   **Decode Stage:** Scheduling is optimized by estimating the total workload of a request (including prompt tokens and expected completion tokens ). A Longest-Process-Time-First (LPT) policy is used to maximize throughput and minimize execution bubbles in the case of varied-length prompts.



## How to Quickly Use Omni Proxy

```bash
# Compile the modules
bash build.sh

# set the PYTHONHASHSEED
export PYTHONHASHSEED=123

# Generate the nginx.conf for Omni Proxy
bash omni_proxy.sh \
  --nginx-conf-file /usr/local/nginx/conf/nginx.conf \
  --start-core-index 0 \
  --core-num 4 \
  --listen-port 8080 \
  --prefill-endpoints 127.0.0.1:8001,127.0.0.1:8002 \
  --decode-endpoints 127.0.0.1:9001,127.0.0.1:9002 \
  --omni-proxy-pd-policy sequential \
  --omni-proxy-model-path /path/to/DeepSeek
```
Note: if you want to turn on the APC-aware scheduling, 
* Set `ENABLE_APC_EVENT=1` in `omniinfer/tools/scripts/pd_run.sh` and also set `USE_OMNI_PROXY=1` in ansible template file
* Provide the model path `--omni-proxy-model-path /path/to/DeepSeek` to `omni_proxy.sh`.