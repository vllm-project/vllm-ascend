# Prefill Decode Disaggregation

## Overview
Prefill & Decode Disaggregation is a built-in feature in vLLM-Ascend designed to decouple the model’s prefill (Prefill) and decode (Decode) processes across different devices or processes, improving throughput and latency performance. By separating producer and consumer roles and utilizing the KV cache for efficient state transfer between nodes, it enables large-scale multi-device distributed inference.

## 

# Prefill & Decode Disaggregation 原理图

```text
                       +--------------------+
                       |                    |
                       |     Client /       |
                       |  OpenAI-compatible |
                       |     Request        |
                       |                    |
                       +--------------------+
                                 |
                                 v
                       +--------------------+
                       |                    |
                       |      Proxy         |
                       | (p2p_disagg_proxy) |
                       |                    |
                       +--------------------+
                          |            |
         Prefill Request  |            |  Decode Request
                          v            v
                  +----------------+   +----------------+
                  |                |   |                |
                  |   Prefill      |   |     Decode     |
                  |  (KV Producer) |   |  (KV Consumer) |
                  |                |   |                |
                  +----------------+   +----------------+
                          |                  ^
                          | KV Cache         | Use KV Cache
                          | Transfer         |
                          +------------------+
```
- **Client**  
  The user or application that sends a standard `v1/completions` request.

- **Proxy**  
  Receives the user request, dispatches it to the Prefill and Decode nodes, and aggregates the final result.

- **Prefill Instance**  
  - Performs KV cache generation, which involves the forward pass of the Transformer (embedding + attention + KV output).  
  - Acts as the `kv_producer`, and transfers the cache to the Decode node via the `AscendSimpleConnector`.

- **Decode Instance**  
  - Performs token-by-token decoding using the KV cache received from the Prefill node.  
  - Acts as the `kv_consumer`, completing the final text generation.
