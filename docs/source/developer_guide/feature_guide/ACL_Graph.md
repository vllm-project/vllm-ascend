# ACL Graph

## Why we need ACL Graph?

When in LLM inference, each token requires nearly thousand operator executions, and when host launching operators are slower than device, it will cause host bound. In severe cases, the device will be idle for more than half of the time. To solve this problem, we use graph in LLM inference.

```
eager mode:

host:   |  launch op1  |  launch op2  |  launch op3  |  launch op4  |  launch op5  |

device:                | run op1 |free| run op2 |free| run op3 |free| run op4 |free| run op5 |

        | <-----                           total time                                 -----> |

graph mode:

host:   |  launch graph  |

device:                  | run op1 | run op2 | run op3 | run op4 | run op5 |

        | <-----                    total time                      -----> |

```

## How to use ACL Graph?

ACL Graph is enabled by default in V1 Engine, just need to check that `enforce_eager` is not set to `True`. More details see: [Graph Mode Guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html)

## How it works?

In short, graph mode works in two steps: **capture and replay**. When engine starts, we will capture all of the ops in model forward and save it as a graph, and when req come in, we just replay the graph on devices, and waiting for result.

But in reality, graph mode is not that simple.

### Padding and Bucketing

Due to graph can only replay the ops captured before, without doing tiling and checking graph input, we need to ensure the consistency of the graph input, but we know that model input's shape depends on the request scheduled by Scheduler, we can't ensure the consistency.

Obviously, we can solve this problem by capturing the biggest shape and padding all of the model input to it. But it will bring a lot of redundant computing and make performance worse. So we can capture multiple graphs with different shape, and pad the model input to the nearest graph, which will greatly reduce redundant computing. But when `max_num_batched_tokens` is very large, the number of graphs that need to be captured will also become very large. But we know that when intensor's shape is large, the computing time will be very long, and graph mode is not necessary in this case. So all of things we need to do is:
1. Set a threshold;
2. When `num_scheduled_tokens` is bigger than the threshold, use `eager_mode`;
3. Capture multiple graphs within a range below the threshold;

```
|    graph1    |
|           graph2           |
|                    graph3                    |
|                              graph4                              |    # the threshold

| input1 | pad |    # use graph1
|           input2           |  # don't need pad
|                      input3                      |      pad      |    # use graph4
|                                    input4                                    |    # use eager mode

```

### Piecewise and Full graph

Due to the increasing complexity of the attention layer in current LLM, we can't ensure all types of attention can run in graph. In MLA, prefill_tokens and decode_tokens have different calculation method, so when a batch has both prefills and decodes in MLA, graph mode is difficult to handle this situation.

vLLM solves this problem with piecewise graph mode. We use eager mode to launch attention's ops, and use graph to deal with others. But it also bring some problems: The cost of launching ops has become large again, although much smaller than eager mode, but it will also lead to host bound when cpu is poor or `num_tokens` is small.

Altogether, we need to support both piecewise and full graph mode.

1. When attention can run in graph, we tend to choose full graph mode to achieve optimal performance;
2. When full graph is not work, use piecewise graph as a substitute;
3. When piecewise graph's performance is not good and full graph mode is blocked, separate prefills and decodes, and use full graph mode in **decode_only** situation. Because when a batch include prefill req, usually `num_tokens` will be quite big and not cause host bound.

> Currently, due to stream resource constraint, we can only support a few buckets in piecewise graph mode now, which will cause redundant computing and may lead to performance degradation compared with eager mode.

## How it be implemented?

vLLM has already implemented most of the modules in graph mode. You can see more details at: [CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs.html)

When in graph mode, vLLM will call `current_platform.get_static_graph_wrapper_cls` to get current device's graph model wrapper, so what we need to do is to implement the graph mode wrapper on Ascend: `ACLGraphWrapper`.

vLLM has added `support_torch_compile` decorator to all models, this decorator will replace the `__init__` and `forward` interface of the model class, and when `forward` called, the code inside the `ACLGraphWrapper` will be executed, and it will do capture or replay as mentioned above.

When use piecewise graph, we just need to follow the above-mentioned process, but when in full graph, due to the complexity of the attention, sometimes we need to update attention op's param before execution. So we implement `update_attn_params` and `update_mla_attn_params` funcs for full graph mode. And when forward, memory will be reused between different ops, so we can't update attention op's param before forward. In ACL Graph, we use `torch.npu.graph_task_update_begin` and `torch.npu.graph_task_update_end` to do it, and use `torch.npu.ExternalEvent` to ensure order between params update and ops execution.

## DFX

### Stream resource constraint

Currently, we can only capture 1800 graphs at most, due to the limitation of ACL graph that a graph requires a separate stream at least. This number is bounded by the number of streams, which is 2048, we save 248 streams as a buffer. Besides, there are many variables that can affect the number of buckets:

+ Piecewise graph will divides the model into `num_hidden_layers + 1` sub modules, based on attention layer. Every sub module is a single graph which need to cost stream, so the number of buckets in piecewise graph mode is very tight compared with full graph mode.

+ The number of streams required for a graph is related to the number of comm domains. Each comm domain will increase one stream consumed by a graph.

+ When multi-stream is explicitly called in sub module, it will consumes an additional stream.

There are some other rules about ACL Graph and stream. Currently, we use func `update_aclgraph_sizes` to calculate the maximum number of buckets and update `graph_batch_sizes` to ensure stream resource is sufficient.

We will expand the stream resource limitation in the future.

## Limitation

1. `FULL` and `FULL_AND_PIECEWISE` are not supported now;
2. When use ACL Graph and MTP and `num_speculative_tokens > 1`, as vLLM don't support this case in v0.11.0, we need to set `cudagraph_capture_sizes` explicitly.
3. `use_inductor` is not supported now;
