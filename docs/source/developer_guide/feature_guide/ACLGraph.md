# ACLGraph

## Why we need ACLGraph?

When in LLM inference, each token requires nearly thousand operator executions, and when CPU launching operators are slower than GPU, it will cause host bound. In severe cases, the GPU will be idle for more than half of the time. To solve this problem, we use graph in LLM inference.

```
eager mode:

cpu: |  launch op1  |  launch op2  |  launch op3  |  launch op4  |  launch op5  |

gpu:                | run op1 |free| run op2 |free| run op3 |free| run op4 |free| run op5 |

     | <-----                           total time                                 -----> |

graph mode:

cpu: |  launch graph  |

gpu:                  | run op1 | run op2 | run op3 | run op4 | run op5 |

     | <-----                    total time                      -----> |

```

## How to use ACLGraph?

ACLGraph is enabled by default in V1 Engine, just set to use V1 Engine is enough.

## How it works?

In short, graph mode works in two steps: **capture and replay**. When engine starts, we will capture all of the ops in model forward and save it as a graph, and when req come in, we just replay the graph on gpus, and waiting for result.

But in reality, graph mode is not that simple.

### Padding and Bucketing

Due to graph can only replay the ops captured before, without doing tiling and checking graph input, so we need to ensure the consistency of the graph input, but we know that model input's shape depends on the request scheduled by Scheduler, we can't ensure the consistency.

Obviously, we can solve this problem by capturing the biggest shape and padding all of the model input to it. But it will bring a lot of redundant computing and make performance worse. So we can capture multiple graphs with different shape, and pad the model input to the nearest graph, it will greatly reduce redundant computing, but when `max_num_batched_tokens` is very large, the number of graphs that need to be captured will also become very large. But we know that when intensor's shape is large, the computing time will be very long, and graph mode is not necessary in this case. So all of things we need to do is:
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

## How it be implemented?

vLLM has already implemented most of the modules in graph mode, and when in graph mode, vLLM will call `current_platform.get_static_graph_wrapper_cls` to get current device's graph model wrapper,so what we need to do is to implement the graph mode wrapper on Ascend: `ACLGraphWrapper`.

vLLM has added `support_torch_compile` decorator to all models, this decorator will replace the `__init__` and `forward` interface of the model class, and when `forward` called, the code inside the `ACLGraphWrapper` will be executed, and it will do capture or replay as mentioned above.

When use piecewise graph, we just need to follow the above-mentioned process, but when in full graph, due to the complexity of the attention, sometimes we need to update attention op's param before execution. So we implement `update_attn_params` and `update_mla_attn_params` funcs for full graph mode. And when forward, memory will be reused between different ops, so we can't update attention op's param before forward. In ACLGraph, we use `torch.npu.graph_task_update_begin` and `torch.npu.graph_task_update_end` to do it, and use `torch.npu.ExternalEvent` to ensure order between params update and ops execution.

## DFX

### Stream resource constraint

When use piecewise graph mode, every sub module will use at least one stream in ACLGraph. Due to stream resource constraint, the number of bucketing will be restricted.

Currently, we calculate the maximum number of bucketing under the current case through a formula:

```python
    # NOTE: Currently, we can only capture 1800 graphs at most,
    # due to the limitation of ACL graph. This number is bounded by
    # the number of streams, which is 2048, we save 248 streams
    # as a buffer.
    # Maximum number of graphs that can be captured by ACL Graph
    # TODO: Find out whether we need to solve allreduce function
    MAX_CAPTURE_SIZE = 1800

    # Store original configuration and temporarily clear it
    compilation_config = vllm_config.compilation_config
    original_sizes, compilation_config.cudagraph_capture_sizes = \
        compilation_config.cudagraph_capture_sizes, None

    # Calculate parallel configuration factor
    hf_config = vllm_config.model_config.hf_config
    if hasattr(hf_config, 'num_hidden_layers'):
        num_hidden_layers = hf_config.num_hidden_layers
    else:
        num_hidden_layers = get_max_hidden_layers(hf_config)
    parallel_config = vllm_config.parallel_config

    # Calculate maximum supported batch sizes considering model architecture
    resources_per_graph = num_hidden_layers + 1
    if vllm_config.speculative_config is not None:
        draft_model_hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        resources_per_graph += draft_model_hf_config.num_hidden_layers + 1

    # TODO: Find out whether we need to take into account the pp_size
    num_comm_groups = sum(size > 1 for size in [
        parallel_config.data_parallel_size,
        parallel_config.tensor_parallel_size,
    ])

    if os.getenv("HCCL_OP_EXPANSION_MODE") == 'AIV':
        # TODO: Find out whether we need to take into account the pp_size
        parallel_factor = 1 + num_comm_groups + int(
            parallel_config.enable_expert_parallel) + int(
                vllm_config.additional_config.get(
                    "multistream_overlap_shared_expert", False))
        if is_moe_model(vllm_config):
            parallel_factor += (parallel_config.data_parallel_size > 1)
        else:
            # When AIV mode is enabled, the allreduce operator of the dense
            # layer model will occupy additional streams, which are buffered here.
            MAX_CAPTURE_SIZE = MAX_CAPTURE_SIZE - parallel_factor * resources_per_graph

        # Calculate maximum supported batch sizes considering model architecture on the A2 Hardware Device
        # Assume the following case:
        # MAX_CAPTURE_SIZE = 1920, num_hidden_layers = 48, data_parallel_size is 1, tensor_parallel_size is 4,
        # According to the formula, max_num_batch_sizes = math.floor(1920 / (48 + 1) / 2) = 19
        max_num_batch_sizes = math.floor(MAX_CAPTURE_SIZE /
                                         resources_per_graph / parallel_factor)
        logger.info(
            "Calculated maximum supported batch sizes for ACL graph: %s",
            max_num_batch_sizes)
    else:
        # The above describes an empirical formula applicable to the A2 hardware.
        # Under this configuration, HCCL employs the FFTS+ method for execution unfolding,
        # which adds only 1 concurrent stream without consuming collective communication execution unfolding streams.
        # On A3 hardware, HCCL defaults to the AICPU method.
        # This approach may additionally allocate up to rank_size (max 16) - 1 streams per collective communication domain on the device (worst case).
        # Using the default collective communication unfolding method on A3 will lead to a significant reduction in the maximum supported sizes.
        # Therefore, the calculation formula has been modified as follows:
        # Assume the following case:
        # MAX_CAPTURE_SIZE = 1920, num_hidden_layers = 48, data_parallel_size is 1, tensor_parallel_size is 4,
        # According to the formula, max_num_batch_sizes = math.floor((1920 - 1 * 40) / (48 + 1) / (1 + 1 * 2)) = 12
        max_num_batch_sizes = math.floor(
            (MAX_CAPTURE_SIZE - num_comm_groups * 40) / resources_per_graph /
            (1 + num_comm_groups * 2))
        logger.info(
            "Calculated maximum supported batch sizes for ACL graph: %s",
            max_num_batch_sizes)
        logger.warning(
            "Currently, communication is performed using FFTS+ method, which reduces "
            "the number of available streams and, as a result, limits the range of runtime "
            "shapes that can be handled. To both improve communication performance and "
            "increase the number of supported shapes, set HCCL_OP_EXPANSION_MODE=AIV."
        )
```

We will expand the stream resource limitation in the future.

## Limitation

1. `FULL_AND_PIECEWISE` is not supported now;
2. When use ACLGraph and MTP and `num_speculative_tokens > 1`, as vLLM don't support this case in v0.11.0, we need to set `cudagraph_capture_sizes` explicitly.
3. `use_inductor` is not supported now;
