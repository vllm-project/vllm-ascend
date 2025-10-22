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

## DFX



## Limitation

