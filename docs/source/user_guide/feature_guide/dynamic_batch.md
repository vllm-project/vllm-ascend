# Dynamic Batch

Dynamic batch is a technique that dynamically adjusts the chunksize during each inference iteration within the chunked prefilling strategy according to the resources and SLO targets, thereby improving the effective throughput and decreasing the TBT.

Dynamic batch is controlled by the value of the `--SLO_limits_for_dynamic_batch`.
Notably, only 910 B3 is supported with decode token numbers scales below 2048 so far.
Especially, the improvements are quite obvious on Qwen, Llama models.
We are working on further improvements and this feature will support more XPUs in the future.

## Getting started

### Prerequisites

1. Dynamic batch now depends on an offline cost model saved in a lookup table to refine the token budget. The lookup table is saved in '.csv' file, which should be first downloaded from [here](https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/dynamic_batch_scheduler/A2-B3-BLK128.csv), renamed, and saved to the path `vllm_ascend/core/profile_table.csv`

2. `Pandas` is needed to load the lookup table, in case `pandas` is not installed.
  
 ```bash
    pip install pandas 
    ```

### Tuning Parameter
`--SLO_limits_for_dynamic_batch` is the tuning parameters (integer type) for the dynamic batch feature, greater values impose more constraints on the latency limitation, leading to higher effective throughput. The parameter can be selected according to the specific models or service requirements.

```python
--SLO_limits_for_dynamic_batch =-1 # default value, dynamic batch disabled.
--SLO_limits_for_dynamic_batch = 0  # baseline value for dynamic batch, dynamic batch disabled, FCFS and decode-first chunked prefilling strategy is used.
--SLO_limits_for_dynamic_batch > 0 # user-defined value for dynamic batch, dynamic batch enabled with FCFS and decode-first chunked prefilling strategy.
```

### Supported Models
So far, dynamic batch performs better on several dense models including Qwen and Llama (from 8B to 32B) with `tensor_parallel_size=8`. For different models, a proper `SLO_limits_for_dynamic_batch` parameter is needed. The empirical value of this parameter is generally `35, 50, or 75`. Therefore, some additional tests are needed to select the best parameter.

## Usage
Dynamic batch is used in the online inference. A fully executable example is as follows:

```shell
SLO_LITMIT=50
vllm serve Qwen/Qwen2.5-14B-Instruct\
    --additional_config '{"SLO_limits_for_dynamic_batch":'${SLO_LITMIT}'}' \
    --max-num-seqs 256 \
    --block-size 128 \
    --tensor_parallel_size 8 \
    --load_format dummy \
    --max_num_batched_tokens 1024 \
    --max_model_len 9000 \
    --host localhost \
    --port 12091 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code
```
