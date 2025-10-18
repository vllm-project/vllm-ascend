# Dynamic Batch

Dynamic batch is a technique that dynamically adjusts the chunksize during each inference iteration within the chunked prefilling strategy according to the resources and SLO targets, thereby improving the effective throughput and decreasing the TBT.

Dynamic batch is controled by the value of the `--SLO_limits_for_dynamic_batch`. 
Notably, only 910 B3 is supported with decode token numbers scales below 2048 so far. 
Especially, the improvements are quite obvious on Qwen, Llama models.
We are working on further improvements and this feature will support more XPUs in the future.

## Getting started


### Prerequisites

1. Dynamic batch now depends on a offline cost model saved in a look-up table to refine the token budget. The lookup-table is saved in '.csv' file, which should be first downloaded and save under `vllm_ascend/core/`

2. `Pandas` is needed to load the look-up table.
    ```bash
    pip install pandas 
    ```

### Tuning Parameter
`--SLO_limits_for_dynamic_batch` is the tunning parameters (integer type) for the dynamic batch feature, greater values impose more constraints on the latency limitation, leading to higher effective throughput. The parameter can be selected according to the specific models or service requirements. 

```python
--SLO_limits_for_dynamic_batch =-1 # default value, dynamic batch disabled.
--SLO_limits_for_dynamic_batch = 0  # baseline value for dynamic batch, dynamic batch disabled, FCFS and decode-first chunked prefilling strategy is used.
--SLO_limits_for_dynamic_batch > 0 # user-defined value for dynamic batch, dynamic batch enabled with FCFS and decode-first chunked prefilling strategy.
```
## Usage
Dynamic batch is used in the online inference.
```shell
vllm serve ${model_directory}\
    --block-size 128 \
    --additional_config '{"SLO_limits_for_dynamic_batch":'${SLO_LITMIT}'}' \
    ... # other parameters
```
