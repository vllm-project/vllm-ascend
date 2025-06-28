# ModelRunner_prepare_inputs

## Before Start:
### Some coventions
In `ModelRunner class`(`NPUModelRunner` or `GPUModelRunner`), typically there are two corresponding tensors stored on the **CPU** (`self.xxx_cpu`) and **device** (`self.xxx`). Generally, we process data on the CPU first, and then copy the processed CPU-side variables to the device. We also frequently convert CPU-side tensors to **NumPy arrays** `(self.xxx_np)` for convenient processing. In terms of memory view, these two types of variables (CPU tensors and NumPy arrays) **share the same memory space**, meaning an operation on one is equivalent to an operation on both. However, in `gpu_input_batch.py` , the `self.token_ids_cpu_tensor` is `tensor`, `self.token_ids_cpu` is `numpy ndarray`. The different naming conventions across various `Python modules` can indeed be confusing (Me too!!!!😭), so we should carefully distinguish them.
### Create an example for better understanding
Let’s assume that in the current model's forward step, 5 requests are scheduled. Requests `0` and `1` are in the decode phase, requests `2` and `3` are undergoing full prefill, and request `4` is in chunked prefill. The scheduled tokens for these requests are `1`, `1`, `93`, `75`, and `30`, respectively. So the number of total scheduled tokens are `1 + 1 + 93 + 75 + 30 = 200` (for more detail, let’s say `200` is the initial token budget in `scheduler.py`, you can read the code of `scheduler.py` later or before, either is fine, it won't affect your understanding of what follows.) And let’s say that the `request 0` and `request 1` already computed `54`, `145` tokens respectively. And we assume the `block_size = 16` i.e. one block will contain 16 tokens (vllm page attention mechanism), the `max_model_len = 240`, `max_num_blocks_per_req = ceil(240/16)= 15`. We assume the LLM has `32` self-attention layers.

## Different level view for important variables.
1. `request level`: in this level, for example, variable `scheduler_output.num_scheduled_tokens` may like: `{ '0' : 1, '1' : 1, '2' : 93, '3' : 75, '4' : 30}` its length is number of request, in this case, there are total 5 requests. And also, variable `num_scheduled_tokens` is also in `request level` but its the `numpy ndarray` form, not the `dict` form, just like: `np.array(list(scheduler_output.num_scheduled_tokens.values()))`: `[1, 1, 93, 75, 30]` making the `str` index to `int` index (implicit). `num_scheduled_tokens` totally have `5` elements, equal to the number of scheduled requests, so it at request level.
    - Important `request level` variable overview:
        - `num_scheduled_tokens` , `cu_num_tokens` , `self.input_batch.num_computed_tokens_cpu` , `self.query_start_loc_np` , `self.seq_lens_np` .
2. `token level`: in this level, for example, variable `req_indices` : `[0, 1, 2, 2, …,2, 3, 3, …, 3, 4, 4, …, 4]`. It indicates all the scheduled tokens belong to which request. Continuing from the previous example, requests `0` and `1` each have `1` token. There are 93 tokens belong to request 2, and so on. `req_indices` totally have `1 + 1 + 93 + 75 + 30 = 200` tokens (i.e. the number of scheduled tokens), so it at token level.
    - Important `token level` variable overview:
        - `req_indices`, `arange` , `self.positions_np` , `token_indices` , `self.input_ids` , `block_table.slot_mapping_np` , `block_table_indices`, `block_offsets` .
3. `System level` : In this level, most variables are organized into table structures, such as the **`block table`** and the **`token_ids table`**. The **`block table`** records the mapping of a request's block logical addresses to their physical addresses in HBM which is a core data structure for page attention. The **`token_ids table`** records the corresponding `token_id` for each token's logical position within each request. These `token_ids` are obtained from a `tokenizer` , which are then converted into `embeddings` via an embedding table.
    - Important system level variable overview:
        - `self.input_batch.token_ids_cpu_tensor` (i.e. `token_ids table`), `block_table`.

## Go throgh the _prepare_inputs() method:
Here I would use the example I created before to go through the `_prepare_inputs()` method in in `ModelRunner`(`NPUModelRunner` or `GPUModelRunner`):

### Prepare basic useful variables
- `total_num_scheduled_tokens: int = scheduler_output.total_num_scheduled_tokens` .This is the total number of tokens currently scheduled (including all requests).
    - It equal to `1 + 1 + 93 + 75 + 30 = 200`
- `scheduler_output.num_scheduled_tokens` is a `dict` map the `req_ids(str)` to its corresponding `number of scheduled tokens`.
    - It equal to `{ '0' : 1, '1' : 1, '2' : 93, '3' : 75, '4' : 30}`. (**request level**)
- `num_scheduled_tokens` is a `ndarray` which is the is the `ndarray` form of `scheduler_output.num_scheduled_tokens`, just like flatten the `dict.values()` to a `list` i.e. ignore its key field.
    - It equal to `array([1, 1, 93, 75, 30])`. (**request level**)
- `num_input_tokens =`
    - `total_num_scheduled_tokens` if in eager mode.
    - `self.vllm_config.pad_for_cudagraph(total_num_scheduled_tokens)` if using graph mode.
    - In this example, I will only introduce eager mode. so `num_input_tokens = total_num_scheduled_tokens = **200**`
- `num_reqs = self.input_batch.num_reqs` the `self.input_batch.num_reqs` may be updated from `_update_states()` . It is the number of the scheduled request in this model forward step.
    - It equal to `5` : `0, 1, 2, 3, 4`.
- `req_indices` : The shape of `req_indices` is `(total_num_scheduled_tokens,)`, where each token position mapping to which request the token belong to, and the requests are represented by `int` rather than `str`.
    - It equal to `[0, 1, 2, 2, …,2, 3, 3, …, 3, 4, 4, …, 4]` where `2` repeats `93` times, `3` repeats `75` times, `4` repeats `30` times. (**token level**)
- `_get_cumsum_and_arange()` : Get the **cumulative sum** `cu_num_tokens` and **batched arange** `arange` of the given array, i.e. accumulate different requests and set an `arange` for every request. The results of `_get_cumsum_and_arange()`: `cu_num_tokens` and `arange`:
    - `cu_num_tokens` equals to `[1, 2, 95, 170，200]`. (**reqest level**)
    - `arange` equals to `[0, 0, 0, 1, 2, ..., 92, 0, 1, ..., 74, 0, 1, ..., 29]`. (**token level**)
- `positions_np = self.positions_np[:total_num_scheduled_tokens]` at CPU end. The `positions of each token = computed_tokens + arange`. So its the token relative position for each individual request.
    - It is equal to `[54, 145, 0, 1, 2, …, 92, 0, 1, 2, …, 74, …, 0, 1, 2, …, 29]` . Note that the `request 0` and `request 1` already computed `54 (0~53)` and `145 (0~144)` tokens before. (**token level**)
- `positions = self.positions[:num_input_tokens]` at Device end (GPU or NPU) which is copied from the CPU (via `self.positions_cpu`) to the device.

### Prepare input_ids
- `token_indices` represents indices of `token_ids_cpu` for every scheduled token. It will select the correspoding `token_ids` in the `token_ids_cpu`. It is derived from the `positions` and `req_indices * max_model_len`. Its shape is `(total_num_scheduled_tokens, )`
    - Using `M=max_model_len` , then it equal to `[54, 145 + M, 0 + 2 * M, 1 + 2 * M, …, 92 + 2 * M, 0 + 3 * M, 1 + 3 * M, …, 74 + 3 * M, 0 + 4 * M, …, 29 + 4 * M]` (**token level**)
- `self.input_batch.token_ids_cpu`: Its shape is `(self.max_num_reqs, self.model_config.max_model_len)`. Unlike the block-based `block_table` in `self.input_batch` (which has shape `(max_num_reqs, max_num_blocks_per_req)`). Note that `max_num_blocks_per_req = ceil(max_model_len / block_size)`. (**system level**)
    - Although `token_ids_cpu` and `block_table` are somewhat similar in shape and organization, they play different roles.
        1. `token_ids_cpu` stores `token_id` (`token_id` is the index of the **vocabulary** that the large model currently using, every `token_id` representing a unique token, later can be convert to word `embedding` using `embedding_table`):For example, the value of `token_ids_cpu[0][1]` is the `token_id` of the second token in `request 0`. 
            - For the above example, we will use the `token_ids_cpu[0][54], token_ids_cpu[1][145], token_ids_cpu[2][0~92], token_ids_cpu[3][0~74], token_ids_cpu[4][0~29]`. But by **flattening** the 2D array, `token_ids_cpu`, into a one-dimensional array, we can then use the `token_indices` we just got before to directly pick out the `token_id`s we want. **Note:** these selected `token_ids` serve as the input for our model's forward pass, we refer to them as **`input_ids`**.

            Example of **vocabulary**:
            ```
            | Token ID     | Token         | 
            |--------------|---------------|
            | 0            | [PAD]         |
            | 1            | <|endoftext|> |
            | 2            | <|start|>     |
            | 3            | [SEP]         |
            | 4            | try           |
            | 5            | the           |
            | 6            | be            |
            | 7            | of            |
            | 8            | and           |     
            | ...          | ...           |     
            | ...          | ...           |
            | vocab_size-1 | <|im_end|>    |
            ```
        2. The role of `block_table` is to record the **physical block ID** in `HBM` (more specifically, the second dimension index of `self.kv_caches[0~31]`, I will talk this later) for each block in a request. For example, the value of `block_table[1][2]` represents the index of the third block of `request 1` in HBM kv_caches.
- `self.input_ids_cpu`'s valid value shape is also `(total_num_scheduled_tokens, )` . The elements in `self.input_ids_cpu` are the selected `token_ids` from the `token_ids_cpu` using the `token_indices`.
    - the element type typically is `torch.int32` , And its value ranges from `0` to **`vocabulary_size - 1`** . Because the value of the elements is corresponding to the actual tokens, so I will make up a fake `self.input_ids_cpu`: `[2039, 8347, 1923, …,273, 8211, …, 48, 293, …, 21, …, 104]` there are total 200 `input_ids` (i.e. **`vocabulary indices`**) and each of these indices represents its corresponding token. (**token level**)
- `self.input_ids`: this is consistent with `self.input_ids_cpu` but in the `gpu or npu` end.

### Calculate the slot mapping
- `block_table_cpu` is the CPU end `block_table` which is a tensor and shape is `(max_num_reqs, max_num_blocks_per_req)`.
- `self.kv_caches: list[torch.Tensor] = []` is a list, which record each attention layer's actual kv_cache tensors. E.g. `self.kv_caches[1]` is the second attention layer's kv_cache tensor. In common case, we typically use full self-attention layer in text-only model, then kv_cache tensor's shape are all same in every self-attention layer. And if the model have total 32 layers, then `len(self.kv_caches) = 32`.
    - The shape of `self.kv_caches[i]` (`i` ranges form `0` to `31`) may equal to `torch.Size([2, 5146, 16, 4, 128])`, where `2` means there are 2 caches, one for k_cache and another for v_cache, `5146` means there are `5146` available kv_cache blocks for each attention layer, which is derived from the `memory size of your device` (gpu or npu) and the `block_size` (for more detail, please refer to initialize_kv_cache() method), `16` is the block_size, `4` means `num_kv_heads` (for more detial, please refer to Grouped Query Attention, GQA), `128` means the `head_size` of each head.
- Every element in `block_table` is a `physical block_id` in the `HBM` which totally has `num_gpu_blocks` blocks. More detail, every `physical block_id` is the index of `self.kv_caches[i]` along second dimension. We will reserve the `kv_cache_block 0` as a dummy block for indicating we didn't allocated the block yet in `block_table`. We will use the block 1 as the first allocated `kvcache_block`.
    - At this moment, we assume the `block_size` is 16, so request `0`, `1`, `2`, `3`, `4` occupies `ceil(54/16)=4`, `ceil(145/16)=10`, `ceil(93/16)=6`, `ceil(75/16)=5`, `ceil(30/16)=2` blocks respectively. And we assume there is not Auto prefix caching and the `self.kv_cache_manager.allocate_slots` in `scheduler.py` allocated the `KVCacheBlock` sequentially to these requests, and the `max_num_blocks_per_req = ceil(max_model_len / block_size) = 15` ,note that the `block 0` is a dummy block we discussed before, then the `block_table` equal to: (**system level**)
        ```
        | 1  | 2  | 3  | 4  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  |
        | 5  | 6  | 7  | 8  | 9  | 10 | 11 | 12 | 13 | 14 | 0  | 0  | 0  | 0  | 0  |
        | 15 | 16 | 17 | 18 | 19 | 20 | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  |
        | 21 | 22 | 23 | 24 | 25 | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  |
        | 26 | 27 | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  |
        | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  |
        ...
        ...
        | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  |
        ```
    Every element in this `block_table` is a index of `self.kv_caches` along the second dimension. So currently, the HBM (`self.kv_caches`) has `27` valid blocks (excluding dummy block).
- `block_table_indices = (req_indices * self.max_num_blocks_per_req + positions_np // self.block_size)` :
    - Each scheduled token has a related block in `block_table`. Note that these indices are make scense only after the `block_table` was flattened. So it equal to `[0 + 3, 15 + 9, 30 + 0, 30 + 0, ..., 30 + 5, 45 + 0, 45 + 0, ..., 45 + 4, 60 + 0, 60 + 0, ..., 60 + 1] = [3, 24, 30, 30, ..., 35, 45, 45, ..., 49, 60, 60, ..., 61]`. (**token level**)
- `block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()`. Every scheduled tokens corresponding to a `kvcache_block_tensor` in `HBM` (`self.kv_caches[i]`). And the `block_numbers` used to map each token to its own `kvcache_block_tensor` in `self.kv_cahes[i]` (more specifically, the element of `block_numbers` is the index of `self.kv_caches[i]` along second dimension we discussed before). We access the related `kvcache_block_id` for each token using `block_table_indices`.
    - It equals to `[4, 14, 15, 15, ..., 20, 21, 21, ..., 25, 26, 26, ..., 27]` (**token level**)
- `block_offsets = positions_np % self.block_size`: `block_offsets` is the token relative position **inside** each block, its values are form `0` to `self.block_size - 1`.
    - It equals to `[6, 1, 0, 1, ..., 12, 0, 1, ..., 10, 0, 1, ..., 13]` (**token level**)
- `self.slot_mapping = block_numbers * self.block_size + block_offset` this is the physical token position in the `self.kv_caches[i]`.
    - It equals to `[70, 225, 240, 241, ..., 332, 336, 337, ..., 410, 416, 417, ..., 445]`

#### Tips:
There is also a subtle difference between the `self.input_batch.token_ids_cpu` and `self.input_batch.block_table[0].block_table` (In vLLM, different **attention strategies** like Full-attention, SlidingWindow-attention, and Mamba are distinguished by using different **`block_table` groups**. Here, assuming our large language model primarily uses **Full-attention** (which is typically the case), we would utilize `block_table[0]`. For more detailed information, please refer to `kv_cache_interface.py`.) :
- **`token_ids_cpu` is a variable that exists only on the CPU, with no corresponding variable on the device. In contrast, `block_table` exists on both ends.** This is because we only perform `tokenize` and `detokenize` (including `token` to `token_id`) operations on the `CPU` end. On the device, we would use `block_table` to convert the logical addresses of blocks to their corresponding physical addresses.

### Prepare the attention metadata.
- `self.query_lens` is `num_scheduled_tokens` tensor form which is not used in `model_runner_v1.py`.
- `self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens` will indicate the index in the `self.position` where the first token of each request is scheduled.
    - It equals to `[0, 1, 1, 2, 95, 170，200]`. (**request level**, but noting that there is an additional element in the last denoting the end position of the last query)
- `self.seq_lens_np` is the request sequences length after this model forward step, which equal to `computed tokens + this time scheduled token`.
    - It equals to `[55, 146, 93, 75, 30]`. (**request level**)
- `seq_lens = self.seq_lens_cpu[:num_reqs]`

#### Making attn_metadata for NPUModelRunner:
- `attn_mask` this is the attention mask for the request.
    - For `PrefillNoCache` situation:
        - Will create one mask matrix for all request: it just using one `max_seq_len * max_seq_len` mask matrix to serve all the scheduled requests.
    - For  `DecodeOnly`:
        - Will not create attention mask.
    - For `ChunkedPrefill`:
        - Will create `mask vector` for the `decode request`, and create `mask matrix` for the `prefill request`. For simplicity, we can first create a `max_seq_len * max_seq_len` mask matrix first, and then using the `position` to select one `row vector` (i.e. `mask vector` for `decode request`) or many `row vector`s (i.e. `mask matrix`). So for each `scheduled request`, there is either a `mask vector` or a `mask matrix`. These `mask vector` and `mask matrix` seem like concatenated together to form a whole `mask tensor`. `max_seq_len = max(seq_lens)` So if the `max_seq_len` is shorter than the preallocated `self._seq_len_cached` then the concatenated tensor’s shape is `(num_input_tokens, max_seq_len)`. `num_input_tokens` is explained before. It is worth noting that during the `decode phase`, if speculative decoding is not used, the created `vector` is actually `an all-zero vector`. This is because the last token can always attend to all previous tokens, meaning that all positions including itself and those before it are visible. (If a position is not visible, its corresponding value would be filled with -INF). 
    - For `PrefillCacheHit` :
        - This is only work when `ascend_scheduler_config.enabled` is True, we will not include this case because we only discuss the `vllm default scheduler` for now.

- `attn_metadata` is a `dataclass` which has the necessary information for model forward. Some important attributes `attn_metadata` of are: `attn_mask` , `attn_state`, `block_tables`, `max_query_len`, `num_actual_tokens(i.e. total_num_scheduled_tokens)`, `num_input_tokens(may be padded for graph mode)`, `query_lens`, `query_start_loc`, `seq_lens`, `slot_mapping`. They are already described above.

### Multimodal model related
- If this model is `multimodal model`:
    - `self._execute_mm_encoder(scheduler_output)` will run the encoder part (e.g. the VIT part in Qwen2.5 vl) of the mm_model. And then store the output of encoder. It is
    - `self._gather_mm_embeddings()` will retrieve the `multimodal embeddings` corresponding to the current scheduled requests (which were just computed by `self._execute_mm_encoder(scheduler_output)`) from `self.encoder_cache`, resulting in the `mm_embeds: list[tensor]`.
- Also assume that we are using `multimodal model`: collect these `multimodal embeddings` we should convert the `place_holder_input_ids` to the actual embeddings:
    - If this model is `multimodal model`:
        - We will use the `input_ids` and `mm_embeds` to generate the correct `inputs_embeds` (`input_ids` are converted to `embeddings` through the `embedding_table`).  Specifically, we will use the `self.model.get_input_embeddings(input_ids, mm_embeds)` function. This function first converts the `input_ids` into `inputs_embeds`. If the current `mm_embeds` is not empty, it indicates the presence of `multimodal input`. In this case, the converted `inputs_embeds` will contain `placeholder embeddings` generated through `placeholder input_ids`. The system then identifies whether these `inputs_embeds` include `placeholder embeddings` (instead of directly checking if `embeddings` in the `inputs_embeds` equal the `placeholder embeddings`, it achieves this by recognizing the `placeholder_token_id` within the `input_ids` and recording their indices. The replacement of `embeddings` is subsequently performed directly via these indices).
        - At last, we will get `inputs_embeds` from the `self.input_embeds` . And we will set the `input_ids` to `None` for directly passing `inputs_embeds` to the model, since the step of converting `input_ids` to `input_embeds` has already been performed.
    - If this model is text-only model:
        - We will just use the `input_ids` as the model input. Because it is possible to include the `embedding layer` to the `ACL graph or CUDA graph` for improving performance.
- If using `mrope` specifically designed for `Qwen2.5 vl`, then `self.mrope_positions` will be utilized in place of `positions` as input to the model.

## At last
Thank you very much for reading! I hope this article has provided you with some insights. After you gain a deeper understanding of vllm, we also hope you can contribute code, documentation for relevant features, help promote the project, and participate in building the vllm open-source community.