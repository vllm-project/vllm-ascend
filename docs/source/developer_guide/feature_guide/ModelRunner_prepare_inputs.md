# Prepare inputs for model forwarding

## Purpose
What information should we have in order to perform model forward pass?
 - the inputs
 - the corresponding attention metadata of the inputs

The following diagram shows what we should prepare for the model inference.

```
              +---------------+
  inputs  --> |               |
              |     model     |  --> output
attn_meta --> |               |
              +---------------+  
```

Therefore, as long as we have these two pieces of information mentioned above, we can perform the model's forward propagation.

This article will explain **how we obtain the inputs and their corresponding attention metadata** which are on the left part of above diagram.

## Overview
### 1. Obtain inputs
The workflow of obtain inputs:
1. Get `token positions`: The relative position of each token within its request sequence.

2. Get `token indices`: the index of each scheduled token in the token table.

3. Get `Token IDs`: Using token indices to retrieve the Token IDs from **token id table**.

At last, these `Token IDs` required to feed into the model, and also, `positions` should be send into model to create `Rope` (Rotary positional embedding). Both of them are the inputs of a model.

**Note**: because the `Token IDs` is the inputs of the model, so we will call it `Inputs IDs`
### 2. Build inputs attention metadata
The model requires these attention metadata during the forward pass:
- `query start location`: represents the start and end location of each request corresponding to the scheduled tokens.
- `sequence length`: the length of each request including both computed tokens and newly scheduled tokens.
- `number of computed tokens`: the number of computed tokens for each request.
- `number of requests`: the number of requests in this batch.
- `number of tokens`: Total number of scheduled tokens in this batch.
- **`block table`**: translates the logical address (within its sequence) of each block to its global physical address in the device's memory.
- `max query len`: the longest scheduled tokens length in this requests batch.
- `slot mapping`: the indices of each token that input token will be stored into.
- `attention mask`: The mask matrix applied to attention scores before softmax to control which tokens can attend to each other. (usually a causal attention)

## Before start
There are mainly three types of variables.
- token level: represents one attribute corresponding to each scheduled token, so the length of this variable is the number of scheduled tokens
- request level: represents one attribute of each scheduled request, which length usually is the number of scheduled requests. (`query start location` is a special case, which has one more element)
- system level:
  1. **Token IDs table**: store the token ids (i.e. the inputs of the model) of each request. The shape of this table is `(max num request, max model len)`. Here, `max num request` is maximum count of concurrent requests allowed in a forward batch and `max model len` is the max token count can be handled at one request sequence in this model.
  2. **Block table**: translates the logical address (within its sequence) of each block to its global physical address in the device's memory. The shape of this table is `(max num request, max model len / block size)`

**Note**: How were these two tables formed?
- Both of them are come from the `_update_states` method before **prepare inputs**. You can take a look if you need more inspiration.

### Tips
What is `Token ID`?
For simple, a `token ID` is an **integer** (usually `int32`), which represents a token.
example of `Token ID`:

```
| Token ID     | Token         | 
|--------------|---------------|
| 0            | [PAD]         |
| 1            | <|endoftext|> |
| 2            | <|start|>     |
| 3            | [SEP]         |
| 4            | I             |
| 5            | the           |
| 6            | be            |
| 7            | of            |
| 8            | and           |     
| ...          | ...           |     
| ...          | ...           |
| vocab_size-1 | <|im_end|>    |
```

## Go through details
Make a simple example, assumption:
- max tokens can be scheduled at once: 10.
- `block size`: 2
- Totally schedule 3 requests. Their prompt lengths are 3, 2, and 8 respectively.
- `max model length`: 12 (the max token count can be handled at one request sequence in this model).

These assumption are configured in the beginning when starting the vllm. They are not fixed, so you can manually set them.
### Step 1: All requests in the prefill phase

#### Obtain inputs
Due to the max schedule token count limitation is 10, The scheduled token of each request: `{'0': 3, '1': 2, '2': 5}`. Note that the `request_2` is in chunked prefill, still has 3 prompt tokens not be scheduled.

##### 1. Get token positions:
First, find out each token belong to which request: the 0~2 tokens belong to request_0, 3~4 tokens belong to request_1 and 5~9 tokens belong to request_2. So, we can use `request indices` to point out each token belongs to which request. `request indices`: `[0, 0, 0, 1, 1, 2, 2, 2, 2, 2]`

For each request, use **the number of tokens already computed** + **the relative position in current scheduled tokens**: `request_0: [0 + 0, 0 + 1, 0 + 2]`, `request_1: [0 + 0, 0 + 1]`, `request_2: [0 + 0, 0 + 1,..., 0 + 4]` and then concat them together: `[0, 1, 2, 0, 1, 0, 1, 2, 3, 4]`. Note: there is more efficient way (using `request indices`) to create positions in actual code.

Finally, `token opsitions` is `[0, 1, 2, 0, 1, 0, 1, 2, 3, 4]`. This variable is **token level**

##### 2. Get token indices:
Current **Token IDs table**, which shape is `(max num request, max model len)`.

Why these `T_3_5`, `T_3_6`, `T_3_7` are in this table even them are not scheduled this time?
- We will fill all Token IDs in one request sequence to this table at once, but we only retrieve the tokens we scheduled this time. Then we will retrieve the remain Token IDs next time.

```
| T_0_0 | T_0_1 | T_0_2 |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |
| T_1_0 | T_1_1 |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |
| T_2_0 | T_2_1 | T_3_2 | T_3_3 | T_3_4 | T_3_5 | T_3_6 | T_3_7 |   ?   |   ?   |   ?   |   ?   |
|   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |
......
......
......
```

Note that the `T_x_x` is an `int32`

Let's say `M = max model len`, Then we can use `token positions` together with the `request indices` of each token to construct `token indices`.

So `token indices` = `[0 + 0 * M, 1 + 0 * M, 2 + 0 * M, 0 + 1 * M, 1 + 1 * M, 0 + 2 * M, 1 + 2 * M, 2 + 2 * M, 3 + 2 * M, 4 + 2 * M]` = `[0, 1, 2, 12, 13, 24, 25, 26, 27, 28]`

##### 3. Retrieve the Token IDs
As mentioned before, we will refer to these `Token IDs` as `Input IDs`.

We use the `token indices` to select out the corresponding `Input IDs` from the token table, The Pseudocode like:

```
input_ids = token_table[token_indices]
```

As mentioned before, we will refer these Token IDs as Inputs IDs:
- `Input IDs` = `[T_0_0, T_0_1, T_0_2, T_1_0, T_1_1, T_2_0, T_2_1, T_3_2, T_3_3, T_3_4]`

#### Build inputs attention metadata
Current **Block Table**, we use the first block (i.e. block_0) to mark the unused block. The shape of the block is `(max num request, max model len / block size)`, the `max model len / block size = 12 / 2 = 6`

```
| 1  | 2  | 0  | 0  | 0  | 0  |
| 3  | 0  | 0  | 0  | 0  | 0  |
| 4  | 5  | 6  | 0  | 0  | 0  |
| 0  | 0  | 0  | 0  | 0  | 0  |
......
......
......
```

The kv cache block in the device memory is like:

```
| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | ...... 
```

Let's say `K = max model len / block size = 6`, we can get token `device block number` from

The workflow of achieving slot mapping:
1. get `block table indices` using `K`, `positions` and `request indices`. Purpose: For each token, it could be used to select the `device block number` from `block table`.
2. get `device block number` using `block table indices`. Purpose: `device block number` indicates each token belong to which device block.
3. get `block offsets` using `positions` and `block size`. Purpose: `block offsets` indicates the offsets of each token within a block.
4. construct `slot mapping` using `device block number` and `block offsets`. Purpose: we can use `slot mapping` to store the Token IDs into token slots.

Details:
1. Using a simple formula to calculate the `block table indices`: `request indices * K + positions / block size`. So it equal to `[0 * 6 + 0 / 2, 0 * 6 + 1 / 2, 0 * 6 + 2 / 2, 1 * 6 + 0 / 2, 1 * 6 + 1 / 2, 2 * 6 + 0 / 2, 2 * 6 + 1 / 2, 2 * 6 + 2 / 2, 2 * 6 + 3 / 2, 2 * 6 + 4 / 2] = [0, 0, 1, 6, 6, 12, 12, 13, 13, 14]`. This could be used to select the `device block number` from `block table`. **token level**
2. Using the `block table indices` to select out the `device block number` for each scheduled token. The Pseudocode like: `block_numbers = block_table[block_table_indices]`. So `device block number =  [1, 1, 2, 3, 3, 4, 4, 5, 5, 6]`**token level**
3. `block offsets` could be computed by `block offsets = positions % block size = [0, 1, 0, 0, 1, 0, 1, 0, 1, 0]`. **token level**
4. At last, use `block offsets` and `device block number` to create `slot mapping`: `device block number * block size + block_offsets = [2, 3, 4, 6, 7, 8, 9, 10, 11, 12]`

First, we know the scheduled token count is `[3, 2, 5]` **request level**

- So, we can use prefix sum to calculate the `query start location`: `[0, 3, 5, 10]`. **request level**
- Because in step_1 all the tokens in prefill, computed tokens count is 0, then `sequence length` = `[3, 2, 5]`. **request level**
- As mentioned above, `number of computed tokens` are all 0: `[0, 0, 0]`. **request level**
- `number of requests`: `3`.
- `number of tokens`: `[3, 2, 5]`. **request level**
- `max query len`: `5`.
- `slot mapping`: `[2, 3, 4, 6, 7, 8, 9, 10, 11, 12]`. **token level**
- `attention mask`: For all request do prefill, we simply create only one mask matrix for reuse across different requests. The shape of this mask matrix is `5 * 5`:

### Step 2: Chunked prefill
In Step 2, we will no longer provide explanations or perform calculations; instead, we will directly present the final result.

#### Obtain inputs
The scheduled token of each request: `{'0': 1, '1': 1, '2': 3}`.

1. `request indices`: `[0, 1, 2, 2, 2]`
2. `token positions`: `[3, 2, 5, 6, 7]`

Current **Token IDs table**:

```
| T_0_0 | T_0_1 | T_0_2 | T_0_3 |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |
| T_1_0 | T_1_1 | T_1_2 |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |
| T_2_0 | T_2_1 | T_3_2 | T_3_3 | T_3_4 | T_3_5 | T_3_6 | T_3_7 |   ?   |   ?   |   ?   |   ?   |
|   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |
......
......
......
```

**Note**: The **T_0_3**, **T_1_2** are new Token IDs of request_0, request_1 respectively. them are sampled from the output of the model.

3. `token indices`: `[3, 14, 29, 30, 31]`
4. `Input IDs`: `[T_0_3, T_1_2, T_3_5, T_3_6, T_3_7]`

#### Build inputs attention metadata
Current **Block Table**. **Note**: We allocate the `7` and `8` block to `request_1` and `request_2` respectively. Because they need more space in device to store kv cache after generate new tokens or chunked prefill new tokens.

```
| 1  | 2  | 0  | 0  | 0  | 0  |
| 3  | 7  | 0  | 0  | 0  | 0  |
| 4  | 5  | 6  | 8  | 0  | 0  |
| 0  | 0  | 0  | 0  | 0  | 0  |
......
......
......
```

The kv cache block in the device memory is still like:

```
| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | ...... 
```

1. `block table indices`: `[1, 7, 14, 15, 15]`. **token level**
2. `device block number`: `[2, 7, 6, 8, 8]`. **token level**
3. `block offsets`: `[1, 0, 1, 0, 1]` **token level**
4. `slot mapping`: `[5, 14, 13, 16, 17]` **token level**

scheduled token count is `[1, 1, 3]`
- `query start location`: `[0, 1, 2, 5]`
- `sequence length`: `[4, 3, 8]`
- `number of computed tokens`: `[3, 2, 5]`
- `number of requests`: `3`
- `max query len`: `3`
- `slot mapping`: `[5, 14, 13, 16, 17]`
- `attention mask`: `5 * 8` Each token will have a `1 * 8` vector, and there are 5 scheduled tokens.

## At last
If you under stand the step_1 and step_2, you will know the all following steps.

Hope this article can help you get better understand to how vllm prepare inputs for model forwarding. If you have any good idea, welcome to contribute to us.
