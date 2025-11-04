# Prepare inputs for model forwarding

## Purpose
Information required to perform model forward pass:
 - the inputs
 - the corresponding attention metadata of the inputs

The following diagram shows what we should prepare for model inference.

```
              +---------------+
  inputs  --> |               |
              |     model     |  --> output
attn_meta --> |               |
              +---------------+  
```

Therefore, as long as we have these two pieces of information mentioned above, we can perform the model's forward propagation.

This document will explain **how we obtain the inputs and their corresponding attention metadata**.

## Overview
### 1. Obtain inputs
The workflow of obtaining inputs:
1. Get `token positions`: relative position of each token within its request sequence.

2. Get `token indices`: index of each scheduled token in the token table.

3. Get `Token IDs`: using token indices to retrieve the Token IDs from **token id table**.

At last, these `Token IDs` are required to be fed into a model, and also, `positions` should be sent into the model to create `Rope` (Rotary positional embedding). Both of them are the inputs of the model.

**Note**: The `Token IDs` are the inputs of a model, so we also call them `Inputs IDs`.

### 2. Build inputs attention metadata
A model requires these attention metadata during the forward pass:
- `query start location`: start and end location of each request corresponding to the scheduled tokens.
- `sequence length`: length of each request including both computed tokens and newly scheduled tokens.
- `number of computed tokens`: number of computed tokens for each request.
- `number of requests`: number of requests in this batch.
- `number of tokens`: total number of scheduled tokens in this batch.
- **`block table`**: translates the logical address (within its sequence) of each block to its global physical address in the device's memory.
- `max query len`: the longest scheduled tokens length in this request batch.
- `slot mapping`: indices of each token that input token will be stored into.
- `attention mask`: mask matrix applied to attention scores before softmax to control which tokens can attend to each other (usually a causal attention).

## Before start
There are mainly three types of variables.
- token level: represents one attribute corresponding to each scheduled token, so the length of this variable is the number of scheduled tokens
- request level: represents one attribute of each scheduled request, whose length usually is the number of scheduled requests. (`query start location` is a special case, which has one more element)
- system level:
  1. **Token IDs table**: stores the token IDs (i.e. the inputs of a model) of each request. The shape of this table is `(max num request, max model len)`. Here, `max num request` is the maximum count of concurrent requests allowed in a forward batch and `max model len` is the maximum token count that can be handled at one request sequence in this model.
  2. **Block table**: translates the logical address (within its sequence) of each block to its global physical address in the device's memory. The shape of this table is `(max num request, max model len / block size)`

**Note**: Both of these two tables are come from the `_update_states` method before **preparing inputs**. You can take a look if you need more inspiration.

### Tips
What is `Token ID`?
Simply put, a `token ID` is an **integer** (usually `int32`), which represents a token.
Example of `Token ID`:

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
Assumptions:
- maximum number of  tokens can be scheduled at once: 10
- `block size`: 2
- Totally schedule 3 requests. Their prompt lengths are 3, 2, and 8 respectively.
- `max model length`: 12 (the maximum token count can be handled at one request sequence in a model).

These assumptions are configured in the beginning when starting vLLM. They are not fixed, so you can manually set them.
### Step 1: All requests in the prefill phase

#### Obtain inputs
As the maximum number of tokens that can be schedules is 10, the scheduled tokens of each request can be represented as `{'0': 3, '1': 2, '2': 5}`. Note that`request_2` uses chunked prefill, leaving 3 prompt tokens unscheduled.

##### 1. Get token positions:
First, determine which request each token belongs to: tokens 0–2 are assigned to **request_0**, tokens 3–4 to **request_1**, and tokens 5–9 to **request_2**. To represent this mapping, we use `request indices`, for example, `request indices`: `[0, 0, 0, 1, 1, 2, 2, 2, 2, 2]`.

For each request, use **the number of computed tokens** + **the relative position of current scheduled tokens** (`request_0: [0 + 0, 0 + 1, 0 + 2]`, `request_1: [0 + 0, 0 + 1]`, `request_2: [0 + 0, 0 + 1,..., 0 + 4]`) and then concatenate them together (`[0, 1, 2, 0, 1, 0, 1, 2, 3, 4]`).

Note: there is more efficient way (using `request indices`) to create positions in actual code.

Finally, `token positions` can be obtained as `[0, 1, 2, 0, 1, 0, 1, 2, 3, 4]`. This variable is **token level**.

##### 2. Get token indices:
The shape of the current **Token IDs table** is `(max num request, max model len)`.

Why these `T_3_5`, `T_3_6`, `T_3_7` are in this table without being scheduled?
- We fill all Token IDs in one request sequence to this table at once, but we only retrieve the tokens we scheduled this time. Then we retrieve the remain Token IDs next time.

```
| T_0_0 | T_0_1 | T_0_2 |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |
| T_1_0 | T_1_1 |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |
| T_2_0 | T_2_1 | T_3_2 | T_3_3 | T_3_4 | T_3_5 | T_3_6 | T_3_7 |   ?   |   ?   |   ?   |   ?   |
|   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |   ?   |
......
......
......
```

Note that`T_x_x` is an `int32`.

Let's say `M = max model len`. Then we can use `token positions` together with `request indices` of each token to construct `token indices`.

So `token indices` = `[0 + 0 * M, 1 + 0 * M, 2 + 0 * M, 0 + 1 * M, 1 + 1 * M, 0 + 2 * M, 1 + 2 * M, 2 + 2 * M, 3 + 2 * M, 4 + 2 * M]` = `[0, 1, 2, 12, 13, 24, 25, 26, 27, 28]`

##### 3. Retrieve the Token IDs
We use `token indices` to select out the corresponding `Input IDs` from the token table. The pseudocode is as follows:

```
input_ids = token_table[token_indices]
```

As mentioned before, we refer to these `Token IDs` as `Input IDs`.
- `Input IDs` = `[T_0_0, T_0_1, T_0_2, T_1_0, T_1_1, T_2_0, T_2_1, T_3_2, T_3_3, T_3_4]`

#### Build inputs attention metadata
In the current **Block Table**, we use the first block (i.e. block_0) to mark the unused block. The shape of the block is `(max num request, max model len / block size)`, where `max model len / block size = 12 / 2 = 6`.

```
| 1  | 2  | 0  | 0  | 0  | 0  |
| 3  | 0  | 0  | 0  | 0  | 0  |
| 4  | 5  | 6  | 0  | 0  | 0  |
| 0  | 0  | 0  | 0  | 0  | 0  |
......
......
......
```

The KV cache block in the device memory is like:

```
| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | ...... 
```

Let's say `K = max model len / block size = 6`, and we can get token `device block number`.

The workflow of achieving slot mapping:
1. Get `block table indices` using `K`, `positions` and `request indices`.

   Purpose: For each token, it could be used to select `device block number` from `block table`.

2. Get `device block number` using `block table indices`.

   Purpose: `device block number` indicates which device block each token belongs to.

3. Get `block offsets` using `positions` and `block size`.

   Purpose: `block offsets` indicates the offsets of each token within a block.

4. construct `slot mapping` using `device block number` and `block offsets`.

   Purpose: we can use `slot mapping` to store Token IDs into token slots.

Details:
1. (**Token level**) Use a simple formula to calculate `block table indices`: `request indices * K + positions / block size`. So it equal to `[0 * 6 + 0 / 2, 0 * 6 + 1 / 2, 0 * 6 + 2 / 2, 1 * 6 + 0 / 2, 1 * 6 + 1 / 2, 2 * 6 + 0 / 2, 2 * 6 + 1 / 2, 2 * 6 + 2 / 2, 2 * 6 + 3 / 2, 2 * 6 + 4 / 2] = [0, 0, 1, 6, 6, 12, 12, 13, 13, 14]`. This could be used to select `device block number` from `block table`.
2. (**Token level**) Use `block table indices` to select out `device block number` for each scheduled token. The Pseudocode is `block_numbers = block_table[block_table_indices]`. So `device block number=[1, 1, 2, 3, 3, 4, 4, 5, 5, 6]`
3. (**Token level**) `block offsets` could be computed by `block offsets = positions % block size = [0, 1, 0, 0, 1, 0, 1, 0, 1, 0]`.
4. At last, use `block offsets` and `device block number` to create `slot mapping`: `device block number * block size + block_offsets = [2, 3, 4, 6, 7, 8, 9, 10, 11, 12]`

(**Request level**) As we know the scheduled token count is `[3, 2, 5]`:

- (**Request level**) Use prefix sum to calculate `query start location`: `[0, 3, 5, 10]`.
- (**Request level**) All tokens in step 1 are in the prefill stage, and the computed tokens count is 0; then `sequence length` = `[3, 2, 5]`.
- (**Request level**) As mentioned above, `number of computed tokens` are all 0s: `[0, 0, 0]`.
- `number of requests`: `3`
- (**Request level**) `number of tokens`: `[3, 2, 5]`
- `max query len`: `5`
- (**Token level**) `slot mapping`: `[2, 3, 4, 6, 7, 8, 9, 10, 11, 12]`
- `attention mask`: For all requests that initiate a prefill process, we simply create only one mask matrix for reuse across different requests. The shape of this mask matrix is `5 * 5`:

### Step 2: Chunked prefill
In Step 2, we no longer provide explanations or perform calculations; instead, we directly present the final result.

#### Obtain inputs
Scheduled token of each request: `{'0': 1, '1': 1, '2': 3}`

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

**Note**: **T_0_3**, **T_1_2** are new Token IDs of **request_0** and **request_1** respectively. They are sampled from the output of the model.

3. `token indices`: `[3, 14, 29, 30, 31]`
4. `Input IDs`: `[T_0_3, T_1_2, T_3_5, T_3_6, T_3_7]`

#### Build inputs attention metadata
We allocate the blocks `7` and `8` to `request_1` and `request_2` respectively, as they need more space in device to store KV cache following token generation or chunked prefill.

Current **Block Table**:

```
| 1  | 2  | 0  | 0  | 0  | 0  |
| 3  | 7  | 0  | 0  | 0  | 0  |
| 4  | 5  | 6  | 8  | 0  | 0  |
| 0  | 0  | 0  | 0  | 0  | 0  |
......
......
......
```

KV cache block in the device memory:

```
| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | ...... 
```

1. (**Token level**) `block table indices`: `[1, 7, 14, 15, 15]`
2. (**Token level**) `device block number`: `[2, 7, 6, 8, 8]`
3. (**Token level**) `block offsets`: `[1, 0, 1, 0, 1]`
4. (**Token level**) `slot mapping`: `[5, 14, 13, 16, 17]`

Scheduled token count:`[1, 1, 3]`
- `query start location`: `[0, 1, 2, 5]`

- `sequence length`: `[4, 3, 8]`

- `number of computed tokens`: `[3, 2, 5]`

- `number of requests`: `3`

- `max query len`: `3`

- `slot mapping`: `[5, 14, 13, 16, 17]`

- `attention mask`: `5 * 8`

  Each token has a `1 * 8` vector, and there are 5 scheduled tokens.

## At last
If you understand the step_1 and step_2, you will know the all following steps.

Hope this document can help you better understand how vLLM prepares inputs for model forwarding. If you have any good idea, welcome to contribute to us.
