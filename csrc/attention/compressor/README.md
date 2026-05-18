# Compressor

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- API功能：Compressor是推理场景下SAS和QLI的前处理算子，用于将每4或128个token的KV cache压缩成一个，然后每个token与这些压缩的KV cache进行DSA计算。在长序列的情况下，Compressor可以有效地减少计算开销。

- 计算公式：

    压缩阶段：
    1. 计算矩阵乘法：
        - C4A: $\left[kv\_state^a, score\_state^a\right] = X @ \left[W^{aKV}, W^{aGate}\right], \left[kv\_state^b, score\_state^b\right] = X @ \left[W^{bKV}, W^{bGate}\right];$
        - C128A: $\left[kv\_state, score\_state\right] = X @ \left[W^{KV}, W^{Gate}\right]$
    2. 计算分组加法：
        - C4A: $score\_state_i^\prime = \left[score\_state_{\left[4(i-1)+1:4i,:\right]}^a; score\_state_{\left[4i+1:4(i+1),:\right]}^b\right] + Ape,~i=1,2,\cdots, \frac{s}{4};$
        - C128A: $score\_state_i^\prime = score\_state_{\left[128(i-1)+1:128i,:\right]} + Ape,~i=1,2,\cdots, \frac{s}{128};$
    3. 计算分组Softmax：
        - C4A: $S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{4};$
        - C128A: $S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{128};$
    4. 计算Hadamard乘积：
        - C4A: $(S_H)_i = S_i^\prime \odot \left[kv\_state^a_{\left[4(i-1)+1:4i,:\right]} ; kv\_state^b_{\left[4i+1:4(i+1),:\right]}\right],~i=1,2,\cdots, \frac{s}{4};$
        - C128A: $S_H = S_i^\prime \odot kv\_state;$
    5. 沿着压缩轴分组求和：
        - C4A: $C_{i}^{\text{Comp}} = \left[1\right]_{1\times8} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{4};$
        - C128A: $C_{i}^{\text{Comp}} = \left[1\right]_{1\times128} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{128};$

    后处理阶段：

    6. 计算RMSNorm：
        - $\text{RMS}(C^{\text{Comp}}) = \sqrt{\frac{1}{N} \sum_{i=j* N}^{(j+1)* N} {(C_{i}^{\text{Comp}})}^{\text{2}} + norm\_eps} ,N=head\_dim, ~j=1,2,\cdots, \frac{s}{cmp\_ratio}$
        - $\text{RmsNorm}(C^{\text{Comp}}) = norm\_weight \cdot \frac{C_{i}^{\text{Comp}}}{\text{RMS}(C^{\text{Comp}})}$
    7. 计算Rope；

- 主要计算过程为：
    1. 将输入$X$与$W^{KV}$做Matmul运算得到$kv\_state$，将输入$X$与$W^{Gate}$做Matmul运算后再与$Ape$做Add运算得到$score\_state$，$kv\_state$与$score\_state$根据输入的start_pos及cu_seqlens完成更新。
    2. 在coff为2的情况下对$kv\_state$和$score\_state$进行数据重排。
    3. 对$score\_state$进行softmax运算将softmax结果与$kv\_state$做Mul计算，后进行ReduceSum运算。
    4. 根据输入数据norm_weight、rope_sin、rope_cos，进行RMSNorm和Rope运算，得到$cmp\_kv$结果输出。

## 参数说明

| 参数名                      | 输入/输出/属性 | 描述  | 数据类型       | 数据格式   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| x | 输入 | 公式中的$X$，表示原始不经压缩的数据。 | FLOAT16、BFLOAT16 | ND         |
| wkv | 输入 | 公式中的$W^{KV}$，表示kv压缩权重。  | FLOAT16、BFLOAT16 | ND |
| wgate | 输入 | 公式中的$W^{Gate}$，表示gate压缩权重。 | FLOAT16、BFLOAT16 | ND |
| kv_state | 输入 | 公式中的$kv\_state$，表示kv\_state的历史数据。 | FLOAT32     | ND         |
| score_state | 输入 | 公式中的$score\_state$，表示score\_state中的历史数据。 | FLOAT32       | ND         |
| ape | 输入 | 公式中的$Ape$，表示positional biases。 | FLOAT32       | ND         |
| norm\_weight | 输入 | 表示计算RmsNorm时的权重系数。 | FLOAT16、BFLOAT16       | ND         |
| rope\_sin | 输入 | 表示Rope计算时sin的权重系数。 | FLOAT16、BFLOAT16       | ND         |
| rope\_cos | 输入 | 表示Rope计算时cos的权重系数。 | FLOAT16、BFLOAT16       | ND         |
| rope\_head\_dim | 属性 | 表示rope_cos和rope_sin的hidden层最小单元大小，当前仅支持64。 | INT32       | -         |
| cmp\_ratio | 属性 | 用于稀疏计算，表示数据压缩率。 | INT32          | -         |
| kv\_block\_table | 可选输入 | 表示kv\_state存储使用的block映射表。当其中元素的值为0时，表示当前位置无需进行更新kv\_state操作。 | INT32 | ND         |
| score\_block\_table | 可选输入 | 表示score\_state存储使用的block映射表。当其中元素的值为0时，表示当前位置无需进行更新score\_state操作。 | INT32 | ND         |
| cu\_seqlens | 可选输入 | 表示不同Batch中的有效token数。  | INT32          | ND         |
| seqused | 可选输入 | 表示不同Batch中实际参与压缩的token数，如果指定为None时，表示和每个Batch上的Sequence Length长度相同。 | INT32          | ND         |
| start\_pos | 可选输入 | 表示计算起始位置。 | INT32          | ND         |
| coff | 可选属性 | 默认值1，支持1/2。当coff=1时，无需进行overlap数据重排。当coff=2时，需要进行overlap数据重排。  | INT32          | -         |
| norm\_eps | 可选属性 | 表示RmsNorm计算的权重系数。默认值1e-6。 | FLOAT32          | -         |
| rotary\_mode | 可选属性 | 表示Rop计算的模式。默认值1，支持1/2。rotary\_mode为1时，代表half模式。rotary\_mode为2时，代表interleave模式。 | INT32          | -         |
| enabled\_grad | 可选属性 | 训练场景使用，表示是否参与反向更新。默认值false，支持false/true。**目前暂不支持输入true**。 | BOOL          | -         |
| cmp\_kv | 输出 | 表示压缩后的数据。 | FLOAT16、BFLOAT16         | ND          |
| wkv\_proj | 可选输出 | 训练反向使用，表示wkv权重Matmul的计算结果，**目前暂不支持返回wkv\_proj**。 | FLOAT16、BFLOAT16         | ND          |
| softmax\_res | 可选输出 | 训练反向使用，表示Softmax计算结果，**目前暂不支持返回softmax\_res**。 | FLOAT16、BFLOAT16         | ND          |
| norm\_x | 可选输出 | 训练反向使用，表示Rms计算的输入，**目前暂不支持返回norm\_x**。 | FLOAT16、BFLOAT16         | ND          |
| norm\_rstd | 可选输出 | 训练反向使用，表示Rms计算的中间结果，**目前暂不支持返回norm\_rstd**。 | FLOAT16、BFLOAT16         | ND          |

## 约束说明

- x参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、D（Head Dim）表示hidden层的最小单元大小、T表示所有Batch输入样本序列长度的累加和。
- 输入shape限制：
    - wkv支持输入shape[coff* D,H]
    - wgate支持输入shape[coff* D,H]
    - kv\_state、score\_state支持输入shape[block_num,block_size,coff* D]，要求block_num>0。
    - ape支持输入shape[cmp_ratio,coff* D]
    - norm\_weight支持输入shape[D,]
    - start\_pos支持输入shape[B,]
    - 若x的维度采用BS合轴，即x的输入shape为[T,H]
        - rope_sin、rope_cos要求输入shape为[min(T,T//cmp_ratio+B),rope_head_dim]。
        - cu\_seqlens输入shape必须为[B+1,]。该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值，且第一位必须位0。
        - seqused，支持输入shape[B,]，要求每个Batch的有效token数要求小于等于对应Sequence Length长度，即seqused[n] <= cu\_seqlens[n+1] - cu\_seqlens[n]，且不小于0。
        - kv\_block\_table、score\_block\_table支持输入shape[B,ceil(Smax/block_size)]。Smax为每个Batch中最大的Sequence Length，即Smax=max(start\_pos)+max(cu\_seqlens[n+1] - cu\_seqlens[n])。
        - cmp\_kv，输出shape为[min(T,T//cmp_ratio+B),D]：<batch0>compressed_tokens + <batch1>compressed_tokens + ... + <batchN>compressed_tokens + pad。
        - wkv\_proj，输出shape为[T,coff* D]。
        - norm\_x，输出shape为[min(T,T//cmp_ratio+B),D]。
        - norm\_rstd，输出shape为[min(T,T//cmp_ratio+B)]。
    - 若x的维度不采用BS合轴，即x的输入shape为[B,S,H]
        - rope_sin、rope_cos要求输入shape为[B,ceil(S/cmp_ratio),rope_head_dim]。
        - cu\_seqlens，参数必须为空。
        - seqused，支持输入shape[B,]，要求每个Batch的有效token数要求小于等于对应Sequence Length长度，即要求seqused[n] <= S，且不小于0。
        - kv\_block\_table、score\_block\_table支持输入shape[B,ceil(Smax/block_size)]。Smax为每个Batch中最大的Sequence Length，即Smax=max(start\_pos)+S。
        - cmp\_kv，输出shape为[B,ceil(S/cmp_ratio),D]：(<batch0>compressed_tokens+pad0) + (<batch1>compressed_tokens+pad1) + ...  + (<batchN>compressed_tokens+padN)。
        - wkv\_proj，输出shape为[B,S,coff* D]。
        - norm\_x，输出shape为[B,ceil(S/cmp_ratio),D]。
        - norm\_rstd，输出shape为[B,ceil(S/cmp_ratio)]。
- 输入值域限制：
    - 该接口支持B、S泛化，且存在如下场景限制：
        - 部分长序列场景下，如果计算量过大可能会导致出现超过NPU内存的报错，注：这里计算量会受x输入shape的影响，值越大计算量越大。典型的长序列（即B、S的乘积或T较大）场景包括但不限于：
      <div style="overflow-x: auto;">
      <table style="undefined;table-layout: fixed; width: 400px"><colgroup>
      <col style="width: 100px">
      <col style="width: 100px">
      </colgroup><thead>
      <tr>
      <th>B</th>
      <th>S</th>
      <th>H</th>
      </tr></thead>
      <tbody>
      <tr>
      <td>100</td>
      <td>65525</td>
      <td>4096</td>
      </tr>
      <tr>
      <td>25</td>
      <td>261120</td>
      <td>4096</td>
      </tr>
      <tr>
      <td>100</td>
      <td>131072</td>
      <td>4096</td>
      </tr>
      <tr>
      <td>100</td>
      <td>261120</td>
      <td>4096</td>
      </tr>
      </tbody>
      </table>
      </div>
- 输入属性限制：
    - 支持D为128/512。
    - 支持H为1K~10K，512对齐。
    - 泛化支持block_size小于等于1024，16对齐。
    - 支持cmp_ratio为4/128。支持如下三种情况：
        - C4A: D=512, coff=2, cmp_ratio=4；
        - C4Li: D=128, coff=2, cmp_ratio=4；
        - C128A: D=512, coff=1, cmp_ratio=128。
    - 支持rotary_mode为2，Rope计算模式为interleave。

## Atlas A3 推理系列产品 调用说明

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import custom_ops
    import torch.nn as nn
    import math

    def get_seq_used_by_batch(batch_idx, S, seqused, cu_seqlens):
        if seqused is not None:
            return seqused[batch_idx]
        else:
            if cu_seqlens is not None:
                return cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]
            else:
                return S

    data_type = torch.bfloat16
    hidden_size = 4096
    rope_head_dim = 64
    norm_eps = 1e-6
    coff = 1 # 1:no overlap 2:overlap
    cmp_ratio = 128
    rotary_mode = 2
    head_dim = 512
    cu_seqlens = [0, 1]
    # -------------
    B = 1
    S = 1
    S_max = 0
    block_size = 128
    start_pos = [8191] * B # (B,)
    start_p=8191
    seqused = None # (B,), None时cu_seqlens的数据全部参与计算，否则按传参实际值计算

    # BS是否合轴
    bs_combine_flag = True
    update_flag = 1

    if seqused is not None:
        seqused = torch.tensor(seqused).to(torch.int32)
    if start_pos is not None:
        start_pos = torch.tensor(start_pos).to(torch.int32)
    else:
        start_pos = torch.full((B,), start_p, dtype=torch.int32)

    if bs_combine_flag:
        if cu_seqlens is None:
            T = B * S
            if T !=0:
                cu_seqlens = torch.arange(0, T + 1, S, dtype=torch.int32)
            else:
                cu_seqlens = torch.zeros((B+1), dtype=torch.int32)
        else:
            cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32)
        for i in range(B):
            if start_pos[i] + cu_seqlens[i + 1] - cu_seqlens[i] > S_max:
                S_max = start_pos[i] + cu_seqlens[i + 1] - cu_seqlens[i]
    else:
        cu_seqlens = None
        S_max = max(start_pos) + S
    ### ======================== gen input data start =============================
    # page state
    max_block_num_per_batch = (S_max + block_size - 1) // block_size
    block_num = B * max_block_num_per_batch
    next_block_id = 1
    print(f"max_block_num_per_batch: {max_block_num_per_batch}")
    block_table = torch.zeros(size=(B, max_block_num_per_batch), dtype=torch.int32)
    for i in range(B):
        # 需要读取state的范围
        cur_start = start_pos[i] // cmp_ratio * cmp_ratio - cmp_ratio
        cur_end = start_pos[i] // cmp_ratio * cmp_ratio + cmp_ratio
        if start_pos[i] % cmp_ratio == 0:
            cur_end = start_pos[i]
        cur_end = min(cur_end, start_pos[i] + S)
        cur_start_block_id = (cur_start // block_size) if cur_start >= 0 else 0
        cur_end_block_id = (cur_end - 1) // block_size
        for j in range(cur_start_block_id, cur_end_block_id + 1):
            block_table[i][j] = next_block_id
            next_block_id = next_block_id + 1
        # 需要写入state的范围
        end_pos = get_seq_used_by_batch(i, S, seqused, cu_seqlens)
        next_start = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio - cmp_ratio
        next_end = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio + cmp_ratio
        if (start_pos[i] + end_pos) % cmp_ratio == 0:
            next_end = start_pos[i] + end_pos
        next_end = min(next_end, start_pos[i] + end_pos)
        next_start_block_id = (next_start // block_size) if next_start >= 0 else 0
        next_end_block_id = (next_end - 1) // block_size
        for j in range(next_start_block_id, next_end_block_id + 1):
            if block_table[i][j] == 0:
                block_table[i][j] = next_block_id
                next_block_id = next_block_id + 1

    if B==0:
        kv_state = torch.tensor(np.random.uniform(-10, 10, (0, block_size, coff * head_dim))).to(torch.float32)
        score_state = torch.tensor(np.random.uniform(-10, 10, (0, block_size, coff * head_dim))).to(torch.float32)
    else:
        kv_state = torch.tensor(np.random.uniform(-10, 10, (torch.max(block_table) + 1, block_size, coff * head_dim))).to(torch.float32)
        score_state = torch.tensor(np.random.uniform(-10, 10, (torch.max(block_table) + 1, block_size, coff * head_dim))).to(torch.float32)

    # other input
    if bs_combine_flag:
        x_shape = (cu_seqlens[-1], hidden_size)
        rope_sin_shape = (min(x_shape[0], x_shape[0] // cmp_ratio + B), rope_head_dim)
        rope_cos_shape = rope_sin_shape
    else:
        x_shape = (B, S, hidden_size)
        rope_sin_shape = (B, (S + cmp_ratio - 1) // cmp_ratio, rope_head_dim)
        rope_cos_shape = rope_sin_shape

    x = torch.tensor(np.random.uniform(-10.0, 10.0, x_shape)).to(data_type).npu()
    wkv = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(data_type).npu()
    wgate = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(data_type).npu()
    ape = torch.tensor(np.random.uniform(-10, 10, (cmp_ratio, coff * head_dim))).to(torch.float32).npu()
    norm_weight = torch.tensor(np.random.uniform(-10, 10, (head_dim))).to(data_type).npu()
    rope_sin = torch.tensor(np.random.uniform(-1, 1, rope_sin_shape)).to(data_type).npu()
    rope_cos = torch.tensor(np.random.uniform(-1, 1, rope_cos_shape)).to(data_type).npu()
    kv_state = kv_state.npu()
    score_state = score_state.npu()
    block_table = block_table.npu()
    start_pos = torch.tensor(start_pos).to(torch.int32).npu()
    if cu_seqlens is not None:
        cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32).npu()
    if seqused is not None:
        seqused = torch.tensor(seqused).to(torch.int32).npu()

    cmp_kv,_ ,_ ,_ ,_ = (
        torch.ops.custom.compressor(
            x,
            wkv,
            wgate,
            kv_state,
            score_state,
            ape,
            norm_weight,
            rope_sin,
            rope_cos,
            kv_block_table = block_table,
            score_block_table = block_table,
            cu_seqlens = cu_seqlens,
            seqused = seqused,
            start_pos = start_pos,
            rope_head_dim = rope_head_dim,
            cmp_ratio = cmp_ratio,
            coff = coff,
            norm_eps = norm_eps,
            rotary_mode = rotary_mode
        )
    )
    ```
- aclgraph调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import torchair
    import custom_ops
    import math

    def get_seq_used_by_batch(batch_idx, S, seqused, cu_seqlens):
        if seqused is not None:
            return seqused[batch_idx]
        else:
            if cu_seqlens is not None:
                return cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]
            else:
                return S

    data_type = torch.bfloat16
    hidden_size = 4096
    rope_head_dim = 64
    norm_eps = 1e-6
    coff = 1 # 1:no overlap 2:overlap
    cmp_ratio = 128
    rotary_mode = 2
    head_dim = 512
    cu_seqlens = [0, 1]
    # -------------
    B = 1
    S = 1
    S_max = 0
    block_size = 128
    start_pos = [8191] * B # (B,)
    start_p=8191
    seqused = None # (B,), None时cu_seqlens的数据全部参与计算，否则按传参实际值计算

    # BS是否合轴
    bs_combine_flag = True
    update_flag = 1

    if seqused is not None:
        seqused = torch.tensor(seqused).to(torch.int32)
    if start_pos is not None:
        start_pos = torch.tensor(start_pos).to(torch.int32)
    else:
        start_pos = torch.full((B,), start_p, dtype=torch.int32)

    if bs_combine_flag:
        if cu_seqlens is None:
            T = B * S
            if T !=0:
                cu_seqlens = torch.arange(0, T + 1, S, dtype=torch.int32)
            else:
                cu_seqlens = torch.zeros((B+1), dtype=torch.int32)
        else:
            cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32)
        for i in range(B):
            if start_pos[i] + cu_seqlens[i + 1] - cu_seqlens[i] > S_max:
                S_max = start_pos[i] + cu_seqlens[i + 1] - cu_seqlens[i]
    else:
        cu_seqlens = None
        S_max = max(start_pos) + S
    ### ======================== gen input data start =============================
    # page state
    max_block_num_per_batch = (S_max + block_size - 1) // block_size
    block_num = B * max_block_num_per_batch
    next_block_id = 1
    print(f"max_block_num_per_batch: {max_block_num_per_batch}")
    block_table = torch.zeros(size=(B, max_block_num_per_batch), dtype=torch.int32)
    for i in range(B):
        # 需要读取state的范围
        cur_start = start_pos[i] // cmp_ratio * cmp_ratio - cmp_ratio
        cur_end = start_pos[i] // cmp_ratio * cmp_ratio + cmp_ratio
        if start_pos[i] % cmp_ratio == 0:
            cur_end = start_pos[i]
        cur_end = min(cur_end, start_pos[i] + S)
        cur_start_block_id = (cur_start // block_size) if cur_start >= 0 else 0
        cur_end_block_id = (cur_end - 1) // block_size
        for j in range(cur_start_block_id, cur_end_block_id + 1):
            block_table[i][j] = next_block_id
            next_block_id = next_block_id + 1
        # 需要写入state的范围
        end_pos = get_seq_used_by_batch(i, S, seqused, cu_seqlens)
        next_start = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio - cmp_ratio
        next_end = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio + cmp_ratio
        if (start_pos[i] + end_pos) % cmp_ratio == 0:
            next_end = start_pos[i] + end_pos
        next_end = min(next_end, start_pos[i] + end_pos)
        next_start_block_id = (next_start // block_size) if next_start >= 0 else 0
        next_end_block_id = (next_end - 1) // block_size
        for j in range(next_start_block_id, next_end_block_id + 1):
            if block_table[i][j] == 0:
                block_table[i][j] = next_block_id
                next_block_id = next_block_id + 1

    if B==0:
        kv_state = torch.tensor(np.random.uniform(-10, 10, (0, block_size, coff * head_dim))).to(torch.float32)
        score_state = torch.tensor(np.random.uniform(-10, 10, (0, block_size, coff * head_dim))).to(torch.float32)
    else:
        kv_state = torch.tensor(np.random.uniform(-10, 10, (torch.max(block_table) + 1, block_size, coff * head_dim))).to(torch.float32)
        score_state = torch.tensor(np.random.uniform(-10, 10, (torch.max(block_table) + 1, block_size, coff * head_dim))).to(torch.float32)

    # other input
    if bs_combine_flag:
        x_shape = (cu_seqlens[-1], hidden_size)
        rope_sin_shape = (min(x_shape[0], x_shape[0] // cmp_ratio + B), rope_head_dim)
        rope_cos_shape = rope_sin_shape
    else:
        x_shape = (B, S, hidden_size)
        rope_sin_shape = (B, (S + cmp_ratio - 1) // cmp_ratio, rope_head_dim)
        rope_cos_shape = rope_sin_shape

    x = torch.tensor(np.random.uniform(-10.0, 10.0, x_shape)).to(data_type).npu()
    wkv = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(data_type).npu()
    wgate = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(data_type).npu()
    ape = torch.tensor(np.random.uniform(-10, 10, (cmp_ratio, coff * head_dim))).to(torch.float32).npu()
    norm_weight = torch.tensor(np.random.uniform(-10, 10, (head_dim))).to(data_type).npu()
    rope_sin = torch.tensor(np.random.uniform(-1, 1, rope_sin_shape)).to(data_type).npu()
    rope_cos = torch.tensor(np.random.uniform(-1, 1, rope_cos_shape)).to(data_type).npu()
    kv_state = kv_state.npu()
    score_state = score_state.npu()
    block_table = block_table.npu()
    start_pos = torch.tensor(start_pos).to(torch.int32).npu()
    if cu_seqlens is not None:
        cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32).npu()
    if seqused is not None:
        seqused = torch.tensor(seqused).to(torch.int32).npu()

    class CompressorNetwork(nn.Module):
        def __init__(self):
            super(CompressorNetwork, self).__init__()

        def forward(self, x, wkv, wgate, kv_state, score_state, ape, norm_weight, rope_sin,
                    rope_cos, rope_head_dim, cmp_ratio, kv_block_table = None, score_block_table = None, cu_seqlens = None,
                    seqused = None, start_pos = None, coff = 1, norm_eps = 1e-6, rotary_mode = 1):
            cmp_kv,_ ,_ ,_ ,_ = (
                torch.ops.custom.compressor(
                    x,
                    wkv,
                    wgate,
                    kv_state,
                    score_state,
                    ape,
                    norm_weight,
                    rope_sin,
                    rope_cos,
                    kv_block_table = kv_block_table,
                    score_block_table = score_block_table,
                    cu_seqlens = cu_seqlens,
                    seqused = seqused,
                    start_pos = start_pos,
                    rope_head_dim = rope_head_dim,
                    cmp_ratio = cmp_ratio,
                    coff = coff,
                    norm_eps = norm_eps,
                    rotary_mode = rotary_mode
                )
            )
            return cmp_kv

    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(CompressorNetwork(), fullgraph=True, backend=npu_backend, dynamic=False)
    cmp_kv = npu_mode(
                    x,
                    wkv,
                    wgate,
                    kv_state,
                    score_state,
                    ape,
                    norm_weight,
                    rope_sin,
                    rope_cos,
                    kv_block_table = block_table,
                    score_block_table = block_table,
                    cu_seqlens = cu_seqlens,
                    seqused = seqused,
                    start_pos = start_pos,
                    rope_head_dim = rope_head_dim,
                    cmp_ratio = cmp_ratio,
                    coff = coff,
                    norm_eps = norm_eps,
                    rotary_mode = rotary_mode)
    ```

更多使用示例见[pytest示例](./tests/pytest/README.md)。
