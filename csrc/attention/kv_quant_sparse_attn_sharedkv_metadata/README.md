# KvQuantSparseAttnSharedkvMetadata

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- API功能：`KvQuantSparseAttnSharedkvMetadata`算子旨在生成一个任务列表，包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及 Q 和 K 的分块的索引，供后续`KvQuantSparseAttnSharedkv`算子使用。

## 参数说明

| 参数名           |输入/输出/属性|    描述    | 数据类型    |数据格式|
|-------------|------------|------|-----|-----|
|num_heads_q|属性|query对应的多头数，目前支持64/128。|INT32|-|
|num_heads_kv|属性|key和value对应的多头数，目前仅支持1。|INT32|-|
|head_dim|属性|注意力头的维度，目前仅支持512。|INT32|-|
|kv_quant_mode|属性|kv nope的量化模式，仅支持1，表示K、V nope为per_tile量化，量化后的KV数据类型为FLOAT8_E4M3FN。|INT32|-|
|cu_seqlens_q|可选输入|表示不同Batch中q的有效token数，维度为B+1。|INT32|ND|
|cu_seqlens_ori_kv|可选输入|预留参数，当前不生效，表示不同Batch中ori_kv的有效token数，维度为B+1。|INT32|ND|
|cu_seqlens_cmp_kv|可选输入|预留参数，当前不生效，表示不同Batch中cmp_kv的有效token数，维度为B+1。|INT32|ND|
|seqused_q|可选输入|预留参数，当前不生效，表示不同Batch中q的有效token数，维度为B。|INT32|ND|
|seqused_kv|可选输入|表示不同Batch中ori_kv的有效token数，维度为B。|INT32|ND|
|batch_size|可选属性|输入样本批量大小。|INT32|-|
|max_seqlen_q|可选属性|表示所有Batch中q的最大有效token数。|INT32|-|
|max_seqlen_kv|可选属性|表示所有Batch中ori_kv的最大有效token数。|INT32|-|
|ori_topk|可选属性|预留参数，当前不生效，表示通过QLI算法从ori_kv中筛选出的关键稀疏token的个数。|INT32|-|
|cmp_topk|可选属性|表示通过QLI算法从cmp_kv中筛选出的关键稀疏token的个数，目前支持512/1024。|INT32|-|
|tile_size|可选属性|表示量化粒度，必须能被rope_head_dim整除，默认值为None，当前仅支持64。|INT32|-|
|rope_head_dim|可选属性|默认值为0，当前仅支持64。|INT32|-|
|cmp_ratio|可选属性|表示对ori_kv的压缩率，数据范围支持4/128，默认值为None。|INT32|-|
|ori_mask_mode|可选属性|表示q和ori_kv计算的mask模式，仅支持输入默认值4，代表band模式的mask。|INT32|-|
|cmp_mask_mode|可选属性|表示q和cmp_kv计算的mask模式，仅支持输入默认值3，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。|INT32|-|
|ori_win_left|可选属性|表示q和ori_kv计算中q对过去token计算的数量，仅支持默认值127。|INT32|-|
|ori_win_right|可选属性|表示q和ori_kv计算中q对未来token计算的数量，仅支持默认值0。|INT32|-|
|layout_q|可选属性|用于标识输入q的数据排布格式，支持BSND和TND，默认值为BSND。|STRING|-|
|layout_kv|可选属性|用于标识输入ori_kv和cmp_kv的数据排布格式，仅支持传入默认值PA_ND（PageAttention）。|STRING|-|
|has_ori_kv|可选属性|用于标识是否含有ori_kv。|BOOL|-|
|has_cmp_kv|可选属性|用于标识是否含有cmp_kv。|BOOL|-|
|device|可选属性|用于获取设备信息。|STRING|-|
|metadata|输出|包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及 Q 和 K 的分块的索引的列表，shape固定为1024。|INT32|-|

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持aclgraph模式。
- Tensor不能全传None。
