# L2 pHash 优化与新版接口适配

更新日期：2026-10-26。

本方案基于 L2 提交 `5fecef105` 迁移而来，保留完整 pHash 直接查询和分段倒排索引。
当前按 CPU 像素方案简化：图片/视频像素张量和 grid 元数据位于 CPU，
`mm_hash` 是 CPU 侧 Python key；encoder embedding 继续使用原来的 NPU 路径。

## 单一 band 数设置

在 `vllm_ascend/distributed/ec_transfer/phash.py` 中修改：

```python
PHASH_NUM_BANDS = 4
```

当前 64 位 pHash 分为 4 个 16-bit band。该常量同时决定 `bands()` 的默认段数、
`max_hamming` 默认值及允许的上界。生产代码的索引写入和查询均调用 `bands(phash)`，
没有另外指定段数，因此修改这个常量即可统一调整分段方式。

| band 数量 | 每段位数 | 默认 `max_hamming` | 允许范围 |
| --- | --- | --- | --- |
| 2 | 32 | 1 | 0–1 |
| 4 | 16 | 3 | 0–3 |
| 8 | 8 | 7 | 0–7 |
| 16 | 4 | 15 | 0–15 |

配置规则是默认值为 `PHASH_NUM_BANDS - 1`，且要求
`0 <= max_hamming < PHASH_NUM_BANDS`。直接构造配置对象和解析外部配置都执行检查。
当前四段方案下，旧配置中的 4–7（包括原默认值 5）会报参数错误，需要调整为 0–3。
没有增加环境变量。

修改段数时还需注意：

- 合法段数为 64 的正整数因子：1、2、4、8、16、32、64。其他值无法等宽分段。
- 外部显式配置的 `max_hamming` 不会自动变化。例如改成 8 段，启动配置仍写着 3，
  实际阈值仍为 3；只有未指定该参数时才使用新的默认值 7。
- band 数量不影响完整 64 位 pHash 的计算与打包结果。固定距离阈值时，更多、更短的 band
  通常带来更多共享分段的候选和更多索引成员；段数更多不意味着检索一定更快。

## 分段检索为什么能替代全表扫描

原实现遍历 `phash_to_resized_mm_hashes.items()`，对全部已保存的 pHash
逐个计算汉明距离，再收集满足条件的图片 key。

当前实现先用 band 索引缩小候选集合，再执行同一个完整汉明距离判断：

```python
candidate_phashes: set[int] = set()
for band in bands(query_phash):
    candidate_phashes.update(self._phash_band_index.get(band, ()))

matched_phashes = (
    candidate_phash
    for candidate_phash in candidate_phashes
    if hamming(query_phash, candidate_phash) <= self._config.max_hamming
)
for candidate_phash in sorted(matched_phashes):
    candidate_mm_hashes_by_hamming.extend(
        self.phash_to_resized_mm_hashes[candidate_phash]
    )
```

### 不漏召回的条件

将完整 pHash 等宽分成 B 段。如果两个 pHash 的汉明距离不超过 τ，且 τ < B，
最多 τ 个不同 bit 就最多影响 τ 个 band，必定至少有一段完全相同。
索引以 `(段号, 段值)` 为 key，保存拥有该分段的完整 pHash 集合。
查询所有分段桶并取并集，就包含了全部满足距离要求的 pHash；`set` 负责去重。

例如四段方案下，每组是一个 16 位 band，以下用十六进制表示：

```text
查询：0000 | 0000 | 0000 | 0000
候选：0001 | 0001 | 0001 | 0000
```

前三段各有一位不同，总汉明距离为 3，第四段完全相同，因此候选一定能被第四个桶召回。
如果四段各不同一位，汉明距离就是 4，可能没有任何相同 band；这正是四段时不能允许
`max_hamming >= 4` 而又只查询相同分段桶的原因。

共享一段只表示候选值得检查，其余位仍可能相差很大。
后续完整 `hamming()` 判断会去掉这些误候选，因此保留的仍是原来的距离条件。
按 pHash 数值排序后再展开各自的 key 列表，与原先先排序全部 pHash 再过滤的顺序一致。
这一等价关系以相同的已登记数据为前提，并要求完整映射和分段索引同步维护。

### 精确模式、索引维护与性能边界

- `max_hamming == 0`：只有完整 pHash 相等才符合要求，直接查询字典并返回 key 列表副本；
  不扫描全表、不计算汉明距离，也不填充 band 索引。
- `1 <= max_hamming <= 3`：当前四段方案查询 4 个 band 桶，合并候选后计算完整距离。
  不回退到全表检索，通过限制汉明距离保证分段召回的正确性。
- 同一批次重复 pHash 复用检索结果，同时保留输出的原有重复次数和顺序。
- 新增 pHash 时同步登记分段索引。如果以后增加删除或淘汰逻辑，也需同步移除索引项，
  否则可能遗漏候选或通过失效索引访问不存在的完整映射。
- pHash 相同或接近仍须经过布局兼容性检查与 SSIM，不能直接认定 embedding 可复用。

旧版对全部 N 个不同 pHash 计算距离，新版只对共享至少一个分段的 C 个候选计算距离，
并付出读取分段桶、合并集合和排序的开销。通常 C 小于 N；hash 分布集中时 C 仍可能接近 N，
因此不承诺固定加速比。
索引只存在于 matcher 实例内，没有新增跨进程共享、持久化或淘汰机制。
非零距离模式下，四段索引增加约 4 倍于不同 pHash 数量的集合成员关系。

## 计算 pHash、延迟保存像素与发布索引

pHash 生命周期分为两个阶段。

`ensure_cache_available()` 阶段做本地索引预登记。对没有 pHash 的 exact miss，
`_compute_phash_and_cache_metadata(mm_hash, feature)` 计算 pHash，并把它和
确定性的 resized key 写入 `similarity_cache`、`phash_to_resized_mm_hashes`
以及 band 反向索引；但不会把 resized pixel tensor 写入 Memcache。此时候选
key 可能暂时没有 Memcache 值，后续读取路径会通过 key-info 检查跳过缺失值。

`update_state_after_alloc()` 完成最终匹配后，如果结果仍是 save miss，
connector 会在把该图加入 `mm_hashes_need_saves` 之前调用
`persist_missed_resized_tensor(mm_hash, feature)`。此时匹配已经结束，
新持久化并索引的 tensor 只会成为未来请求的候选。

### 提取、计算与缓存 pHash 元数据

1. `extract_resized_tensor(feature)` 提取预处理后的像素张量，
   `extract_image_grid(feature)` 提取 `(T, H, W)` patch 网格信息。任一缺失就跳过。
2. `compute_phash(resized_tensor, grid, merge_size=...)` 计算完整 64 位 pHash，
   结果是 Python 整数。数据不符合计算要求、返回 `None` 时跳过；此时尚未分 band。
3. 生成确定性的 resized key，并在同一把状态锁内发布
   `phash_to_resized_mm_hashes`、band 索引和 `similarity_cache` 元数据；
   此阶段不执行 `_tensor_store.put()`。

### 最终 miss 后保存像素并发布索引

`persist_missed_resized_tensor()` 优先复用预登记的 pHash；若没有则从当前
feature 重新计算。随后执行：

1. `_tensor_store.put(mm_hash, resized_tensor, grid)` 确认像素已存在或成功写入 Memcache。
   返回 `False` 时保留已有的本地索引悬空项；调用方会记录失败，但仍会按原逻辑保存 embedding。
2. `get_resized_key(...)` 生成包含图片标识、grid、shape、dtype 和模型信息的存储 key，
   后续据此检查布局兼容性并读取像素。
3. 保存像素后，在同一把状态锁内确认反向索引已登记。

这里的 `mm_hash` 是原始图片标识，`image_phash` 是用于相似度粗筛的感知哈希，
`resized_key` 是 resized 像素在 Memcache 中的存储 key，三者用途不同。

### 在同一把锁内发布三个关联状态

| 状态 | 内容 | 用途 |
| --- | --- | --- |
| `phash_to_resized_mm_hashes` | 完整 pHash → resized key 列表 | 将一个感知哈希映射回所有对应图片 |
| `_phash_band_index` | `(段号, 段值)` → 完整 pHash 集合 | 按相同分段召回候选 |
| `similarity_cache` | `mm_hash` → pHash 与已知 SSIM 匹配记录 | 复用图片的计算结果 |

不同图片可以具有同一个 pHash，因此完整映射的值必须是 key 列表。
首次遇到某个 pHash 时创建空列表，并在非零距离模式下登记它的全部 band：

```python
candidate_resized_keys = self.phash_to_resized_mm_hashes.get(image_phash)
if candidate_resized_keys is None:
    candidate_resized_keys = []
    self.phash_to_resized_mm_hashes[image_phash] = candidate_resized_keys
    if self._config.max_hamming > 0:
        for band in bands(image_phash):
            self._phash_band_index.setdefault(band, set()).add(image_phash)
if resized_key not in candidate_resized_keys:
    candidate_resized_keys.append(resized_key)
```

相同 pHash 的分段完全相同，因此只在这个 pHash 首次出现时登记 band 即可。
`candidate_resized_keys` 与字典中保存的是同一个列表对象，`append()` 后无需再次赋回字典；
追加前检查 key，避免重复登记。

最后在 `mm_hash` 尚无记录时初始化 `similarity_cache` 记录，
其中 SSIM 匹配列表初始为空。已有记录会被保留，之前计算的 SSIM 结果不会被清空。

上述三个状态由同一把 `_similarity_state_lock` 保护，使后台 pHash 预登记任务
不会交错发布不一致的元数据状态。pHash 计算和 Memcache 写入在这把锁之外进行，
减少锁内工作量。
这把状态锁与下面保护 PUT 注册生命周期的 `_put_lock` 用途不同。

## CPU 预处理与传输

`perceptual_ssim_cache_matcher.py` 已移除 `has_device_inputs` 检查。
pHash 预处理仅按批量大小调度：数量不超过 3 时在调用线程执行，超过 3 时使用
最多 8 个 worker。该简化依赖像素和 grid 位于 CPU 的约定。

`memcache_resized_tensor_store.py` 中：

| 操作 | 当前行为 |
| --- | --- |
| PUT | 整理为连续 CPU 张量，固定使用 `COPY_H2G` |
| 普通接收缓冲区 | 延迟分配 CPU 缓冲区，后续复用 |
| oversized 临时接收缓冲区 | 分配 CPU 缓冲区，使用后注销 |
| GET | 固定使用 `COPY_G2H` |
| 方向工具函数 | `_transfer_direction` 重命名为 `get_transfer_direction`，保留 CPU/NPU 映射能力，但当前生产调用点不调用该函数 |

### 为什么接收缓冲区位于 CPU

```python
recv_buffer = torch.empty(
    self._recv_buffer_size,
    dtype=torch.uint8,
    device="cpu",
)
```

该缓冲区接收 resized 图片像素，供 CPU 上的 SSIM 比较使用。
`uint8` 表示按字节分配空间，`self._recv_buffer_size` 的单位是字节；
加载后再按原图片的 dtype 和 shape 解释这些数据，并非要求图片原始 dtype 为 `uint8`。

| 接收的数据 | 目标设备 | 从 Memcache 读取的方向 |
| --- | --- | --- |
| resized 图片像素 | CPU | `COPY_G2H` |
| encoder embedding | NPU | `COPY_G2L` |

方向枚举中的 G 表示 Memcache 的全局内存，并不表示 PyTorch 的 GPU 设备。
如果将接收缓冲区改成 NPU，必须同时调整传输方向与后续计算路径，不能只改 `device`。
当前固定 host 方向只针对 resized pixel tensor，embedding 的 NPU 存取逻辑不变。

读取输出继续 clone，避免接收缓冲区复用覆盖已取得的候选；异常路径保留清理。
已有 `_synchronize_npu` 防御代码保留，在当前 CPU 数据路径中不执行 NPU 同步。
pHash/SSIM 数值函数的 FP32、常量缓存及 8 字节 hash 打包实现保持不变。
SSIM 内部的设备检查本次没有调整；正常调用中查询和读取的候选均为 CPU 张量。
之前讨论的异设备 SSIM 自动对齐方案不包含在本次简化中。

## PUT 为什么保留锁

`self._put_lock = Lock()` 保护同一个 store 实例的完整 PUT 注册生命周期：

```text
获取锁 → 注册缓冲区 → 传输像素 → finally 中注销缓冲区 → 释放锁
```

CPU 数据约定不能替代这把锁。批量预登记现在只并行计算 pHash 元数据；
最终 miss 后的 `_tensor_store.put()` 在调度线程执行。保留锁可以防御
未来并发 PUT，以及 store 层其他调用路径带来的并发。

保留锁的具体原因有两个：

1. `MemcacheBackend.register_buffer()` 使用共享的 `self._pending_buffers`
   保存待注册地址，再由 `_register_buffers_if_needed()` 读取并清空。
   该封装没有对这组操作提供完整并发保护，多个调用交错时存在状态覆盖风险。
2. `tensor.detach().to(device="cpu").contiguous()` 不保证复制内存。
   `detach()` 脱离自动求导图但共享底层存储；输入已经是连续 CPU 张量时，后两步也可能
   复用原存储。如果两个调用使用同一地址，一个调用注销缓冲区时，另一个可能还在传输。

本地 `memcache-release-1.2` 的 SDK 实现对注册、注销各自使用内部互斥锁，
但这并不能保证“注册—传输—注销”三个操作作为一个整体不被其他调用插入。
也不能据此认为上层 Python 的共享注册状态已经得到保护。

这把锁的代价是同一个 store 的 PUT 临界区串行执行，pHash 元数据计算仍可并行。
当前实现保留该锁，也保留传输异常时的注销清理。

## 线程池、Future 与异常等待

CPU SSIM 评分在候选数量超过 3 时也可使用线程池。当前代码还保留设备防御判断：
只有查询灰度图和全部候选张量均在 CPU 上才进入该并行分支，设备输入走调用线程。
相关代码如下：

```python
futures = [
    self._similarity_compute_executor.submit(
        self._score_candidate,
        current_gray_plane,
        resized_key,
        candidate_tensor,
    )
    for resized_key, candidate_tensor in zip(score_resized_keys, score_tensors)
]
wait(futures)
score_results = [future.result() for future in futures]
```

- `zip(...)` 将候选 key 与对应像素张量配对，每一对提交一次 `_score_candidate()`。
  所有任务读取同一张查询灰度图 `current_gray_plane`，分别与各自候选比较。
- `submit()` 返回 Future，即任务的执行句柄，不是已经算好的分数。
  线程池按可用 worker 执行任务，超过容量的任务排队。
- `wait(futures)` 默认等待所有任务完成，包括正常返回和以异常结束的任务；
  它本身不会抛出任务内部的异常。
- `future.result()` 获取返回值，当前为 `(ssim_score, candidate_mm_hash)` 或 `None`；
  若任务失败则在这里重新抛出异常。后续会过滤 `None`。
- 结果按 Future 的提交顺序收集，与线程完成先后无关，也不等于按 SSIM 分数排序。

先等待全部完成，再逐个取结果，是为了保证异常向上传播时其他任务也已经结束。
如果直接逐个调用 `result()`，前面的任务一旦抛错，其他任务可能仍在读取张量，
而外层已经开始异常处理。等待所有任务可明确读者的结束边界，再进行外层清理。
pHash 预处理分支也采用相同的等待模式，避免后台持久化或索引更新越过本次调用生命周期。

## SSIM 阈值、排序与提前停止

加载、批次构造和评分继续使用 `ResizedTensorLoadRequest` 的
`output_index`、`resized_key`、`nbytes` 字段，每批输出索引重新编号。
model/grid/shape/dtype key 格式及布局兼容性检查保持不变。

最终接受条件为严格 `score > ssim_threshold`，与当前缓存查找、预热分类和批次早停一致。
遇到首个含合格结果的批次即可停止，选择的是已评分范围内的最佳项，
不保证未读取候选的全局最优。

### 当前排序的作用

普通批次和 oversized 批次目前都执行：

```python
calculated_scores.extend(batch_scores)
calculated_scores.sort(key=lambda item: (-item[0], item[1]))
if batch_scores and max(score for score, _ in batch_scores) > ssim_threshold:
    # 记录命中日志并返回 calculated_scores。
    ...
```

排序将 SSIM 分数高的候选放在前面，同分时按候选 `mm_hash` 排序，保证选择稳定。
`max(...)` 自身不要求列表有序，前面的 `batch_scores` 检查则避免对空列表调用 `max()`。
但其他代码依赖首项已经是最高分：

- 当前命中日志读取 `calculated_scores[0]`。
- `_get_ssim_score_list()` 用 `regular_scores[0][0]` 判断是否可跳过 oversized 加载。
- `find_and_cache_matched_ssim()` 用 `ssim_score_list[0]` 选择本次复用的 embedding key。
- 缓存写入也按分数与 key 排序，以便后续按顺序选择最高分的合格缓存候选。

因此，不能仅删除这一行排序而保持所有首项访问不变。
例如返回顺序是 0.991、0.999，阈值为 0.99，第一项虽合格却不是已评分候选中的最高分。

