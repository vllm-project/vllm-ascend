# API 文档索引

基于 `$ASC_DEVKIT_DIR/docs/zh/api/` 的完整 API 文档索引。

---

## 文档位置

```
$ASC_DEVKIT_DIR/docs/zh/api/  — Ascend C API 文档根目录
```

> 子目录结构随 CANN 版本演进有变化（如 `context/` 扁平结构 → `SIMD-API/SIMT-API` 层级结构）。
> **不要假设具体的子目录名**，统一用 `find "$ASC_DEVKIT_DIR/docs/zh/api/" -name "{APIName}*.md"` 搜索。

---

## 一、基础数据结构

| API | 说明 |
|-----|------|
| `LocalTensor` | 存放 AI Core 中 Local Memory 的数据 |
| `GlobalTensor` | 存放 Global Memory 的全局数据 |
| `Coordinate` | 表示张量在不同维度的位置信息 |
| `Layout` | 描述多维张量内存布局的基础模板类 |
| `TensorTrait` | 描述 Tensor 相关信息的基础模板类 |

---

## 二、基础 API

### 表2：标量计算 API
| API | 说明 |
|-----|------|
| `ScalarGetCountOfValue` | 获取标量值计数 |
| `ScalarCountLeadingZero` | 计数前导零 |
| `ScalarCast` | 标量类型转换 |

### 表3：矢量计算 API
| 类别 | API |
|-----|-----|
| 算术运算 | `Add`、`Sub`、`Mul`、`Div`、`Abs` |
| 三角函数 | `Sin`、`Cos`、`Tan`、`Asin`、`Acos`、`Atan` |
| 指数对数 | `Exp`、`Log`、`Sqrt`、`Rsqrt` |
| 比较运算 | `Greater`、`Less`、`Equal`、`NotEqual` |
| 逻辑运算 | `And`、`Or`、`Not`、`Xor` |

### 表4：数据搬运 API
| API | 说明 | 对齐要求 |
|-----|------|---------|
| `DataCopy` | 数据拷贝 | 512 字节 |
| `DataMove` | 数据移动 | 32 字节 |

### 表5：资源管理 API
| API | 说明 |
|-----|------|
| `MemAlloc` | 内存分配 |
| `MemFree` | 内存释放 |

### 表6：同步控制 API
| API | 说明 |
|-----|------|
| `Sync` | 同步等待 |
| `Barrier` | 屏障同步 |

### 表7：缓存处理 API
| API | 说明 |
|-----|------|
| `Cache` | 缓存控制操作 |

### 表8：系统变量访问 API
| API | 说明 |
|-----|------|
| `GetSysVar` | 获取系统变量 |

### 表9：原子操作接口
| API | 说明 |
|-----|------|
| `AtomicAdd`、`AtomicSub`、`AtomicMin`、`AtomicMax` | 原子算术操作 |

### 表10：调试接口
| API | 说明 |
|-----|------|
| `Debug` | 调试相关操作 |

### 表11：工具函数接口
| API | 说明 |
|-----|------|
| 通用工具函数 | 各种辅助函数 |

### 表12：Kernel Tiling 接口
| API | 说明 |
|-----|------|
| `GetTilingKey` | 获取 Tiling Key |
| `SetTilingKey` | 设置 Tiling Key |

### 表13：ISASI 接口
| API | 说明 |
|-----|------|
| 硬件体系结构相关接口 | 底层硬件访问 |

---

## 三、高阶 API

> 实际目录 `docs/zh/api/SIMD-API/高阶API/` 下含 13 个子类，下表为索引。HCCL 通信类详见[第七章](#七hccl-通信-api)。

| 子目录 | 类别 | 典型 API |
|--------|------|---------|
| `数学计算/` | 三角/双曲/位运算/类型转换 | `Acos`、`Acosh`、`Cos`、`Cast`、`BitwiseAnd`、`BitwiseOr`、`Addcdiv`、`Addsub` |
| `量化操作/` | 量化/反量化 | 量化相关操作 |
| `归约操作/` | 归约 | `ReduceMax`、`ReduceSum` |
| `排序操作/` | 排序 | `Sort`、`TopK` |
| `张量变换/` | 张量重排 | `Broadcast`、`Transpose` |
| `归一化操作/` | 归一化 | `LayerNorm` 相关 |
| `激活函数/` | 激活 | `Relu`、`Sigmoid`、`Gelu` |
| `矩阵计算/` | 矩阵 | `Mmad` 相关 |
| `卷积计算/` | 卷积 | 卷积相关 |
| `索引计算/` | 索引 | 索引相关 |
| `数据过滤/` | 过滤 | 数据过滤相关 |
| `随机函数/` | 随机 | 随机数生成 |
| `HCCL通信类/` | 集合通信 | 详见[第七章](#七hccl-通信-api) |

---

## 四、Utils API

> 实际目录 `docs/zh/api/Utils-API/`，含调测接口（printf、asc_dump）等。用 `find "$ASC_DEVKIT_DIR/docs/zh/api/Utils-API/" -name "*.md"` 查阅。

---

## 五、AI CPU API

> 实际目录 `docs/zh/api/AI-CPU-API/`。用 `find "$ASC_DEVKIT_DIR/docs/zh/api/AI-CPU-API/" -name "*.md"` 查阅。

---

## 六、C API

| 类别 | 说明 |
|-----|------|
| `atomic/` | 原子操作 C API |
| `cache_ctrl/` | 缓存控制 C API |
| `cube_compute/` | Cube 计算 C API |
| `vector_compute/` | 矢量计算 C API |

---

## 七、HCCL 通信 API

HCCL（集合通信）API 文档位于 `docs/zh/api/SIMD-API/高阶API/HCCL通信类/`，分三个子目录：

| 子目录 | 内容 | 典型 API |
|--------|------|---------|
| `HCCL-Kernel侧接口/` | Kernel 侧通信原语 | `Hccl::InitV2`、`Hccl::AlltoAllV`、`Hccl::Wait`、`Hccl::Finalize` |
| `HCCL-Tiling侧接口/` | Host 侧 Tiling 配置 | `Mc2CcTilingConfig`、`SetCcTilingV2` |
| `HCCL-Context/` | 通信上下文 | `HcclCombineOpParam` |

**建议先读使用说明**，获取完整调用流程和代码示例，再按需查阅单个 API 文档：
- `HCCL-Kernel侧接口/HCCL使用说明.md` — Kernel 侧 6 步调用流程（InitV2 → SetCcTilingV2 → Prepare → Commit → Wait → Finalize），含完整 Kernel 代码示例
- `HCCL-Tiling侧接口/HCCL-Tiling使用说明.md` — Tiling 侧配置流程（创建 Mc2CcTilingConfig → Set 系列配置 → GetTiling），含代码示例
- `HCCL-Context/HCCL-Context简介.md` — 通信上下文 GetHcclContext/SetHcclContext 说明

查找命令（路径含中文，必须加引号）：
```bash
find "$ASC_DEVKIT_DIR/docs/zh/api/SIMD-API/高阶API/HCCL通信类/" -name "*.md"
```

HCCL 头文件另见 `$ASC_DEVKIT_DIR/include/adv_api/hccl/`（`hccl.h`、`hccl_common.h`、`hccl_tiling.h`、`hccl_tilingdata.h`）。

---

## 使用建议

1. **API 文档查找**：
   ```bash
   find "$ASC_DEVKIT_DIR/docs/zh/api/" -name "${APIName}*.md"
   ```
   （不依赖具体子目录结构）

2. **查阅 API 文档时注意**：
   - **Restriction 章节**：查看使用限制和对齐要求
   - **Parameters 章节**：确认参数类型和范围
   - **Returns 章节**：了解返回值含义
   - **Example 章节**：参考使用示例

3. **常见对齐要求**：
   - 大多数操作：32 字节对齐
   - DataCopy：512 字节对齐
   - 某些特殊 API：64/128 字节对齐

---

## 相关资源

- [示例代码目录](example-catalog.md)
- [环境兼容性表](compatibility.md)
