### Kirin Vector 算子开发指南（Add 分支）

> **适用平台**：KirinX90 / Kirin9030，仅支持 sim（仿真）和 NPU 模式，不支持 CPU 调测模式。

## 模板文件结构

Kirin 模板为单文件结构，无 PyTorch 对接层：

```
kirin_add_template/
├── add_custom.cpp          算子核函数实现（需替换）
├── add_custom_tiling.h     Tiling 常量与数据结构（需修改）
├── main.cpp                Host侧主程序（需修改）
├── data_utils.h            通用文件读写/日志工具（无需修改）
├── run.sh                  一键编译+执行+验证（需修改）
├── CMakeLists.txt          顶层CMake配置（无需修改）
├── cmake/
│   ├── Modules/            CCE编译器检测模块（无需修改）
│   └── npu/CMakeLists.txt  NPU编译配置（无需修改）
├── ext_lib/                 Kirin 运行时库目录（需初始化）
├── scripts/
│   ├── acl.json            ACL初始化配置（默认空JSON）
│   ├── gen_data.py         生成输入数据和golden真值（需修改）
│   └── verify_result.py    输出与golden对比验证（需修改）
├── input/                  脚本生成的输入数据目录
└── output/                 算子运行输出数据和真值目录
```

## Kirin 特性

- **单文件 kernel**：核函数类 + `__aicore__` 入口同在一个 `.cpp` 文件；核函数通过 `<<<>>>` 在 main.cpp 中直接调用
- **Tiling 机制**：Tiling 常量（`TILE_LENGTH`、`DOUBLE_BUFFER`）和数据结构（`AddTilingData`）定义在独立 `.h` 文件中；Host侧在 main.cpp 计算填充 Tiling 数据，通过 `aclrtMemcpy` 传入 Device
- **`ASCENDC_CPU_DEBUG` 宏**：区分 CPU 调测分支和 NPU/仿真分支；Kirin SoC 仅支持 NPU/仿真分支
- **Host 流程**：ACL 初始化 → Tiling 计算 → `aclrtMalloc`+`aclrtMemcpy` 数据搬运 → kernel `<<<>>>` 调用 → `WriteFile` 写回结果

## Simulator 环境注意事项

### PlatformAscendCManager::GetInstance() 必须指定芯片平台

**问题**：在 Simulator 模式下，`PlatformAscendCManager::GetInstance()`（无参数版本）会返回 `nullptr`，导致后续解引用时 Segmentation fault。

**原因**：
- Simulator 环境无法自动检测当前芯片型号
- `GetInstance()` 内部调用 `PlatformAscendCManagerInit(nullptr)` 失败
- 返回 `nullptr` 后，代码直接解引用导致崩溃

**解决方案**：必须传入明确的芯片平台参数
```cpp
// ❌ 错误用法（Simulator 环境会返回 nullptr）
platform_ascendc::PlatformAscendC* ascendcPlatform = 
    platform_ascendc::PlatformAscendCManager::GetInstance();
const platform_ascendc::PlatformAscendC& ascendCPlatform = *ascendcPlatform;  // Segmentation fault!

// ✅ 正确用法（传入明确的芯片平台）
platform_ascendc::PlatformAscendC* ascendcPlatform = 
    platform_ascendc::PlatformAscendCManager::GetInstance("Kirin9030");  // 或 "KirinX90"
if (ascendcPlatform == nullptr) {
    ERROR_LOG("PlatformAscendCManager::GetInstance failed");
    return -1;
}
const platform_ascendc::PlatformAscendC& ascendCPlatform = *ascendcPlatform;
```

**适用场景**：
- 使用需要 `PlatformAscendC` 对象的 Tiling API
- KirinX90 / Kirin9030 Simulator 环境

**参考位置**：
- `platform_ascendc.h` 第163-184行：`GetInstance(const char *customSocVersion)` API 定义
- 支持的 SocVersion 字符串：`"Kirin9030"`, `"KirinX90"`（见 `platform_ascendc.h` 第60-86行）

## 开发步骤

0. **工程初始化**（下载运行时依赖）：
   ```bash
   cd references/kirin_add_template
   
   # 创建 ext_lib 目录
   mkdir -p ext_lib
   
   # 下载 libstdc++.so.6.0.34（来自 conda-forge）
   # 适用于 x86-64 Linux 系统，包含 GLIBCXX_3.4.32
   curl -fsSL --connect-timeout 30 --max-time 120 \
     "https://conda.anaconda.org/conda-forge/linux-64/libstdcxx-15.2.0-h8f9b012_7.conda" \
     -o /tmp/libstdcxx.conda
   
   # 解压并提取库文件
   mkdir -p /tmp/libstdcxx-extracted
   unzip -o /tmp/libstdcxx.conda -d /tmp/libstdcxx-extracted
   tar --zstd -xf /tmp/libstdcxx-extracted/pkg-libstdcxx-*.tar.zst -C /tmp/libstdcxx-extracted/
   
   # 复制到工程目录
   cp /tmp/libstdcxx-extracted/lib/libstdc++.so.6.0.34 ext_lib/
   
   # 清理临时文件
   rm -rf /tmp/libstdcxx.conda /tmp/libstdcxx-extracted
   ```
   
   > **注意**：
   > - 此库仅适用于 **x86-64 Linux** 系统，其他架构请从对应平台下载
   > - conda-forge 提供多平台版本：`linux-aarch64`（ARM64）、`linux-ppc64le`（POWER）等
   > - 如系统已安装更高版本 libstdc++（GLIBCXX ≥ 3.4.32），可跳过此步骤
   > - 该库用于run.sh中的 export LD_PRELOAD="${SCRIPT_DIR}/ext_lib/libstdc++.so.6.0.34"使用，被当前版本的 Kirin Simulator 所依赖

1. **复制样例目录**：
   ```bash
   cp -r references/kirin_add_template <your_op>
   cd <your_op>
   rm -rf build *_sim *_cpu *_npu cceprint npuchk *.vcd sim_log input/*.bin output/*.bin
   ```

2. **全局替换算子名**：`add_custom` → `<your_op>_custom`，同时替换 `Add` → `<YourOp>`（类名、Tiling 结构名等）

3. **按步骤修改各文件**（以下以开发 `mul` 算子为例）：

   **步骤 1 — `run.sh`**：修改 `FILE_NAME`
   ```bash
   FILE_NAME="mul"    # 原: FILE_NAME="add"
   ```

   **步骤 2 — `add_custom_tiling.h` → `<your_op>_custom_tiling.h`**（Tiling 常量与结构）：
   - **常量**：调整 `TILE_LENGTH`、`DOUBLE_BUFFER`
   - **Tiling 结构名**：`AddTilingData` → `MulTilingData`，字段按需增减

   **步骤 3 — `add_custom.cpp` → `<your_op>_custom.cpp`**（核函数实现）：
   - **include**：改为新 tiling 头文件名
   - **Kernel 类名**：`KernelAdd` → `KernelMul`
   - **`Init`/`Process`/`CopyIn`/`Compute`/`CopyOut`**：修改逻辑
     - `Compute`：将 `AscendC::Add` 替换为目标计算接口（如 `AscendC::Mul`）
     - `CopyIn`/`CopyOut`：调整输入/输出队列数量和搬运逻辑
   - **核函数入口**：改名为新算子名，参数列表包含 tiling
     ```cpp
     extern "C" __global__ __aicore__ void mul_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tiling)
     ```

   **步骤 4 — `main.cpp`**（Host侧调用接口）：
   - **include**：改为新 tiling 头文件名
   - **extern声明**：改为新算子名（含 tiling 参数）
   - **数据大小**：根据新算子的 shape 和 dtype 调整 `sizeof` 倍数（当前为 `sizeof(uint16_t)` 对应 half）
   - **Tiling 计算**：按需调整 `totalLength`、`availableCoreNum` 及 Tiling 字段
   - **调用点**：改函数名，参数包含 tilingDevice
     ```cpp
     mul_custom<<<blockNum, nullptr, stream>>>(xDevice, yDevice, zDevice, tilingDevice);
     ```
   - **输入数据读取**：输入参数数量/名称变化时同步调整 `ReadFile` 的文件名

   **步骤 5 — `scripts/gen_data.py`**（生成输入和真值）：
   - **Shape**：修改 `[8, 2048]` 及 dtype（`np.float16`/`np.float32`/`np.int32` 等）
   - **输入参数数量**：增减输入文件
   - **计算公式**：将 `+` 改为目标运算
   - **输出文件名**：多输出时生成多个 golden 文件

   **步骤 6 — `scripts/verify_result.py`**（验证逻辑）：
   - 调整输出 dtype（`np.float16` → `np.float32` 等）
   - 调整容差（fp32 可用更小容差：`rtol=1e-5, atol=1e-8`）

4. **编译运行**：
   ```bash
   bash run.sh -r sim -v Kirin9030
   bash run.sh -r sim -v KirinX90
   ```
   > `run.sh` 会自动检测 `ASCEND_TOOLKIT_HOME`/`ASCEND_HOME_PATH`，无需手动 export。运行时通过 `LD_PRELOAD` 加载 `ext_lib/libstdc++.so.6.0.34`，需确保该文件已下载（见步骤0）。

## 常见扩展场景

| 场景 | 需修改的文件 |
|------|-------------|
| 改 dtype（half→float） | `*_custom.cpp`（Tensor类型）、`*_custom_tiling.h`（无需改）、`main.cpp`（sizeof）、`gen_data.py`（dtype）、`verify_result.py`（dtype+容差） |
| 改 Shape | `main.cpp`（totalLength + Tiling 计算）、`gen_data.py`（shape） |
| 多输入（3个以上） | `*_custom.cpp`（增加队列和GlobalTensor）、`*_custom_tiling.h`（无需改）、`main.cpp`（增加malloc/read/memcpy）、`gen_data.py`（增加输入文件） |
| 多输出 | `*_custom.cpp`（增加输出队列）、`main.cpp`（增加malloc+WriteFile）、`gen_data.py`（增加golden文件）、`verify_result.py`（多输出对比） |
| 多核并行 | `*_custom.cpp`（GetBlockIdx 逻辑已有）、`main.cpp`（调整 availableCoreNum） |