## 目录结构介绍
``` 
├── kirin_add_template
│   ├── cmake                   // 编译工程文件
│   │   ├── Modules/            // CCE编译器检测模块（通用，无需修改）
│   │   └── npu/CMakeLists.txt  // NPU编译配置（通用，无需修改）
│   ├── input                   // 存放脚本生成的输入数据目录
│   ├── output                  // 存放算子运行输出数据和真值数据的目录
│   ├── scripts
│   │   ├── acl.json            // ACL初始化配置（默认空JSON）
│   │   ├── gen_data.py         // 生成输入数据和golden真值（需修改）
│   │   └── verify_result.py    // 输出与golden对比验证（需修改）
│   ├── add_custom.cpp          // 算子核函数实现（需替换）
│   ├── data_utils.h            // 通用文件读写/日志工具（无需修改）
│   ├── main.cpp                // Host侧主程序（需修改）
│   ├── CMakeLists.txt          // 顶层CMake配置（无需修改）
│   └── run.sh                  // 一键编译+执行+验证脚本（需修改）
``` 

## 代码实现介绍
本调用样例中实现的是固定shape为8*2048的Add算子。
- kernel实现   
  Add算子的数学表达式为：  
  ```
  z = x + y
  ```
  计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用计算接口完成两个输入参数相加，得到最终结果，再搬出到外部存储上。  
  
  Add算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor xGm和yGm搬运到Local Memory，分别存储在xLocal、yLocal，Compute任务负责对xLocal、yLocal执行加法操作，计算结果存储在zLocal中，CopyOut任务负责将输出数据从zLocal搬运至Global Memory上的输出Tensor zGm中。具体请参考[add_custom.cpp](./add_custom.cpp)。

- 调用实现  
  1. CPU侧运行验证主要通过ICPU_RUN_KF CPU调测宏等CPU调测库提供的接口来完成；  
  2. NPU侧运行验证主要通过使用<<<>>>内核调用符来完成。

  应用程序通过ASCENDC_CPU_DEBUG 宏区分代码逻辑运行于CPU侧还是NPU侧。

## 运行样例算子
- 打开样例目录

  ```
  cd $HOME/ops/kirin_add_template
  ```
- 配置环境变量

  这里的$HOME需要替换为本仓根目录
  ```
  export ASCEND_INSTALL_PATH=$HOME/Ascend/cann-mobile/cann-8.5.0
  ```

- 样例执行

  ```
  bash run.sh -r [RUN_MODE] -v  [SOC_VERSION] 
  ```
  - RUN_MODE：编译方式，sim（NPU仿真）。
  - SOC_VERSION：KirinX90 或者 Kirin9030。

  示例如下。
  ```
  bash run.sh -r sim -v Kirin9030
  ```   

  `run.sh` 的完整执行流程为：
  1. **解析参数**：通过 `-v`（SoC版本）、`-r`（运行模式：sim/cpu/npu）、`-i`（CANN安装路径）解析命令行参数
  2. **确定CANN路径**：按优先级从 `ASCEND_TOOLKIT_HOME` → `ASCEND_HOME_PATH` → 用户指定路径 → 默认路径查找
  3. **校验参数**：验证SoC版本必须是 KirinX90/Kirin9030，运行模式必须是 sim/cpu/npu，Kirin SoC不支持cpu模式
  4. **配置仿真环境**：如果是sim模式，设置模拟器库路径（`LD_LIBRARY_PATH`）和日志目录
  5. **编译**：清理build目录，通过CMake配置并编译，生成可执行文件
  6. **生成测试数据**：用 `gen_data.py` 生成输入和golden数据
  7. **执行**：运行编译出的可执行文件，在模拟器上执行算子
  8. **验证结果**：用 `verify_result.py` 对比输出与golden数据，确认误差在容忍范围内
  9. **清理**：删除仿真日志和vcd文件

## 基于本模板开发新自定义算子

以下以开发一个 `mul`（乘法）算子为例，说明如何将本模板改造为新算子工程。

### 步骤1：复制模板并重命名

```bash
cp -r kirin_add_template mul_template
cd mul_template
rm -rf build add_sim *_cpu *_npu cceprint npuchk *.vcd sim_log input/*.bin output/*.bin
```

### 步骤2：修改 `run.sh` — FILE_NAME

`run.sh:27` 中的 `FILE_NAME` 控制编译产物名称和CMake target名。改为新算子名：

```bash
FILE_NAME="mul"    # 原: FILE_NAME="add"
```

### 步骤3：替换 `add_custom.cpp` — 核函数实现

这是最核心的修改，整个文件替换为新算子实现。需要修改的内容：

1. **计算常量**（行19-24）：根据新算子的数据总量、核心数、分块策略调整
   ```cpp
   constexpr int32_t TOTAL_LENGTH = 8 * 2048;
   constexpr int32_t USE_CORE_NUM = 1;
   constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;
   constexpr int32_t TILE_NUM = 8;
   constexpr int32_t BUFFER_NUM = 2;
   constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM;
   ```

2. **Kernel类名和逻辑**：替换类名，修改 `Init`/`Process`/`CopyIn`/`Compute`/`CopyOut`
   - `Init`：调整 GlobalTensor 数量和类型
   - `Compute`：将 `AscendC::Add` 替换为目标计算接口，例如 `AscendC::Mul`
   - `CopyIn`/`CopyOut`：调整输入/输出队列数量和搬运逻辑

3. **核函数入口**（行84-88）：改名为新算子名，调整参数列表
   ```cpp
   extern "C" __global__ __aicore__ void mul_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
   {
       KernelMul op;
       op.Init(x, y, z);
       op.Process();
   }
   ```

4. **Host调用桥接函数**（行94-97）：改函数名，与核函数名一致
   ```cpp
   void mul_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *y, uint8_t *z)
   {
       mul_custom<<<blockDim, l2ctrl, stream>>>(x, y, z);
   }
   ```

### 步骤4：修改 `main.cpp` — Host侧调用接口

1. **extern声明**（行21/24）：改为新算子名
   ```cpp
   extern void mul_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y, uint8_t* z);
   // CPU分支:
   extern "C" __global__ __aicore__ void mul_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z);
   ```

2. **数据大小**（行30-31）：根据新算子的输入/输出shape和数据类型调整
   ```cpp
   size_t inputByteSize = 8 * 2048 * sizeof(uint16_t);   // half = uint16_t
   size_t outputByteSize = 8 * 2048 * sizeof(uint16_t);
   ```

3. **调用点**（行75）：改函数名，参数与核函数保持一致
   ```cpp
   mul_custom_do(blockDim, nullptr, stream, xDevice, yDevice, zDevice);
   ```

4. **输入数据读取**（行68-69）：如果输入参数数量/名称变化，需同步调整 `ReadFile` 的文件名

### 步骤5：修改 `scripts/gen_data.py` — 生成输入和真值

根据新算子的数学定义修改：

```python
def gen_golden_data_simple():
    input_x = np.random.uniform(1, 100, [8, 2048]).astype(np.float16)
    input_y = np.random.uniform(1, 100, [8, 2048]).astype(np.float16)
    golden = (input_x * input_y).astype(np.float16)    # 改 Add 为 Mul

    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")
```

需要修改的内容：
- **Shape**：如果算子shape不同，修改 `[8, 2048]` 及数据类型（`np.float16`/`np.float32`/`np.int32` 等）
- **输入参数数量**：增减输入文件（如3输入算子需要额外生成 `input_w.bin`）
- **计算公式**：将 `+` 改为目标运算
- **输出文件名**：如有多输出，需生成多个 golden 文件

### 步骤6：修改 `scripts/verify_result.py` — 验证逻辑

根据输出数据类型调整：

```python
# 如输出为 float32:
output = np.fromfile(output, dtype=np.float32).reshape(-1)
golden = np.fromfile(golden, dtype=np.float32).reshape(-1)

# 调整容差（float32精度更高时可用更小容差）
relative_tol = 1e-5
absolute_tol = 1e-8
error_tol = 1e-5
```

### 无需修改的文件

| 文件 | 原因 |
|------|------|
| `data_utils.h` | 通用工具函数，与算子无关 |
| `CMakeLists.txt`（顶层） | 只做 `add_subdirectory(cmake/npu)`，无算子名 |
| `cmake/npu/CMakeLists.txt` | 使用 `${smoke_testcase}` 变量，由CMake参数传入 |
| `cmake/Modules/*` | CCE编译器检测模块，通用 |
| `scripts/acl.json` | ACL配置，默认为空 `{}` |

### 常见扩展场景的修改要点

| 场景 | 需修改的文件 |
|------|-------------|
| 改数据类型（half→float） | `add_custom.cpp`（GlobalTensor/LocalTensor类型）、`main.cpp`（sizeof）、`gen_data.py`（dtype）、`verify_result.py`（dtype+容差） |
| 改Shape | `add_custom.cpp`（TOTAL_LENGTH等常量）、`main.cpp`（inputByteSize）、`gen_data.py`（shape） |
| 多输入（3个以上） | `add_custom.cpp`（增加队列和GlobalTensor）、`main.cpp`（增加malloc/read/memcpy）、`gen_data.py`（增加输入文件） |
| 多输出 | `add_custom.cpp`（增加输出队列）、`main.cpp`（增加malloc+WriteFile）、`gen_data.py`（增加golden文件）、`verify_result.py`（多输出对比） |
| 多核并行 | `add_custom.cpp`（调整USE_CORE_NUM和GetBlockIdx逻辑）、`main.cpp`（调整blockDim） |

### 关键概念速查

- **核函数**：`extern "C" __global__ __aicore__` 修饰的函数，是NPU上的执行入口
- **`_do` 桥接函数**：在核函数实现文件中用 `<<<>>>` 调用符包装核函数，供 `main.cpp` 的Host侧调用
- **`ASCENDC_CPU_DEBUG` 宏**：区分CPU调测分支和NPU/仿真分支，Kirin SoC目前仅支持NPU/仿真分支
- **双缓冲（BUFFER_NUM=2）**：通过 `InitBuffer(que, 2, size)` 开启双缓冲实现流水线，一个buffer搬运时另一个buffer计算