# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# 专家部署流水线说明文档

**作者：陶壮**  
**更新日期：2025年7月8日**  


## 概述
本流水线是为混合专家模型（Mixture of Experts, MoE）设计的专家部署优化工具，旨在从专家激活数据生成跨 rank（如 GPU 或计算节点）的专家放置模式，并分析负载均衡效果。流水线支持两种放置策略：
- **仅重新排列（rearrange-only）**：确保每个专家在每层仅部署一次，通过优化分配实现负载均衡。
- **冗余（redundant）**：允许专家在高负载层多次部署，以进一步优化性能。

流水线从日志文件（`.log`）或文本文件（`.txt`）生成激活计数 CSV 文件，基于此生成放置模式 `.npy` 文件，验证模式的有效性，生成可视化视图，并分析负载均衡效果。整个流水线适用于分布式计算环境中 MoE 模型的专家部署优化，广泛应用于高性能计算和深度学习场景。

流水线包含以下主要步骤：
1. 从输入文件提取激活数据，生成每层专家激活计数的 CSV 文件。
2. 根据激活数据生成优化后的专家放置模式，支持仅重新排列和冗余两种模式。
3. 验证放置模式的有效性，并生成可视化视图以展示部署分布。
4. 分析放置模式的负载均衡效果，生成热图、柱状图和量化分析 CSV 文件。

## 使用方式

### 1. 环境准备
在运行流水线之前，请确保以下环境要求已满足：
- **Python 环境**：Python 3.6 或以上版本。
- **依赖库**：安装以下 Python 库以支持数据处理、可视化和统计分析：
  ```bash
  pip install numpy pandas matplotlib seaborn scipy
  ```
- **系统要求**：
  - **Linux/MacOS**：支持 Bash 的系统以运行 `pattern_generation_pipeline.sh` 脚本。
  - **Windows**：可直接运行 `pipeline.py`，但需手动配置参数。
- **字体支持**：
  - 为确保可视化中的中文字符（例如标题和标签）正确显示，建议安装支持中文的字体（如 SimHei、Microsoft YaHei 或 Arial Unicode MS）。
  - 如果未安装相关字体，可视化图像可能显示乱码，但不影响数据处理或文件生成。
- **输入数据准备**：
  - **日志模式（`input_mode=log`）**：准备 `.log` 文件，记录专家激活数据，格式见“输入文件格式”中的“日志文件格式”。
  - **文本模式（`input_mode=txt`）**：准备包含 `.txt` 文件的文件夹，每个文件对应一个 rank 的激活数据，例如 `/path/to/input/txt/folder`，格式见“输入文件格式”中的“文本文件格式”。

### 2. 配置参数
流水线通过命令行参数或修改 `pattern_generation_pipeline.sh` 中的默认参数进行配置。以下是一些典型的使用示例：

- **文本模式示例（流水线）**：
  ```bash
  ./pattern_generation_pipeline.sh \
    --input_txt_folders "/path/to/input/txt/folder" \
    --input_mode txt \
    --num_ranks_of_collecting_data 1 \
    --num_ranks_target_pattern 256 \
    --pattern_mode all \
    --collecting_modes decode \
  ```

- **参数说明**：
  - 流水线参数可以直接在 `pattern_generation_pipeline.sh` 中修改默认值，或通过命令行传递。
  - 参数值中包含空格（如文件路径）需用双引号括起来，例如 `--input_txt_folders "/path/to/input/txt/folder"`.
  - 详见“参数说明”部分，了解每个参数的作用和默认值。

### 3. 运行流水线
流水线支持两种运行方式：

- **使用 Bash 脚本（推荐，适用于 Linux/MacOS）**：
  1. 确保 `pattern_generation_pipeline.sh` 和所有 Python 脚本（`pipeline.py`, `step_1_generate_csv_with_ceiling.py`, `step_2_placement_pattern_generation.py`, `step_3_placement_pattern_checking_and_plot.py`, `step_4_load_analysis_and_plot.py`）位于同一目录。
  2. 赋予脚本执行权限：
     ```bash
     chmod +x pattern_generation_pipeline.sh
     ```
  3. 运行脚本：
     ```bash
     ./pattern_generation_pipeline.sh
     ```
  4. 可通过命令行参数覆盖默认配置，例如：
     ```bash
     ./pattern_generation_pipeline.sh --input_txt_folders "/path/to/input/txt/folder" --pattern_mode all
     ```

- **直接运行 Python 脚本（适用于所有系统）**：
  1. 确保依赖库已安装，所有 Python 脚本位于同一目录。
  2. 使用 `python` 命令运行 `pipeline.py`，并指定参数，例如：
     ```bash
     python pipeline.py \
       --input_txt_folders "/path/to/input/txt/folder" \
       --input_mode txt \
       --num_ranks_of_collecting_data 1 \
       --num_ranks_target_pattern 256 \
       --pattern_mode all \
       --collecting_modes decode \
       --recordstep_range 0:100000
     ```

### 4. 查看输出
流水线运行完成后，会在指定目录生成以下输出文件：

- **激活计数 CSV 文件**：
  - 路径：`topk_id_count_dir`（默认 `./topk_id_count`）。
  - 示例：`./topk_id_count/topk_ids_count_<timestamp>_<collecting_modes>_step<start>to<end>.csv`。
  - 内容：每层每个专家的激活计数，用于后续放置模式生成。
- **放置模式文件**：
  - 路径：`placement_pattern_dir`（默认 `./placement_pattern`）。
  - 示例：
    - 冗余模式：`./placement_pattern/placement_pattern_<timestamp>_<num_redundant_layers>_redundant_layers_<num_layers_target_pattern>_layers_<num_ranks_target_pattern>_ranks_<suffix>.npy`。
    - 重新排列模式：`./placement_pattern/placement_pattern_<timestamp>_<num_redundant_layers>_rearrange_layers_<num_layers_target_pattern>_layers_<num_ranks_target_pattern>_ranks_<suffix>.npy`。
  - 内容：三维 NumPy 数组，记录专家在各 rank 和层的部署情况。
- **放置模式可视化图像**：
  - 路径：`placement_pattern_view_dir`（默认 `./placement_pattern_view`）。
  - 示例：`./placement_pattern_view/placement_pattern_<timestamp>_<num_redundant_layers>_<mode>_layers_<num_layers_target_pattern>_layers_<num_ranks_target_pattern>_ranks_<suffix>.png`。
  - 内容：包含两个子图，展示专家部署次数和 rank 专家数量的分布。
- **负载分析图像**：
  - 路径：`placement_pattern_analysis_dir`（默认 `./placement_pattern_analysis`）。
  - 示例：
    - 热图：`./placement_pattern_analysis/Heat_<num_ranks>_Ranks_<dataset_name>.png`。
    - 柱状图：`./placement_pattern_analysis/Bars_<num_ranks>_Ranks_<dataset_name>.png`。
  - 内容：展示各层各 rank 的负载分布和最大负载比较。
- **负载分析 CSV 文件**：
  - 路径：`placement_pattern_analysis_dir`（默认 `./placement_pattern_analysis`）。
  - 示例：`./placement_pattern_analysis/Max_Load_Reduction_<num_ranks>_Ranks_<dataset_name>.csv`。
  - 内容：量化不同放置模式的负载均衡效果，包含负载降低百分比。
- **日志文件**：
  - 路径：当前目录。
  - 示例：`pattern_generation_pipeline_<timestamp>.log`。
  - 内容：记录运行的详细日志，包括执行信息、警告和错误。

详见“输出文件详细说明”部分，了解每个文件的格式和用途。

## 输入文件格式

### 1. 日志文件格式（`input_mode=log`）
日志文件（`.log`）记录专家激活数据，每行表示某 rank 在某层的激活计数，典型格式如下：

```
[dump activation] prefill step 0 in rank 0 for layer 0 get 8 experts data: 128\t256\t0\t512\t384\t64\t192\t320
[dump activation] decode step 1 in rank 1 for layer 1 get 8 experts data: 96\t160\t32\t448\t288\t80\t224\t352
```

- **格式说明**：
  - 每行以 `[dump activation]` 开头，包含以下字段：
    - **模式**：`prefill`（预填充）或 `decode`（解码）。
    - **步骤号**（`step`）：训练或推理的步骤编号。
    - **rank ID**（`rank`）：数据收集的 rank 编号（0 到 `num_ranks_of_collecting_data-1`）。
    - **层 ID**（`layer`）：MoE 层编号（0 到 `num_layers-1`）。
    - **专家数量**（`expert_count`）：当前 rank 的专家数据数量。
    - **专家激活数据**：制表符（`\t`）分隔的整数，表示专家激活计数。
  - 激活数据表示每个专家的激活次数，通常为非负整数。
  - 专家数量需满足：`expert_count * num_ranks_of_collecting_data = num_positions_of_routed_experts`。
  - 文件需涵盖所有层（0 到 `num_layers-1`）和所有 rank（0 到 `num_ranks_of_collecting_data-1`）。
- **编码**：支持 UTF-8 和 GBK 编码，流水线会自动尝试两种编码。
- **过滤**：通过 `collecting_modes` 参数（`prefill`、`decode` 或 `all`）和 `recordstep_range` 参数（格式如 `0:100000`）过滤所需模式和步骤范围的数据。

### 2. 文本文件格式（`input_mode=txt`）
文本文件（`.txt`）位于指定文件夹（如 `/path/to/input/txt/folder`），每个文件对应一个 rank 的激活数据，文件名为 `activation_counts_recordstep_*_rank_<rank_id>.txt`。典型文件内容如下：

```
128\t256\t0\t512\t384\t64\t192\t320
96\t160\t32\t448\t288\t80\t224\t352
...
```

- **格式说明**：
  - **文件名**：格式为 `activation_counts_recordstep_*_rank_<rank_id>.txt`，其中：
    - `<rank_id>`：rank 编号（0 到 `num_ranks_of_collecting_data-1`）。
    - `*`：任意步骤标识符（如 `100`）。
  - **文件内容**：
    - 每个文件包含 `num_layers` 行，对应模型的层数（默认 58）。
    - 每行包含 `num_positions_of_routed_experts / num_ranks_of_collecting_data` 个整数（制表符分隔），表示该层该 rank 的专家激活计数。
  - **数据要求**：
    - 每行数据数量需与 `num_positions_of_routed_experts / num_ranks_of_collecting_data` 一致（默认 256 / 32 = 8）。
    - 数据为非负整数，表示激活次数。
  - **编码**：目前仅支持 UTF-8 编码。
  - **模式信息**：文本文件不包含 `prefill` 或 `decode` 模式信息，`collecting_modes` 参数仅用于输出文件名（如 CSV 文件名后缀），用户需确保指定的值与数据实际来源一致。
  - **步骤过滤**：通过 `recordstep_range` 参数（如 `0:100000`）过滤特定步骤范围的文件。

## 参数说明
以下为 `pattern_generation_pipeline.sh`、`pipeline.py` 支持的参数、作用、默认值及说明：

### 流水线参数（pattern_generation_pipeline.sh 和 pipeline.py）
| 参数                              | 作用                                              | 默认值                                   | 说明                                                                 |
|-----------------------------------|--------------------------------------------------|-----------------------------------------|----------------------------------------------------------------------|
| `--input_log_files`               | 日志文件路径列表（`input_mode=log` 时使用）       | `./dump_to_log-1.log ./dump_to_log-2.log` | 指定一个或多个 `.log` 文件路径，需用空格分隔，包含空格的路径需加引号。 |
| `--input_txt_folders`             | 包含 `.txt` 文件的文件夹路径列表（`input_mode=txt` 时使用） | `/path/to/input/txt/folder` | 指定包含 `activation_counts_recordstep_*_rank_<rank_id>.txt` 的文件夹。 |
| `--input_mode`                    | 输入模式                                         | `txt`                                   | `log`：使用日志文件；`txt`：使用文本文件。                            |
| `--topk_id_count_dir`             | 激活计数 CSV 输出目录                            | `./topk_id_count`                       | 存储生成的 CSV 文件，记录每层专家激活计数。                           |
| `--placement_pattern_dir`         | 放置模式 `.npy` 文件输出目录                     | `./placement_pattern`                   | 存储三维放置模式数组文件。                                           |
| `--placement_pattern_view_dir`    | 放置模式可视化图像输出目录                       | `./placement_pattern_view`              | 存储可视化 PNG 图像，展示专家部署分布。                              |
| `--placement_pattern_analysis_dir`| 负载分析图像及 CSV 输出目录                      | `./placement_pattern_analysis`          | 存储负载分布热图、柱状图和分析 CSV 文件。                            |
| `--output_csv`                    | 输出 CSV 文件名                                  | 空（自动生成时间戳命名）                | 若为空，生成 `topk_ids_count_<timestamp>_<collecting_modes>_step<start>to<end>.csv`。 |
| `--num_layers`                    | 模型层数                                         | `58`                                    | MoE 模型的层数，需与输入数据匹配。                                   |
| `--num_ranks_of_collecting_data`  | 数据收集的 rank 数                               | `1`                                    | 输入数据对应的 rank 数量，需为正整数。版本相关：在 ≤0.2.0 版本中，`num_ranks_of_collecting_data` 支持任意正整数；在 ≥0.3.0 版本中，该参数固定为 1。                               |
| `--num_positions_of_routed_experts`| 路由专家位置数                                   | `256`                                   | 每层的专家总数，需能被 `num_ranks_of_collecting_data` 整除。          |
| `--num_ranks_target_pattern`      | 目标放置模式的 rank 数                           | `256`                                   | 放置模式的目标 rank 数量，需为正整数。                               |
| `--num_redundant_layers` | 指定优化分配的层数列表 | `58` | 在 redundant 模式下，指定哪些层被视为高负载层（high_load_layers），这些层将允许专家多次部署（最多 1 + expert_redundant_limit 次），通过 allocate_expert_deployments_improved 和 distribute_experts_to_ranks 分配，budget_limit 设为 num_ranks_target_pattern；非高负载层同样使用优化分配，但 budget_limit=0，限制为每个专家部署一次。在 rearrange 模式下，指定哪些层为高负载层，使用 allocate_expert_deployments_improved 和 distribute_experts_to_ranks 进行优化分配（每个专家部署一次）；非高负载层使用 distribute_experts_sequentially 进行顺序分配。支持多个值，空格分隔。 |
| `--expert_redundant_limit`        | 每个专家最大额外部署次数                         | `199`                                   | 冗余模式下，专家总部署次数上限为 `1 + expert_redundant_limit`。      |
| `--num_layers_target_pattern`     | 目标放置模式的层数                               | `58`                                    | 放置模式的层数，通常与 `num_layers` 相同。                           |
| `--num_eps_target_pattern`        | 每层专家数                                       | `256`                                   | 放置模式的专家数量，需能被 `num_ranks_target_pattern` 整除。         |
| `--dataset_name`                  | 数据集名称，用于输出文件命名                     | `dataset_name` | 用于标识输出文件的数据集，影响可视化和分析文件名。                   |
| `--output_file_prefix`            | 输出文件名前缀                                   | `output_file_prefix` | 用于放置模式文件命名，增加文件可读性。                               |
| `--pattern_mode`                  | 放置模式生成方式                                 | `all`                                   | `rearrange`：仅重新排列；`redundant`：允许冗余；`all`：两者都生成。   |
| `--collecting_modes`              | 数据收集模式                                     | `decode`                                | `prefill`：预填充；`decode`：解码；`all`：两者均处理（仅日志模式）。 |
| `--recordstep_range`              | 步骤范围                                         | `0:100000`                              | 格式为 `start:end`，过滤特定步骤范围的数据（如 `0:100000`）。         |

- **注意事项**：
  - **输入模式匹配**：
    - `input_mode=log` 时，必须提供 `input_log_files`，`input_txt_folders` 会被忽略。
    - `input_mode=txt` 时，必须提供 `input_txt_folders`，`input_log_files` 会被忽略。
    - 统计分析仅支持 `txt` 模式，需提供 `input_txt_folder`。
  - **收集模式**：
    - 在 `log` 模式下，`collecting_modes` 用于过滤日志行（仅处理 `prefill`、`decode` 或两者）。
    - 在 `txt` 模式下，`collecting_modes` 仅影响输出 CSV 文件名，用户需确保与数据模式一致。
  - **参数一致性**：
    - `num_positions_of_routed_experts` 必须能被 `num_ranks_of_collecting_data` 整除。
    - `num_eps_target_pattern` 必须能被 `num_ranks_target_pattern` 整除。
    - 所有数值参数（如 `num_layers`、`num_ranks_of_collecting_data` 等）必须为正整数。
    - `layer_id` 必须在有效范围内（0 到 `num_layers-1`）。

## 脚本功能
流水线由以下脚本组成，每个脚本负责特定步骤，协同完成专家部署优化和统计分析任务：

### 1. `step_1_generate_csv_with_ceiling.py`
- **功能**：
  - 从 `.log` 文件或 `.txt` 文件夹提取专家激活数据。
  - 对激活计数进行上界处理（除以 128 后取上界），生成每层专家激活计数的 CSV 文件。
- **输入**：
  - `log` 模式：一个或多个 `.log` 文件，包含激活数据。
  - `txt` 模式：包含 `activation_counts_recordstep_*_rank_<rank_id>.txt` 文件的文件夹。
- **输出**：
  - CSV 文件，存储在 `topk_id_count_dir`，记录每层每个专家的激活计数。
- **特点**：
  - 支持 `log` 和 `txt` 两种输入模式，通过 `input_mode` 参数切换。
  - 支持 `recordstep_range` 参数，过滤特定步骤范围的数据。
  - `log` 模式支持 UTF-8 和 GBK 编码，自动尝试解码；`txt` 模式仅支持 UTF-8。
  - 对激活计数进行标准化处理（除以 128 取上界），确保数据一致性。
  - 包含详细的输入验证和错误提示，确保输入数据格式正确。

### 2. `step_2_placement_pattern_generation.py`
- **功能**：
  - 根据激活计数 CSV 文件生成专家放置模式，支持仅重新排列（rearrange-only）和冗余（redundant）两种模式。
  - 优化高负载层的专家分配，减少负载不均。
- **输入**：
  - Step 1 生成的 CSV 文件，包含每层专家激活计数。
- **输出**：
  - 三维 `.npy` 文件，存储在 `placement_pattern_dir`，形状为 `(num_ranks_target_pattern, num_layers_target_pattern, num_eps_target_pattern)`。
- **特点**：
  - **仅重新排列模式**：每个专家每层部署一次，通过排序优化分配。
  - **冗余模式**：允许专家多次部署（最多 `1 + expert_redundant_limit` 次），使用堆排序优化高负载专家的分配。
  - 针对高负载层（由 `num_special_layers` 指定）进行优化分配，其余层采用顺序分配。

### 3. `step_3_placement_pattern_checking_and_plot.py`
- **功能**：
  - 验证放置模式的有效性，确保满足以下条件：
    1. 每层每个专家至少部署一次。
    2. 每层每个 rank 部署相同数量的专家。
  - 生成可视化视图，展示专家部署分布。
- **输入**：
  - Step 2 生成的 `.npy` 文件，包含放置模式。
- **输出**：
  - 验证结果：记录在日志文件中，指示是否满足条件。
  - 可视化图像：存储在 `placement_pattern_view_dir`，包含两个子图：
    - 沿 rank 轴求和：显示每个专家在各层的部署次数（层 ID × 专家 ID）。
    - 沿专家轴求和：显示每个 rank 在各层的专家数量（层 ID × rank ID）。
- **特点**：
  - 使用离散颜色映射（`plasma`）增强可视化效果。
  - 支持保存图像到文件或直接显示（由 `fig_save_path` 参数控制）。

### 4. `step_4_load_analysis_and_plot.py`
- **功能**：
  - 分析放置模式的负载分布，比较优化部署与默认部署（`Baseline`）的负载均衡效果。
  - 生成热图、柱状图和量化分析 CSV 文件。
- **输入**：
  - Step 1 的 CSV 文件：提供负载数据。
  - Step 2 的 `.npy` 文件：提供放置模式。
- **输出**：
  - **热图**：存储在 `placement_pattern_analysis_dir`，显示各层各 rank 的负载分布。
  - **柱状图**：存储在 `placement_pattern_analysis_dir`，比较各层最大负载，包含理想负载基线（`Best EP`）。
  - **分析 CSV 文件**：存储在 `placement_pattern_analysis_dir`，记录各模式的负载均衡度和相对于基线的负载降低百分比。
- **特点**：
  - 支持多模式比较（`Baseline` 和多个优化模式）。
  - 使用 `seaborn` 库生成高质量热图，颜色范围统一以便比较。

### 5. `pipeline.py`
- **功能**：
  - 协调整个流水线，依次调用 Step 1 至 Step 4 的脚本，完成从数据提取到负载分析的全流程。
- **输入**：
  - 命令行参数，指定输入文件、输出目录、模型参数等。
- **输出**：
  - 所有中间和最终文件（CSV、`.npy`、PNG、日志文件）。
- **特点**：
  - 支持批量处理多种 `num_redundant_layers` 和 `pattern_mode`。
  - 自动创建输出目录，规范化文件路径。
  - 记录详细日志，存储在 `pattern_generation_pipeline_<timestamp>.log`。

### 6. `pattern_generation_pipeline.sh`
- **功能**：
  - 提供简化的命令行接口，运行 `pipeline.py`，支持默认参数和自定义参数。
- **输入**：
  - 命令行参数或脚本中定义的默认值。
- **输出**：
  - 流水线运行结果，包含所有输出文件。
- **特点**：
  - 使用 GNU `getopt` 解析命令行参数，支持灵活的参数传递。
  - 验证输入参数的有效性（如 `input_mode`、`pattern_mode`、`collecting_modes`）。
  - 检查 Python 环境和脚本文件是否存在，确保运行前准备就绪。

## 输出文件详细说明
流水线和统计分析脚本生成以下输出文件，存储在指定目录中，文件格式和命名规则如下：

### 1. 激活计数 CSV 文件
- **路径**：`topk_id_count_dir`（默认 `./topk_id_count`）。
- **示例**：`./topk_id_count/topk_ids_count_<timestamp>_<collecting_modes>_step<start>to<end>.csv`。
- **格式**：CSV 文件，包含 `num_layers` 行和 `num_positions_of_routed_experts + 1` 列。
- **结构**：
  - **第一列**：层编号（`layer_0` 到 `layer_{num_layers-1}`）。
  - **后续列**：专家激活计数（`ep_0` 到 `ep_{num_positions_of_routed_experts-1}`），值为非负整数（经上界处理）。
- **命名规则**：
  - 格式：`topk_ids_count_<timestamp>_<collecting_modes>_step<start>to<end>.csv`。
- **用途**：
  - 记录每层每个专家的激活计数，作为 Step 2 生成放置模式的输入。

### 2. 放置模式文件
- **路径**：`placement_pattern_dir`（默认 `./placement_pattern`）。
- **示例**：
  - 冗余模式：`./placement_pattern/placement_pattern_<timestamp>_<num_redundant_layers>_redundant_layers_<num_layers_target_pattern>_layers_<num_ranks_target_pattern>_ranks_<suffix>.npy`。
  - 重新排列模式：`./placement_pattern/placement_pattern_<timestamp>_<num_redundant_layers>_rearrange_layers_<num_layers_target_pattern>_layers_<num_ranks_target_pattern>_ranks_<suffix>.npy`。
- **格式**：三维 NumPy 数组（`.npy` 文件），形状为 `(num_ranks_target_pattern, num_layers_target_pattern, num_eps_target_pattern)`。
- **结构**：
  - 值为 `0` 或 `1`，表示某 rank 的某层是否部署了某专家。
- **命名规则**：
  - 格式：`placement_pattern_<timestamp>_<num_redundant_layers>_<mode>_layers_<num_layers_target_pattern>_layers_<num_ranks_target_pattern>_ranks_<suffix>.npy`。
- **用途**：
  - 描述专家在各 rank 和层的部署模式，供 Step 3 验证和 Step 4 负载分析使用。

### 3. 放置模式可视化图像
- **路径**：`placement_pattern_view_dir`（默认 `./placement_pattern_view`）。
- **示例**：`./placement_pattern_view/placement_pattern_<timestamp>_<num_redundant_layers>_<mode>_layers_<num_layers_target_pattern>_layers_<num_ranks_target_pattern>_ranks_<suffix>.png`。
- **格式**：PNG 图像，包含两个子图。
- **结构**：
  - **左图**：沿 rank 轴求和，显示每个专家在各层的部署次数。
  - **右图**：沿专家轴求和，显示每个 rank 在各层的专家数量。
- **命名规则**：
  - 与放置模式文件一致，去掉 `.npy` 后缀，添加 `.png` 后缀。
- **用途**：
  - 直观展示放置模式的分布情况，便于检查专家部署的均匀性。

### 4. 负载分析图像
- **路径**：`placement_pattern_analysis_dir`（默认 `./placement_pattern_analysis`）。
- **示例**：
  - 热图：`./placement_pattern_analysis/Heat_<num_ranks>_Ranks_<dataset_name>.png`。
  - 柱状图：`./placement_pattern_analysis/Bars_<num_ranks>_Ranks_<dataset_name>.png`。
- **格式**：PNG 图像，包含热图和柱状图。
- **结构**：
  - **热图**：显示各层各 rank 的负载分布。
  - **柱状图**：比较各层最大负载，包含理想负载基线（`Best EP`）。
- **命名规则**：
  - 格式：`<type>_<num_ranks>_Ranks_<dataset_name>.png`（`type` 为 `Heat` 或 `Bars`）。
- **用途**：
  - 热图展示负载分布均匀性，柱状图比较优化效果。

### 5. 负载分析 CSV 文件
- **路径**：`placement_pattern_analysis_dir`（默认 `./placement_pattern_analysis`）。
- **示例**：`./placement_pattern_analysis/Max_Load_Reduction_<num_ranks>_Ranks_<dataset_name>.csv`。
- **格式**：CSV 文件，包含以下列：
  - **Placement Method**：放置模式名称。
  - **Load Balance Degree**：最大负载总和。
  - **Reduction Percentage**：相对于基线的负载降低百分比。
- **命名规则**：
  - 格式：`Max_Load_Reduction_<num_ranks>_Ranks_<dataset_name>.csv`。
- **用途**：
  - 量化负载均衡效果，提供精确的性能指标。

### 6. 日志文件
- **路径**：当前目录。
- **示例**：
  - 流水线：`pattern_generation_pipeline_<timestamp>.log`。
  - 统计分析：`layer_stats_pipeline_<timestamp>.log`。
- **格式**：文本文件，记录运行的详细日志。
- **内容**：
  - 执行信息、警告和错误，包含时间戳和日志级别。
- **用途**：
  - 提供运行过程的完整记录，便于调试和错误排查。

## 注意事项
为确保流水线和统计分析脚本顺利运行并获得正确结果，请注意以下事项：

- **输入文件格式**：
  - **日志模式**：确保 `.log` 文件符合 `[dump activation]` 格式，涵盖所有层和 rank。
  - **文本模式**：确保 `.txt` 文件名为 `activation_counts_recordstep_*_rank_<rank_id>.txt`，数据为制表符分隔的非负整数，编码为 UTF-8。
  - **统计分析**：确保输入文件夹包含所有必要 rank 的 `.txt` 文件，格式与文本模式一致。

- **参数一致性**：
  - `num_positions_of_routed_experts` 必须能被 `num_ranks_of_collecting_data` 整除（默认 256 / 32 = 8）。
  - `num_eps_target_pattern` 必须能被 `num_ranks_target_pattern` 整除（默认 256 / 256 = 1）。
  - `layer_id` 必须有效，`recordstep_range` 需与输入文件中的 recordstep 匹配。
  - **版本相关**：在 ≤0.2.0 版本中，`num_ranks_of_collecting_data` 支持任意正整数；在 ≥0.3.0 版本中，该参数固定为 1。

- **中文支持**：
  - 可视化图像包含中文标题和标签，需安装 SimHei 或 Microsoft YaHei 字体。
  - 未安装字体可能导致中文显示为方框，但不影响数据处理。

- **错误排查**：
  - 检查日志文件（`pattern_generation_pipeline_<timestamp>.log` 或 `layer_stats_pipeline_<timestamp>.log`）以定位问题。
  - 常见错误包括文件缺失、格式不符、编码错误或参数不匹配。

- **性能优化**：
  - 对于大规模数据，生成可视化图像可能较慢，可通过设置 `fig_save_path` 保存图像而非显示。
  - 冗余模式计算复杂度较高，建议根据需求调整 `expert_redundant_limit`（默认 199）。
  - 统计分析处理大量 recordstep 文件时，建议优化输入文件夹结构以减少文件扫描时间。