# MSProbe Debugging Guide

During inference or training runs we often encounter accuracy anomalies such as outputs drifting away from the expectation, unstable numerical behavior (NaN/Inf), or predictions that no longer match the labels. To pinpoint the root cause we have to monitor and capture intermediate data produced while the model executes—feature maps, weights, activations, and layer outputs. By capturing key tensors at specific stages, logging I/O pairs for the core layers, and retaining contextual metadata (prompts, tensor dtypes, hardware configuration, etc.), we can systematically trace where the accuracy degradation or numerical error started. This guide describes the end-to-end workflow for diagnosing accuracy issues for AI models (with a focus on vllm-ascend services): preparation, data capture, and analysis & verification.

For more details, see [Ascend/msprobe](https://gitcode.com/Ascend/msprobe).

## 0. Background Concepts

`msprobe` supports three accuracy levels:

- **L0**: dumps tensors at the module level and generates `construct.json` so that visualization tools can rebuild the network structure. A model or submodule handle must be passed in.
- **L1**: collects operator-level statistics only, which is suitable for lightweight troubleshooting.
- **mix**: captures both structural information and operator statistics, which is useful when you need both graph reconstruction and numerical comparisons.

## 1. Prerequisites

### 1.1 Install `msprobe`

Install msprobe with pip:

```bash
pip install mindstudio-probe
```

### 1.2 Graph mode dump (optional)

If you need to dump cudagraph graphs, you need to install from source code:

1. Install `aclgraph_dump` from source code:

   ```bash
   git clone https://gitcode.com/Ascend/msprobe.git
   cd msprobe
   python3 setup.py bdist_wheel --include-mod=aclgraph_dump --no-check
   pip install dist/*.whl
   ```

## 2. Collecting Data with `msprobe`

We generally follow a coarse-to-fine strategy when capturing data. First, identify the token where the issue shows up, and then decide which range needs to be sampled around that token. The typical workflow is described below.

### 2.1 Prepare the dump configuration file

Create a `config.json` that can be parsed by `PrecisionDebugger` and place it in an accessible path. Common fields are:

| Field       | Description                                                                                                                                                                                                 | Required | Eager Mode | Graph Mode |
|:-----------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------:|:-------------:|:-------------:|
| `task`      | Type of dump task. Common PyTorch values include `"statistics"` and `"tensor"`. A statistics task collects tensor statistics (mean, variance, max, min, etc.) while a tensor task captures arbitrary tensors. |    Yes   |      ✅       |      ✅       |
| `dump_path` | Directory where dump results are stored. When omitted, `msprobe` uses its default path.                                                                                                                     |    No    |      ✅       |      ✅       |
| `rank`      | Ranks to sample. An empty list collects every rank. For single-card tasks, you must set this field to `[]`.                                                                                                 |    No    |      ✅       |      ✅       |
| `step`      | Token iteration(s) to sample. An empty list means every iteration.                                                                                                                                         |    No    |      ✅       |      ❌       |
| `level`     | Dump level string (`"L0"`, `"L1"`, or `"mix"`). `L0` targets `nn.Module`, `L1` targets `torch.api`, and `mix` collects both.                                                                                 |    Yes   |      ✅       |      ✅       |
| `async_dump`| Whether to enable asynchronous dump (supported for PyTorch `statistics`/`tensor` tasks). Defaults to `false`.                                                                                              |    No    |      ✅       |      ❌       |
| `scope`     | Module range to sample. An empty list collects every module.                                                                                                                                                |    No    |      ✅       |      ❌       |
| `dump_enable` | Dynamic switch for enabling/disabling dump in `PrecisionDebugger` during one running training/inference job. This allows turning dump on or off on demand in the same job.                             |    No    |      ✅       |      ❌       |
| `list`      | Operator range to sample. An empty list collects every operator.                                                                                                                                            |    No    |      ✅       |      ✅       |

To restrict the operators that are captured, configure the `list` block:

- `scope` (list[str]): In PyTorch PyNative scenarios this field restricts the dump range. Provide two module or API names that follow the tool's naming convention to lock a range; only data between the two names will be dumped. Examples:

  ```json
  "scope": ["Module.conv1.Conv2d.forward.0", "Module.fc2.Linear.forward.0"]
  "scope": ["Cell.conv1.Conv2d.forward.0", "Cell.fc2.Dense.forward.0"]
  "scope": ["Tensor.add.0.forward", "Functional.square.2.forward"]
  ```

  The `level` setting determines what can be provided—modules when `level=L0`, APIs when `level=L1`, and either modules or APIs when `level=mix`.

- `list` (list[str]): Custom operator list. Options include:
    - Supply the full names of specific APIs in PyTorch pynative scenarios to only dump those APIs. Example: `"list": ["Tensor.permute.1.forward", "Tensor.transpose.2.forward", "Torch.relu.3.forward"]`.
    - When `level=mix`, you can provide module names so that the dump expands to everything produced while the module is running. Example: `"list": ["Module.module.language_model.encoder.layers.0.mlp.ParallelMlp.forward.0"]`.
    - Provide a substring such as `"list": ["relu"]` to dump every API whose name contains the substring. When `level=mix`, modules whose names contain the substring are also expanded.

Example configuration:
eager mode: 
```json
{
  "task": "statistics",
  "dump_path": "/home/data_dump",
  "rank": [],
  "step": [],
  "level": "L1",
  "async_dump": false,

  "statistics": {
    "scope": [],
    "list": [],
    "tensor_list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

Graph mode:
```json
{
  "task": "statistics",
  "level": "L1",
  "dump_path": "/home/data_dump",
  "statistics": {
    "list": []
  }
}
```

## 3. Enable `msprobe` in vllm-ascend

1. Start vLLM and pass the config path through `--additional-config`:

   ```bash
   vllm serve Qwen/Qwen2.5-0.5B-Instruct \
     --dtype bfloat16 \
     --host 0.0.0.0 \
     --port 8000 \
     --additional-config '{"dump_config_path": "/data/msprobe_config.json"}' &
   ```

## 4. Send requests and collect dumps

1. Send inference requests as usual, for example:

   ```bash
   curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "Qwen/Qwen2.5-0.5B-Instruct",
           "prompt": "Explain gravity in one sentence.",
           "max_completion_tokens": 32,
           "temperature": 0
         }' | python -m json.tool
   ```

2. Each request drives the sequence `msprobe: start -> forward -> stop -> step`. The runner invokes `step()` on every code path, so you always get a complete dataset even if inference returns early.

3. Dump files are written into `dump_path`. They usually contain:
   - Tensor files grouped by operator/module.
   - `dump.json`, which records metadata such as dtype, shape, min/max, and `requires_grad`.
   - `construct.json`, which is generated when `level` is `L0` or `mix` (required for visualization).

   Example directory layout:
   eager mode:

   ```text
   ├── dump_path
   │   ├── step0
   │   │   ├── rank0
   │   │   │   ├── dump_tensor_data
   │   │   │   │    ├── Tensor.permute.1.forward.pt                       # Format: {api_type}.{api_name}.{call_count}.forward.{input/output}.{arg_index}.
   │   │   │   │    │                                              # arg_index is the nth input or output of the API. If an input is a list, keep numbering with decimals (e.g., 1.1 is the first element of the first argument).
   │   │   │   │    ├── Module.conv1.Conv2d.forward.0.input.0.pt          # Format: {Module}.{module_name}.{class_name}.forward.{call_count}.{input/output}.{arg_index}.
   │   │   │   │    └── Module.conv1.Conv2d.forward.0.parameters.bias.pt  # Module parameter data: {Module}.{module_name}.{class_name}.forward.{call_count}.parameters.{parameter_name}.
   │   │   │   │                                                          # When the `model` argument passed to dump is a List[torch.nn.Module] or Tuple[torch.nn.Module], module-level data names also include the index inside the list ({Module}.{index}.*), e.g., Module.0.conv1.Conv2d.forward.0.input.0.pt.
   │   │   │   ├── dump.json
   │   │   │   ├── stack.json
   │   │   │   ├── dump_error_info.log
   │   │   │   └── construct.json
   │   │   ├── rank1
   │   │   │   ├── dump_tensor_data
   │   │   │   │   └── ...
   │   │   │   ├── dump.json
   │   │   │   ├── stack.json
   │   │   │   ├── dump_error_info.log
   │   │   │   └── construct.json
   │   │   ├── ...
   │   │   │
   │   │   └── rank7
   │   ├── step1
   │   │   ├── ...
   │   ├── step2
   ```

   - `rank`: Device ID. Each card writes its data to the corresponding `rank{ID}` directory. In non-distributed scenarios the directory is simply named `rank`.
   - `dump_tensor_data`: Tensor payloads that were collected.
   - `dump.json`: Statistics for the forward data of each API or module, including names, dtype, shape, max, min, mean, L2 norm (square root of the L2 variance), and CRC-32 when `summary_mode="md5"`. See [dump.json file description](#dumpjson-file-description) for details.
   - `dump_error_info.log`: Present only when the dump tool encountered an error and records the failure log.
   - `stack.json`: Call stacks for APIs/modules.
   - `construct.json`: Hierarchical structure description. Empty when `level=L1`.

   graph mode:

   ```text
   L0_dump
   ├── step0
   │   └── rank0
   │       └── dump.json
   ├── step1
   │   └── rank0
   │       └── dump.json
   ├── step2
   │   └── rank0
   │       └── dump.json
   ├── step3
   │   └── rank0
   │       └── dump.json
   ├── step4
   │   └── rank0
   │       └── dump.json
   └── step5
       └── rank0
           └── dump.json
   ```

   - `dump.json`: Statistics for the forward data of each API or module, including names, dtype, shape, max, min, mean, L2 norm (square root of the L2 variance), and CRC-32 when `summary_mode="md5"`. See [dump.json file description](#dumpjson-file-description) for details.


## 5. Analyze the results

### 5.1 Prerequisites

You typically need two dump datasets: one from the "problem side" (the run that exposes the accuracy or numerical error) and another from the "benchmark side" (a good baseline). These datasets do not have to be identical—they can come from different branches, framework versions, or even alternative implementations (operator substitutions, different graph-optimization switches, etc.). As long as they use the same or similar inputs, hardware topology, and sampling points (step/token), `msprobe` can compare them and locate the divergent nodes. If you cannot find a perfectly clean benchmark, start by capturing the problem-side data, craft the smallest reproducible case by hand, and perform a self-comparison. Below we assume the problem dump is `problem_dump` and the benchmark dump is `bench_dump`.

### 5.2 Visualization

Use `msprobe graph_visualize` to build or compare graphs, then open the generated `*.vis.db` file(s) with TensorBoard (`tb_graph_ascend` plugin).

1. Ensure dump data is visualization-ready:
   - Dump level must be `L0` or `mix` so `construct.json` is non-empty.
   - Each rank directory should contain `dump.json`, `stack.json`, and `construct.json`.

2. Choose command mode:
   - Single-graph build:

     ```bash
     msprobe graph_visualize -tp <target_path> -o <output_path>
     ```

   - Graph comparison:

     ```bash
     msprobe graph_visualize -tp <target_path> -gp <golden_path> -o <output_path>
     ```

   - Common optional flags:
     - `-oc` / `--overflow_check`: enable overflow marking
     - `-fm` / `--fuzzy_match`: enable fuzzy matching for node mapping
     - `-lm` / `--layer_mapping [mapping.yaml]`: cross-framework/layer mapping compare
     - `-tensor_log`: print per-node compare log (tensor dump scenarios)
     - `-progress_log`: print detailed progress log

3. Path granularity is auto-detected by `graph_visualize`:
   - Single-rank: `.../step0/rank0`
   - Multi-rank (batch): `.../step0`
   - Multi-step (batch): dump root path containing `step*`

4. Output files:
   - Single-graph build: `build_{timestamp}.vis.db`
   - Graph comparison: `compare_{timestamp}.vis.db`

5. Launch TensorBoard with the output directory:

   ```bash
   tensorboard --logdir <output_path> --bind_all --port <optional_port>
   ```

6. In the visualization UI, inspect structure and numeric differences:
   - Switch rank/step to locate unstable nodes quickly.
   - Use search/filter to focus on target ops/modules.
   - For compare mode, prioritize highlighted high-difference nodes and trace surrounding I/O/parameters.

## 6. Troubleshooting

- `RuntimeError: Please enforce eager mode`: Restart vLLM and add the `--enforce-eager` flag.
- No dump files: Confirm that the JSON path is correct and every node has write permission. In distributed scenarios set `keep_all_ranks` so that every rank writes its own dump.
- Dumps are too large: Start with a `statistics` task to locate abnormal tensors, then narrow the scope with `scope`/`list`/`tensor_list`, `filters`, `token_range`, etc.

---

## Appendix

### dump.json file description

#### L0 level

An L0 `dump.json` contains forward I/O for modules together with parameters. Using PyTorch's `Conv2d` as an example, the network code looks like:

`output = self.conv2(input)  # self.conv2 = torch.nn.Conv2d(64, 128, 5, padding=2, bias=True)`

`dump.json` contains the following entries:

- `Module.conv2.Conv2d.forward.0`: Forward data of the module. `input_args` represents positional inputs, `input_kwargs` represents keyword inputs, `output` stores forward outputs, and `parameters` stores weights/biases.

**Note**: When the `model` parameter passed to the dump API is `List[torch.nn.Module]` or `Tuple[torch.nn.Module]`, module-level names include the index inside the list (`{Module}.{index}.*`). Example: `Module.0.conv1.Conv2d.forward.0`.

```json
{
 "task": "tensor",
 "level": "L0",
 "framework": "pytorch",
 "dump_data_dir": "/dump/path",
 "data": {
  "Module.conv2.Conv2d.forward.0": {
   "input_args": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      8,
      16,
      14,
      14
     ],
     "Max": 1.638758659362793,
     "Min": 0.0,
     "Mean": 0.2544615864753723,
     "Norm": 70.50277709960938,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.input.0.pt"
    }
   ],
   "input_kwargs": {},
   "output": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      8,
      32,
      10,
      10
     ],
     "Max": 1.6815717220306396,
     "Min": -1.5120246410369873,
     "Mean": -0.025344856083393097,
     "Norm": 149.65576171875,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.output.0.pt"
    }
   ],
   "parameters": {
    "weight": {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      5,
      5
     ],
     "Max": 0.05992485210299492,
     "Min": -0.05999220535159111,
     "Mean": -0.0006165213999338448,
     "Norm": 3.421217441558838,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.parameters.weight.pt"
    },
    "bias": {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32
     ],
     "Max": 0.05744686722755432,
     "Min": -0.04894155263900757,
     "Mean": 0.006410328671336174,
     "Norm": 0.17263513803482056,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.parameters.bias.pt"
    }
   }
  }
 }
}
```

#### L1 level

An L1 `dump.json` records forward I/O for APIs. Using PyTorch's `relu` function as an example (`output = torch.nn.functional.relu(input)`), the file contains:

- `Functional.relu.0.forward`: Forward data of the API. `input_args` are positional inputs, `input_kwargs` are keyword inputs, and `output` stores the forward outputs.

```json
{
 "task": "tensor",
 "level": "L1",
 "framework": "pytorch",
 "dump_data_dir":"/dump/path",
 "data": {
  "Functional.relu.0.forward": {
   "input_args": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      28,
      28
     ],
     "Max": 1.3864083290100098,
     "Min": -1.3364859819412231,
     "Mean": 0.03711778670549393,
     "Norm": 236.20692443847656,
     "requires_grad": true,
     "data_name": "Functional.relu.0.forward.input.0.pt"
    }
   ],
   "input_kwargs": {},
   "output": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      28,
      28
     ],
     "Max": 1.3864083290100098,
     "Min": 0.0,
     "Mean": 0.16849493980407715,
     "Norm": 175.23345947265625,
     "requires_grad": true,
     "data_name": "Functional.relu.0.forward.output.0.pt"
    }
   ]
  }
 }
}  
```

#### mix level

A `mix` dump.json contains both L0 and L1 level data; the file format is the same as the examples above.
