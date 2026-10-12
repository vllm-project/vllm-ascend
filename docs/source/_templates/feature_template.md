# xx Feature

## Feature Introduction (Required)

*A one-sentence description of the feature's functional definition (what it is, what it does), the specific benefits after using this feature (quantified or qualitative benefits such as performance improvement / fault tolerance enhancement / O&M convenience, etc.), and the version in which this feature was introduced or becomes available.*

> **Writing Requirements**:
>
> - (Required) A one-sentence description of the feature's functional definition: what it is, what it does.
> - (Required) A one-sentence description of the specific benefits after using this feature (quantified benefits such as performance improvement, fault tolerance enhancement, and O&M convenience, e.g., "inference throughput increased by N%", "startup time reduced by N%", "fault recovery time reduced").
> - (Required) A one-sentence description of the feature's version capability (e.g., "supported since vLLM-Ascend x.x.x", "supported since MindIE x.x.x", "requires CANN x.x.x or later").
> - (Optional) If it is necessary to specifically highlight the core function points and key advantages that distinguish this feature from other features, they can be listed here. Present them in a table, including: function/advantage name, brief description.

**Example**:

The xxx feature is supported since vLLM-Ascend x.x.x / MindIE x.x.x. Through the xxx mechanism, it achieves xxx in xxx scenarios. It can improve xxx (e.g., inference throughput increased by N%, startup time reduced by N%, fault recovery time reduced, etc.).

| Core Function | Description |
|----------|------|
| Multi-medium awareness | Supports NPU HBM / CPU / Disk three-tier cache hit query, with weighted scoring by medium weight |
| Dynamic probe interval | Dynamically adjusts the probe period based on NPU AI Cube utilization (20s when busy, 5s when idle) |

### Working Principle (Optional)

*Feature principle: For relatively complex features involving interaction between components, use flowcharts whenever possible to illustrate the process, and provide explanations for each process below the flowchart.*

**Note: Required when the feature principle is relatively complex or involves multiple components; for simple features, this section may be omitted after explaining it in one sentence in "Feature Introduction".**

> **Writing Requirements**:
>
> - Use flowcharts (Mermaid or ASCII diagrams) whenever possible to illustrate the end-to-end workflow.
> - Provide numbered explanations for each key step below the flowchart (e.g., "1. After xxx is completed → 2. xxx detects → 3. xxx executes").
> - When internal data structures or interfaces are involved, provide a brief field description table.

**Example**:

```text
┌──────────┐     step 1     ┌──────────┐     step 2     ┌──────────┐
│ Component│ ─────────────► │ Component│ ─────────────► │ Component│
│    A     │                │    B     │                │    C     │
└──────────┘                └──────────┘                └──────────┘
```

1. **xxx**：xxx.
2. **xxx**：xxx.
3. **xxx**：xxx.

*For example: automatic elastic scaling, virtual push health probing, KV Cache affinity scheduling capability deployment*

### Usage Scenarios (Required)

*Describe the business scenarios to which this feature applies, as well as the sub-modes/sub-strategies corresponding to each scenario.*

**Note: Required when the feature has multiple application scenarios; when there is only a single scenario, it is sufficient to explain it in "Feature Introduction", and this section may be omitted.**

> **Writing Requirements**:
>
> - If multiple scenarios exist, list them one by one and give the basis for distinguishing them (e.g., different hardware, different deployment modes, different load characteristics). Explain "under what business scenario it is used" (when to use it), and then explain which **sub-mode/sub-strategy** is adopted in that scenario (how to configure it).
> - For each scenario, give the applicable conditions and the boundaries where it is not applicable.
> - The "Usage Scenarios" section can be written either in Feature Introduction or in Feature Usage; it is sufficient as long as this section exists. This template introduces it by placing it in Feature Introduction as an example.

**Example**:

This feature applies to the following scenarios:

| Scenario | Sub-mode/Strategy | Applicable Conditions |
|------|-------------|----------|
| Scenario 1: xxx | Mode A | Applicable to xxx scenarios, used when xxx |
| Scenario 2: xxx | Mode B | Applicable to xxx scenarios, used when xxx |
| Scenario 3: xxx | Mode C | Applicable to xxx scenarios, used when xxx |

### Constraints and Limitations (Required)

> **Writing Requirements**:
>
> - **Hardware**: The hardware products supported by this feature (e.g., Atlas 800I A2 Inference Server, Atlas 800 A3 Super Node Server, Ascend 950 series, etc.). If not supported, state it explicitly.
> - **Deployment Scenario**: The deployment scenarios supported/not supported by this feature (e.g., only PD disaggregation supported, only PD co-location supported, only single-machine deployment supported, etc.).
> - **Engine**: Only applicable to MindIE products; not applicable to vLLM-Ascend / SGLang. The inference engines supported by this feature (e.g., only vLLM, only SGLang, both vLLM + SGLang supported). If behavior differs between engines, it must be explained.
> - **Model**: The model architectures supported/not supported by this feature (e.g., only Dense models supported, only MoE models supported, MLA supported, etc.). If there is a list of verified models, list them in a table.
> - **Feature Mutual Exclusion**: Whether this feature supports being used simultaneously with other features (e.g., does not support simultaneous use with xx feature). Unless explicitly stated otherwise, it can coexist with other features by default.
> - **Software Dependencies**: The software versions, tools, and libraries this feature depends on (e.g., requires CRIU 3.19, requires msprobe, requires HDK 26.0.RC1+, CANN version, etc.).
> - **Other Limitations**: Port conflicts, network requirements, permission requirements, etc.
> - **Solution Constraints**: If the feature has some solution or function constraints that need to be given with RFC links, please add a row at the end of the table to explain the RFC number and constraint content.

**Example**:

| Category | Description |
|------|------|
| Hardware | Atlas 800I A2 Inference Server, Atlas 800 A3 Super Node Server |
| Deployment Scenario | Supports PD disaggregated deployment, does not support PD co-location |
| Engine | Only supports vLLM (only applicable to MindIE products) |
| Model | Verified Qwen3-8B, Qwen3-30B-A3B, DeepSeek-V3.1-W8A8 |
| Feature Mutual Exclusion | Does not support simultaneous use with xx feature |
| Software Dependencies | CANN >= 8.5.0, HDK >= 25.5, Mooncake >= 0.3.11.post1 |
| Other Limitations | Requires privileged container; inter-node network requires 10Gbps+ |
| Solution Constraints | RFC-001: xxx function is limited, refer to `http://xxx/rfc/001` |

## Feature Usage

### Environment Preparation (Optional)

*Note: Required when this feature requires prerequisite preparation; simple features that can be enabled directly by configuration may omit this section.*

> **Writing Requirements**:
>
> - List the prerequisite steps that must be completed before using this feature (e.g., basic service deployment already completed, dependencies already installed, specific files already prepared, etc.).
> - Provide cross-reference links for the prerequisite steps (e.g., refer to [xxx Deployment](../quick_start_xxx.md) to complete basic service deployment).
> - Provide the source or installation command for software dependencies whenever possible.

**Example**:

- Basic inference service deployment has been completed using MindIE Motor, and the service is running normally. Refer to [xxx Service Deployment](../quick_start_motor.md).
- xxx has been installed, and the version meets the xxx requirement: `pip install xxx==x.x.x`.
- The xxx port of each node in the cluster network is reachable.

### Usage Example (Required)

*Based on actual operations, introduce how to enable the feature and provide clear step-by-step procedures.*

> **Writing Requirements**:
>
> - Provide **step-by-step** operation steps, with specific commands for each step, without skipping steps (including `cd` into directories).
> - Provide a command code block for each step, and explain the key parameters below the code block or through comments.
> - After deployment is complete, provide the **expected output** (e.g., log keywords, command output, HTTP response) to help users confirm that the operation was successful.
>
> **Multi-scenario Writing**: If multiple scenarios are listed in "Feature Introduction > Usage Scenarios", split this into subsections by scenario here, with each subsection corresponding one-to-one to the scenarios in "Usage Scenarios".

**Example 1: vLLM-Ascend (single scenario, including Online/Offline)**:

**Scenario 1: xxx Deployment**

*Scenario description: Corresponds to Scenario 1 in "Usage Scenarios", applicable to xxx conditions.*

**Online Inference（Server Mode）**：

```bash
vllm serve Qwen/Qwen3-8B \
  --additional-config '{"xxx_config": {"enabled": true}}'
```

Key parameter description:

| Parameter | Description |
|------|------|
| `xxx_config.enabled` | Enables this feature, default false |

**Offline Inference（Python API）**：

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-8B",
    additional_config={
        "xxx_config": {
            "enabled": True,
        },
    },
)
outputs = llm.generate("Hello, how are you?")
```

**Scenario 2: xxx Deployment**

*Scenario description: Corresponds to Scenario 2 in "Usage Scenarios", applicable to xxx conditions.*

> The writing method is the same as Scenario 1; here only the configuration differences from Scenario 1 are shown.

**Example 2: MindIE (cluster deployment scenario)**:

**Scenario 1: xxx Deployment**

1. Modify the configuration file:

   ```bash
   cd examples/deployer
   vim ../infer_engines/vllm/user_config.json
   ```

   Add the following configuration in `xxx_config`:

   ```json
   {
     "enable_xxx": true
   }
   ```

2. Deploy the service:

   ```bash
   # Method 1: Specify the configuration directory (recommended)
   python deploy.py --config_dir ../infer_engines/vllm

   # Method 2: Specify the configuration files separately
   python deploy.py --user_config_path ../infer_engines/vllm/user_config.json --env_config_path ../infer_engines/vllm/env.json
   ```

3. Confirm the deployment result:

   ```bash
   kubectl get pod -A -owide
   ```

   Expected output: The xxx instance and xxx component both start successfully, and `xxx registered` can be seen in the Coordinator logs.

**Scenario 2: xxx Deployment**

> If there are different hardware or deployment modes, write them here by scenario. The format is the same as above.

### Verifying the Feature (Required)

*Introduce how to verify that the feature has taken effect, and provide clear steps.*

*Note that every operation step should be presented, avoiding situations where users are assumed to already know certain steps and they are not provided, for example: the command to enter a directory.*

> **Writing Requirements**
>
> - Provide clear verification commands (e.g., `curl`, `kubectl`, log keyword checks, API requests).
> - Give the expected output or success indicators (e.g., HTTP 200, a specific keyword appearing in logs, a response containing a specific field).
> - Present every operation step, avoiding situations where users are assumed to already know certain steps.
> - If multiple scenarios exist and the verification methods differ, split into subsections by scenario, corresponding one-to-one to the scenarios in "Usage Example".

**Example**:

After the service starts, you can verify whether the feature has taken effect in the following ways:

1. Check the logs:

   ```bash
   kubectl logs -n mindie <coordinator-pod> | grep xxx
   ```

   Expected output:

   ```text
   xxx enabled, interval=...
   ```

2. Send an inference request for verification:

   ```bash
   curl http://<ip>:<port>/v1/completions \
       -H "Content-Type: application/json" \
       -d '{
           "model": "xxx",
           "prompt": "The future of AI is",
           "max_tokens": 50,
           "temperature": 0
       }'
   ```

   Expected result: Returns HTTP 200, and the response body contains the `choices` field.

## Configuration Parameters (Required)

*Use a table to list the full set of configurable parameters for the feature.*

> **Writing Requirements**:
>
> - The table includes: parameter name, type, default value, whether required, value range, description.
> - Parameter names must be consistent with those in the code/configuration files.
> - If parameters have dependency relationships (e.g., B takes effect only after A is enabled), this must be noted.
> - If parameter naming or configuration methods differ across products (MindIE / vLLM-Ascend / SGLang), list them by product.

**Example**:

| Parameter | Type | Default Value | Required | Value Range | Description |
|------|------|--------|------|----------|------|
| `enabled` | bool | false | Yes | true/false | Enables this feature |
| `interval` | int | 10 | No | 1-3600 | Probe interval (seconds) |
| `policy` | str | "default" | No | "default"/"aggressive" | Scheduling policy |

## Tuning Suggestions (Optional)

*Note: Required when the feature involves performance parameter tuning; features with no tuning space may omit this section.*

> **Writing Requirements**:
>
> - Provide recommended configurations for different business goals (e.g., throughput first, latency first, long context, resource constrained).
> - Present them in tables or lists, clearly stating the recommended values of key parameters for each scenario.
> - Note the mutual influence between parameters and precautions (e.g., increasing A increases memory usage, so B must be reduced at the same time).
> - If performance data exists, the test environment hardware model must be indicated (e.g., A2/A3/950DT&PR, etc.).

**Example**:

| Scenario | Key Parameter | Recommended Value | Description |
|------|----------|----------|------|
| Throughput first | `param_a` | A relatively large value (e.g., 64) | Increases xxx concurrency, but increases memory usage |
| Latency first | `param_b` | A relatively small value (e.g., 4-8) | Reduces scheduling latency |
| Long context | `param_c` | Maximum value | Must be adjusted together with xxx |

> **Note**: The above configurations are verified based on a specific test environment and are for reference only. The actual optimal configuration depends on factors such as xxx, and it is recommended to test and tune based on actual scenarios.

## FAQs (Optional)

**For general installation, environment, and deployment issues, please refer to the [Public FAQ](../faq.md). The following only includes troubleshooting issues specific to this feature.**

> **Writing Requirements**:
>
> - List common errors and abnormal phenomena during use of this feature.
> - Each FAQ contains three elements: **Problem Description**, **Cause Analysis**, **Solution Steps**.
> - For general environment/installation issues, reference the public FAQ link and do not repeat them here.

**Example**:

### Issue 1: xxx error / xxx does not take effect

**Problem Description**: After xxx starts, the log reports `xxx error`, and the feature does not take effect.

**Cause Analysis**: The value of the xxx configuration item is inconsistent with xxx / the xxx dependency is not installed / the xxx port is occupied.

**Solution Steps**:

1. Check whether `xxx_config.xxx` is configured correctly: `xxx`.
2. Confirm that the xxx dependency has been installed: `xxx --version`.
3. Check port usage: `netstat -tlnp | grep xxx`.

### Issue 2: xxx

**Problem Description**: xxx.

**Cause Analysis**: xxx.

**Solution Steps**: xxx.
