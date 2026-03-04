---
name: main2main
description: "The main2main skill guides an AI agent to adapt the latest vLLM main branch code for vLLM Ascend project."
---

# main2main Skill

This skill guides AI agents to adapt the latest vLLM main branch code for the vLLM Ascend project.

## Workflow

### 1. Get Current vLLM Version Information for vLLM Ascend

Find the vLLM version information for the **main branch** in `docs/source/community/versioning_policy.md` under the `Release compatibility matrix` section:

- **Current adapted vLLM commit**: Format like `83b47f67b1dfad505606070ae4d9f83e50ad4ebd, v0.15.0 tag`
- **Compatible vLLM version**: From the table, e.g., `v0.15.0`

### 2. Get the Latest vLLM Code

Retrieve the latest commit from the local vLLM git repository:

```bash
# The vLLM git repository is typically located in the parent directory
cd ../vllm
git log -1 --format="%H %s"
```

If the vLLM repository is not found at the default location, prompt the user to specify the exact path to the vLLM git repository.

### 3. Compare vLLM Changes

Compare the differences between the vLLM commit currently adapted by vLLM Ascend and the latest commit:

```bash
# View file changes between two commits
git diff <old_commit> <new_commit> --name-only

# View detailed code changes
git log --oneline <old_commit>..<new_commit>
```

### 4. Analyze vLLM Changes and Generate Change Report

Create a file named `vllm_changes.md` to save the list of changes in vLLM that are relevant to vLLM Ascend. This file will be used to guide the adaptation process and should be removed after all work is done.

#### 4.1 Identify Key vLLM Source Files

Focus on vLLM source files under `vllm/vllm/` directory, especially:

```bash
# Get changed files in vLLM source code
git diff <old_commit> <new_commit> --name-only | grep -E "^vllm/" | head -200

# Count total changes
git diff <old_commit> <new_commit> --name-only | wc -l
```

#### 4.2 Categorize Changes by Priority

When analyzing changes, categorize them into the following priority levels:

| Priority | Category | Description |
|----------|----------|-------------|
| **P0** | Breaking Changes | API changes that will cause runtime errors if not adapted |
| **P1** | Important Changes | Changes that affect functionality or performance |
| **P2** | Moderate Changes | Changes that may need review for compatibility |
| **P3** | Model Changes | New models or model updates |
| **P4** | Minor Changes | Configuration, documentation, or minor refactoring |

#### 4.3 Key Areas to Focus On

When analyzing vLLM changes, pay special attention to these areas that typically require vLLM Ascend adaptation:

1. **Platform Interface** (`vllm/platforms/`)
   - New abstract methods that must be implemented
   - Method signature changes
   - New platform features

2. **MoE (Mixture of Experts)** (`vllm/model_executor/layers/fused_moe/`)
   - FusedMoE layer changes
   - Activation function changes
   - Router changes

3. **Attention** (`vllm/model_executor/layers/attention/`)
   - Attention backend changes
   - New parameters or interfaces
   - MLA (Multi-Head Latent Attention) updates

4. **Speculative Decoding** (`vllm/v1/worker/gpu/spec_decode/`, `vllm/config/speculative.py`)
   - Import path changes
   - Config field changes
   - New speculative methods

5. **Distributed** (`vllm/distributed/`)
   - Parallel state changes
   - KV transfer changes
   - Device communicator updates

6. **Models** (`vllm/model_executor/models/`)
   - New model architectures
   - Model interface changes

7. **Worker/Model Runner** (`vllm/v1/worker/gpu/model_runner.py`)
   - New worker methods
   - Model runner changes

8. **Quantization** (`vllm/model_executor/layers/quantization/`)
   - Quantization config changes
   - compress-tensor method changes

#### 4.4 vllm_changes.md Template

Use the following template structure for `vllm_changes.md`:

```markdown
# vLLM Changes Relevant to vLLM Ascend
# Generated: <DATE>
# Old commit: <OLD_COMMIT_HASH> (<OLD_VERSION>)
# New commit: <NEW_COMMIT_HASH>
# Total commits: <COUNT>

================================================================================
## P0 - Breaking Changes (Must Adapt)
================================================================================

### <INDEX>. <CHANGE_TITLE>
FILE: <VLLM_FILE_PATH>
CHANGE: <DESCRIPTION_OF_CHANGE>
IMPACT: <WHAT_BREAKS_IF_NOT_ADAPTED>
VLLM_ASCEND_FILES:
  - <PATH_TO_ASCEND_FILE_1>
  - <PATH_TO_ASCEND_FILE_2>

================================================================================
## P1 - Important Changes (Should Adapt)
================================================================================
...

================================================================================
## P2 - Moderate Changes (Review Needed)
================================================================================
...

================================================================================
## P3 - Model Changes
================================================================================
...

================================================================================
## P4 - Configuration/Minor Changes
================================================================================
...

================================================================================
## Files/Directories Renamed
================================================================================
<LIST_OF_RENAMED_FILES>

================================================================================
## END OF CHANGES
================================================================================
```

#### 4.5 Commands to Analyze Specific Changes

```bash
# Check for breaking changes in commit messages
git log --oneline <old_commit>..<new_commit> | grep -iE "(refactor|breaking|api|rename|remove|deprecate)"

# View specific file changes
git diff <old_commit> <new_commit> -- <FILE_PATH>

# Check for renamed/moved files
git diff <old_commit> <new_commit> --name-status | grep -E "^R"

# Check platform interface changes
git diff <old_commit> <new_commit> -- vllm/platforms/

# Check MoE changes
git diff <old_commit> <new_commit> -- vllm/model_executor/layers/fused_moe/

# Check attention changes
git diff <old_commit> <new_commit> -- vllm/model_executor/layers/attention/

# Check speculative decoding changes
git diff <old_commit> <new_commit> -- vllm/v1/worker/gpu/spec_decode/ vllm/config/speculative.py
```

### 5. Adapt vLLM Ascend Project

For each related change in vLLM from the file `vllm_changes.md`, evaluate whether adaptation in vLLM Ascend is needed:

#### 5.1 Internal Architecture Changes

- Check internal interfaces of vLLM core modules (scheduler, executor, model runner, etc.)
- Update vLLM Ascend's Ascend-specific implementations (e.g., NPU worker/model runner, custom attention、custom ops)
- Preserve vLLM Ascend specific modifications (e.g., code under `vllm_ascend/`)

#### 5.2 Dependency Changes

- Check for dependency version changes in `pyproject.toml` or `setup.py`
- Update dependency declarations in vLLM Ascend

### 5. Test and Verify

- Run vLLM Ascend's CI/CD pipeline
- Verify core functionality (text generation, batching, NPU memory management)
- Ensure backward compatibility: test compatibility with older vLLM versions

## Key File Locations

| Project | Path |
|---------|------|
| vLLM Ascend version compatibility | `docs/source/community/versioning_policy.md` |
| vLLM Ascend source code | `vllm_ascend/` |
| **Core Modules** | |
| Ascend-specific attention | `vllm_ascend/attention/` |
| Ascend-specific executor | `vllm_ascend/worker/` |
| Ascend-specific ops | `vllm_ascend/ops/` |
| **Specialized Implementations** | |
| Ascend 310P specific | `vllm_ascend/_310p/` |
| EPLB load balancing | `vllm_ascend/eplb/` |
| XLite compiler | `vllm_ascend/xlite/` |
| **Compilation & Fusion** | |
| Graph fusion pass manager | `vllm_ascend/compilation/` |
| Compilation passes | `vllm_ascend/compilation/passes/` |
| **Quantization** | |
| Quantization methods | `vllm_ascend/quantization/` |
| ModelSlim integration | `vllm_ascend/quantization/methods/modelslim/` |
| **Distributed & KV Cache** | |
| KV transfer | `vllm_ascend/distributed/kv_transfer/` |
| Device communicators | `vllm_ascend/distributed/device_communicators/` |
| **Speculative Decoding** | |
| MTP proposer | `vllm_ascend/spec_decode/mtp_proposer.py` |
| Eagle proposer | `vllm_ascend/spec_decode/eagle_proposer.py` |
| **Utility Modules** | |
| Common utilities | `vllm_ascend/utils.py` |
| Ascend config | `vllm_ascend/ascend_config.py` |
| Platform detection | `vllm_ascend/platform.py` |
| Environment variables | `vllm_ascend/envs.py` |

## Important Notes

1. **Version Checking**: vLLM Ascend uses version checking to maintain compatibility with multiple vLLM versions. Preserve or update related logic when adapting.

2. **Test Verification**: After adaptation, tests must verify:
    - Compatibility with the latest vLLM version
    - Backward compatibility with older vLLM versions
    - Ascend NPU functionality works correctly

3. **Documentation Sync**: If vLLM documentation has significant changes, update vLLM Ascend's documentation accordingly.

4. **Backward Compatibility**:
    - Maintain compatibility from the version currently adapted by vLLM Ascend to the latest version
    - Use version checking to handle code branches for different versions:
    ```python
    from vllm_ascend.utils import vllm_version_is

    if vllm_version_is("0.15.0"):
        # Use API for v0.15.0
    else:
        # Use API for other versions
    ```

5. Do not forget to update the vLLM version is `.github` for CI files.

6. **Change Logging**: After adaptation, clearly document in the commit message:
   - The range of adapted vLLM commits
   - Main changes made
   - Test results

7. the vLLM python code is under `vllm/vllm` folder.

## Reference

- [Versioning Policy](../../../docs/source/community/versioning_policy.md) - vLLM Ascend versioning strategy
