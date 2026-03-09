# InternVL3-8B Support Design

**Date:** 2026-03-09
**Issue:** #1362
**Author:** AI Assistant
**Status:** Approved

## Overview

This design document outlines the approach for adding InternVL3-8B support to vLLM-Ascend. The implementation will leverage the existing InternVL2.5 codebase and make minimal adaptations to support the InternVL3 architecture.

## Background

### Current State
- vLLM-Ascend currently supports InternVL2.5
- InternVL3-8B and InternVL3-78B are listed in model_list.json but not fully supported
- Documentation marks InternVL2.0/2.5/3.0 as unsupported (issue #2064)

### Requirements
- Add support for InternVL3-8B model from https://huggingface.co/OpenGVLab/InternVL3-8B
- Maintain compatibility with existing InternVL2.5 implementation
- Ensure proper integration with Ascend hardware

## Architecture Design

### Approach: Minimal Adaptation from InternVL2.5

We will use **Approach 1** - minimal adaptation from InternVL2.5, which provides:
- Fastest implementation path
- Lower risk by building on proven code
- Easier maintenance and debugging
- Leverages InternVL3's architectural similarity to 2.5

### Key Components

#### 1. Model Configuration and Registration
- Update model registry to recognize InternVL3-8B model identifier
- Add version detection logic to differentiate between InternVL2.5 and InternVL3
- Configure model-specific parameters (if different from 2.5)

#### 2. Model Implementation
- Check upstream vLLM for InternVL3 implementation
- Adapt or extend existing InternVL2.5 code for version 3.0
- Ensure compatibility with Ascend-specific operations

#### 3. Testing Infrastructure
- Create test configuration for InternVL3-8B
- Add end-to-end test cases
- Validate multimodal inference (image + text)

#### 4. Documentation
- Update supported_models.md to mark InternVL3 as supported
- Add usage examples and tutorials
- Document any version-specific differences

## Data Flow

### Input Processing
1. User specifies model: `OpenGVLab/InternVL3-8B`
2. vLLM identifies model type as InternVL series
3. Load model config and detect version (2.5 vs 3.0)
4. Select appropriate processing logic based on version

### Inference Pipeline
1. **Image Input** → Vision Encoder (InternViT)
2. **Text Input** → Tokenizer
3. **Multimodal Fusion** → Language Model (Qwen2-based)
4. **Output Generation**

## Error Handling

### Scenarios to Handle
- Model files not found or corrupted
- Configuration parameter incompatibility
- Ascend hardware resource constraints
- Version detection failures

### Strategy
- Provide clear error messages with actionable guidance
- Fallback to InternVL2.5 configuration when version uncertain
- Log detailed information for debugging
- Validate hardware compatibility before loading

## Testing Strategy

### Test Levels
1. **Unit Tests**
   - Model configuration loading
   - Version detection logic
   - Parameter validation

2. **Integration Tests**
   - End-to-end inference with sample inputs
   - Multimodal processing (image + text)
   - Various input sequence lengths

3. **Performance Tests**
   - Benchmark on Ascend hardware
   - Compare with InternVL2.5 performance
   - Memory usage profiling

### Test Data
- Official InternVL3-8B examples from HuggingFace
- Multimodal inputs (images with text prompts)
- Edge cases (long sequences, multiple images)

## Implementation Plan

### Phase 1: Core Support
1. Research upstream vLLM InternVL3 implementation
2. Update model registry and configuration
3. Add version detection logic
4. Create basic test configuration

### Phase 2: Testing and Validation
1. Implement unit tests
2. Create end-to-end test cases
3. Validate on Ascend hardware
4. Fix any compatibility issues

### Phase 3: Documentation and Polish
1. Update supported models documentation
2. Add usage examples
3. Create tutorial (if needed)
4. Final testing and validation

## Success Criteria

- [ ] InternVL3-8B model loads successfully
- [ ] Multimodal inference works correctly (image + text)
- [ ] Tests pass on Ascend hardware
- [ ] Documentation updated
- [ ] Performance comparable to InternVL2.5

## Risks and Mitigations

### Risk 1: Architectural Differences
**Mitigation:** Thoroughly review InternVL3 model architecture and compare with 2.5

### Risk 2: Ascend Hardware Compatibility
**Mitigation:** Test early on target hardware, implement fallbacks

### Risk 3: Upstream Dependencies
**Mitigation:** Pin specific versions, maintain compatibility layer

## Future Enhancements

- Support for InternVL3-78B (larger model)
- Ascend-specific optimizations (W8A8 quantization)
- Advanced features (tensor parallelism, pipeline parallelism)
- Performance tuning and optimization

## References

- Issue: https://github.com/vllm-project/vllm-ascend/issues/1362
- Model: https://huggingface.co/OpenGVLab/InternVL3-8B
- InternVL3 Architecture: https://huggingface.co/OpenGVLab/InternVL3-38B
