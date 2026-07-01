# SKILL.md

## Model Validation: deepseek-ai/deepseek-vl2-tiny on vllm-ascend

### AI Assistant Used

Codex (GPT-5-based coding agent)

### Task Summary

Validate whether the deepseek-ai/deepseek-vl2-tiny model can be loaded on vLLM Ascend.

### Lessons Learned

1. Multimodal models may use non-standard config formats.
2. vLLM model config parsing is strict.
3. Always verify config.json structure first.
4. Version compatibility matters for Ascend NPU.
5. Document expected failures.

### SKILL.md Format Reference

https://agentskills.io/specification
