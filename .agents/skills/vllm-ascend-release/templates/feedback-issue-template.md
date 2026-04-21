## vLLM Ascend ${VERSION} Release Feedback

We are preparing to release **vLLM Ascend ${VERSION}**. This issue is for collecting community feedback before the official release.

### What's New

<!-- Release highlights will be added here -->

### How to Test

You can test the release candidate by:

1. **Using Docker (Recommended)**:
   ```bash
   docker pull quay.io/ascend/vllm-ascend:${VERSION}
   ```

2. **Using pip**:
   ```bash
   pip install vllm-ascend==${VERSION}
   ```

3. **Building from source**:
   ```bash
   git clone https://github.com/vllm-project/vllm-ascend.git
   cd vllm-ascend
   git checkout ${VERSION}
   pip install -e .
   ```

### Feedback Guidelines

Please share your feedback in the comments below:

- **Bug Reports**: Describe the issue, steps to reproduce, and your environment
- **Performance Issues**: Include benchmark results and comparison with previous versions
- **Feature Requests**: Describe the use case and expected behavior
- **Documentation Issues**: Point out unclear or missing documentation

### Environment Information

When reporting issues, please include:

```
- vLLM Ascend version:
- vLLM version:
- CANN version:
- torch_npu version:
- Hardware (910B/310P/etc.):
- OS:
```

### Timeline

- **RC Release Date**: ${DATE}
- **Feedback Period**: 1 week
- **Stable Release**: TBD based on feedback

### Related Links

- [Release Checklist](${CHECKLIST_ISSUE_URL})
- [Documentation](https://docs.vllm.ai/projects/ascend/en/latest)
- [Previous Releases](https://github.com/vllm-project/vllm-ascend/releases)

---

Thank you for helping us improve vLLM Ascend!
