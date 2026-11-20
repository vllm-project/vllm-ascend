# Anthropic Messages API minimum output length

When serving models through `/v1/messages`, pass `min_tokens` to prevent EOS or
stop sequences from ending generation before the requested number of output
tokens. It defaults to `0` and must satisfy `0 <= min_tokens <= max_tokens`.
This is a vLLM extension to the Anthropic API. With the Anthropic Python SDK,
pass it through `extra_body`:

```python
message = client.messages.create(
    model="your-served-model",
    max_tokens=64,
    messages=[{"role": "user", "content": "Write two short lines of poetry."}],
    extra_body={"min_tokens": 16},
)
```

vLLM Ascend backports this support for older vLLM releases. Versions containing
[vLLM #65459](https://github.com/vllm-project/vllm/pull/65459) use the native implementation.
