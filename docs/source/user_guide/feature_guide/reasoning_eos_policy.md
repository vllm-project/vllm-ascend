# Reasoning EOS Policy

Some reasoning models can emit their model EOS token before closing the
reasoning block. The optional phase-aware EOS policy masks model EOS tokens
while the generated token stream is inside reasoning, then restores them after
the parser's reasoning-end or tool-call transition.

Enable the policy together with a reasoning parser:

```bash
vllm serve <model> \
  --reasoning-parser <parser> \
  --reasoning-config '{"premature_eos_policy":"mask_in_reasoning"}'
```

The default policy is `allow`, which preserves vLLM's existing behavior. With
`mask_in_reasoning`, masking starts only after the complete reasoning-start
marker has been generated. Multi-token start and exit markers are supported.
Only EOS IDs supplied by the model generation configuration are masked; user
`stop_token_ids` remain effective.

The policy applies to the v1 Ascend model runner for normal sampling and
speculative decoding, including the MTP target-token and bonus-token paths. If
masking every model EOS candidate would leave a row without a finite token, the
processor fails open and restores the original EOS logits.

When requesting log probabilities, `raw_logits` and `raw_logprobs` report
values before this processor, while processed modes include the EOS mask.
