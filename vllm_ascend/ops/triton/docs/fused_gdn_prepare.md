# Fused GDN projection preparation

`fused_gdn_prepare` prepares the non-interleaved Qwen GDN projections in one
Triton kernel. It materializes the packed QKVZ prefix as dense QKV, splits the
packed BA projection into dense B and A tensors, and initializes the complete
GDN output buffer to zero. Z remains a view over the QKVZ projection.

Keeping the output initialization in the fused kernel preserves deterministic
padding rows during graph replay: the GDN core writes only
`num_actual_tokens`, while the following normalization consumes the full graph
buffer.

The operator is a bit-exact data-movement operation. Tests cover empty, decode,
and larger token counts with FP16 and BF16 inputs.
