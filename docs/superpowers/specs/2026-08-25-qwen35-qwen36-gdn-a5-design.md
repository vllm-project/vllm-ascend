# Qwen3.5/Qwen3.6 GDN A5 operator integration design

## Historical status

This 2026-08-25 document originally described the first Stage 1 design:

- Ascend A5 / Ascend 950 only;
- six separately orchestrated GDN prefill operators;
- native-only recurrent decode;
- eager ordinary prefill and decode as the initial delivery boundary.

That design no longer represents the implemented architecture. The integration
was subsequently generalized to A2, A3, and A5, and the preferred prefill path
was changed to the Phase 6 fused GDN core. Recurrent decode can also use the FLA
public operator.

The current authoritative design is:

[`2026-08-29-qwen35-qwen36-gdn-fla-design.md`](2026-08-29-qwen35-qwen36-gdn-fla-design.md)

This file is retained only as a stable historical link. Do not use it for
implementation, validation, backend behavior, or release decisions.
