# Inkling and MTP adaptation plan

Draft placeholder for [RFC #20812](https://github.com/vllm-project/vllm-ascend/issues/20812).
This change records the integration scope; runtime support is pending.

## Planned integration

1. Register the Inkling target and MTP draft through Ascend model/backend dispatch.
   Reuse upstream configuration, checkpoint mappings and frontend components.
2. Adapt relative attention, full/sliding-window attention, short convolution and
   their cache/metadata paths to NPU, alongside Ascend linear and MoE execution.
3. Preserve target embedding/LM-head sharing and the backbone-plus-depth embedding
   normalization chain. Handle per-depth draft blocks, optional chain normalization
   and muP logits scaling consistently with the target.
4. Integrate per-depth KV and convolution state with speculative rejection and
   request remapping. Establish eager correctness before ACL graph replay and
   qualify quantized weights, parallelism and multimodal execution separately.

## Upstream references

1. [Inkling model and frontend support](https://github.com/vllm-project/vllm/pull/48799).
2. [Initial single-token MTP support](https://github.com/vllm-project/vllm/pull/48869).
   Current upstream MTP also implements multiple depths; the initial PR is a
   reference rather than a permanent single-token limit.
3. [Common Triton implementation proposal](https://github.com/vllm-project/vllm/pull/61413).
   NPU kernel support and validation remain part of the Ascend adaptation.
