# DeepSeek-V4.1 vision encoder graphs

Enable the packed ViT and aligner ACL graph path with
`--compilation-config '{"cudagraph_mm_encoder": true}'`.
The existing encoder graph manager selects budgets and falls back to eager
execution for inputs outside captured budgets. Image delimiter assembly remains
outside the captured graph. The default eager path is unchanged.

This integration reuses upstream vLLM's DeepseekV4VLEncoderCudaGraphMixin
(PRs #56625 and #58499) and requires that module in the installed vLLM version.
Ascend MMEncoderAttention supplies FIA capture and replay updates.

Before deployment, compare eager and graph image embeddings and greedy outputs
on NPU for single images, mixed resolutions, odd grids, consecutive replay with
changed images, and budget overflow. CPU tests do not establish NPU capture or
numerical correctness. Assign the release fragment a PR number when submitting.
