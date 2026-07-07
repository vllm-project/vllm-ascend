---
hide:
  - navigation
  - toc
---

<p class="home-hero">
<img class="md-logo__img--light" src="./logos/vllm-ascend-logo-text-light.png#only-light" alt="vLLM">
<img class="md-logo__img--dark" src="./logos/vllm-ascend-logo-text-dark.png#only-dark" alt="vLLM">
</p>

<p style="text-align:center">
<strong>vLLM Ascend Plugin
</strong>
</p>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/vllm-project/vllm-ascend" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/vllm-project/vllm-ascend/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/vllm-project/vllm-ascend/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>

vLLM Ascend plugin (vllm-ascend) is a community-maintained hardware plugin for running vLLM on the Ascend NPU.

This plugin is the recommended approach for supporting the Ascend backend within the vLLM community. It adheres to the principles outlined in the [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162), providing a hardware-pluggable interface that decouples the integration of the Ascend NPU with vLLM.

By using vLLM Ascend plugin, popular open-source models, including Transformer-like, Mixture-of-Experts, Embedding, Multi-modal LLMs can run seamlessly on the Ascend NPU.
