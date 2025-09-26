# Contributing to Omni Infer

Thank you for your interest in contributing to Omni Infer! Our community is open to everyone and welcomes all kinds of contributions, no matter how small or large. There are several ways you can contribute to the project:

* Identify and report any issues or bugs.
* Request or add support for a new model.
* Suggest or implement new features.
* Improve documentation or contribute a how-to guide.

We also believe in the power of community support; thus, answering queries, offering PR reviews, and assisting others are also highly regarded and beneficial contributions.

Finally, one of the most impactful ways to support us is by raising awareness about Omni Infer. Talk about it in your blog posts and highlight how it's driving your incredible projects. Express your support on social media if you're using Omni Infer, or simply offer your appreciation by starring our repository!

## Setting Up & Building from source

Before install Omni Infer, you need to install vllm from github.

```
git clone https://github.com/vllm-project/vllm.git
```

New contributors do not have write permission to push to the official Omni Infer repo. Please fork the repository under Gittee account, then clone your fork locally.

```
git clone https://gitee.com/your_user_name/omniinfer.git
cd omniinfer
pip install .
```

## Issues

If you encounter a bug or have a feature request, please [search existing issues](https://gitee.com/omniai/omniinfer/issues) first to see if it has already been reported. If not, please [file a new issue](https://gitee.com/omniai/omniinfer/issues/new?issue%5Bassignee_id%5D=0&issue%5Bmilestone_id%5D=0), providing as much relevant information as possible.

## Pull Requests & Code Reviews[#](https://docs.vllm.ai/en/v0.7.3/contributing/overview.html#pull-requests-code-reviews "Permalink to this heading")

Thank you for your contribution to Omni Infer! Before submitting the pull request, please ensure the PR meets the following criteria. This helps Omni Infer maintain the code quality and improve the efficiency of the review process.

### CLA Signing

The commnunity will provide a CLA bot for you to sign either ICLA or CCLA. Please ensure you have signed the CLA before submitting a pull request.

### PR Title and Classification

Only specific types of PRs will be reviewed. The PR title is prefixed appropriately to indicate the type of change. Please use one of the following:

* `<span class="pre">[Bugfix]</span>` for bug fixes.
* `<span class="pre">[CI/Build]</span>` for build or continuous integration improvements.
* `<span class="pre">[Doc]</span>` for documentation fixes and improvements.
* `<span class="pre">[Model]</span>` for adding a new model or improving an existing model. Model name should appear in the title.
* `<span class="pre">[Frontend]</span>` For changes on the Omni-Infer frontend (e.g., OpenAI API server, `<span class="pre">LLM</span>` class, etc.)
* `<span class="pre">[Kernel]</span>` for changes affecting CANN kernels or other compute kernels.
* `<span class="pre">[Core]</span>` for changes in the core Omni-Infer logic (e.g., `<span class="pre">LLMEngine</span>`, `<span class="pre">AsyncLLMEngine</span>`, `<span class="pre">Scheduler</span>`, etc.)
* `<span class="pre">[Misc]</span>` for PRs that do not fit the above categories. Please use this sparingly.
