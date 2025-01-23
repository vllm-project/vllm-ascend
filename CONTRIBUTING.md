# Contributing to vLLM Ascend backend plugin

## DCO and Signed-off-by

When contributing changes to this project, you must agree to the DCO. Commits must include a `Signed-off-by:` header which certifies agreement with the terms of the DCO.

Using `-s` with `git commit` will automatically add this header.

## Testing

```bash
pip install -r requirements-dev.txt

# 1. linting and formatting
bash format.sh
# 2. Unit tests
pytest tests/
# 3. Commit changed files using `-s`
git commit -sm "your commit info"
```

## PR Title and Classification

Only specific types of PRs will be reviewed. The PR title is prefixed appropriately to indicate the type of change. Please use one of the following:

- `[Attention]` for new features or optimization in attention.
- `[Communicator]` for new features or optimization in communicators.
- `[ModelRunner]` for new features or optimization in model runner.
- `[Platform]` for new features or optimization in platform.
- `[Worker]` for new features or optimization in worker.
- `[Core]` for new features or optimization  in the core vllm-ascend logic (such as platform, attention, communicators, model runner)
- `[Kernel]` changes affecting compute kernels and ops.
- `[Bugfix]` for bug fixes.
- `[Doc]` for documentation fixes and improvements.
- `[Test]` for tests (such as unit tests).
- `[CI]` for build or continuous integration improvements.
- `[Misc]` for PRs that do not fit the above categories. Please use this sparingly.

> [!NOTE]
> If the PR spans more than one category, please include all relevant prefixes.

## Others

You may find more information about contributing to vLLM Ascend backend plugin on [<u>docs.vllm.ai</u>](https://docs.vllm.ai/en/latest/contributing/overview.html).
