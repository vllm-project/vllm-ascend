# Contributing

## Building and testing
It's recommended to set up a local development environment to build and test
before you submit a PR.

### Setup development environment

Theoretically, the vllm-ascend build is only supported on Linux because
`vllm-ascend` dependency `torch_npu` only supports Linux.

But you can still set up dev env on Linux/Windows/macOS for linting and basic
test as following commands:

#### Run lint locally

```bash
# Choose a base dir (~/vllm-project/) and set up venv
cd ~/vllm-project/
python3 -m venv .venv
source ./.venv/bin/activate

# Clone vllm-ascend and install
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend

# Install lint requirement and enable pre-commit hook
pip install -r requirements-lint.txt

# Run lint (You need install pre-commits deps via proxy network at first time)
bash format.sh
```

#### Run CI locally

After complete "Run lint" setup, you can run CI locally:

```{code-block} bash
   :substitutions:

cd ~/vllm-project/

# Run CI need vLLM installed
git clone --branch |vllm_version| https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/build.txt
VLLM_TARGET_DEVICE="empty" pip install .
cd ..

# Install requirements
cd vllm-ascend
# For Linux:
pip install -r requirements-dev.txt
# For non Linux:
cat requirements-dev.txt | grep -Ev '^#|^--|^$|^-r' | while read PACKAGE; do pip install "$PACKAGE"; done
cat requirements.txt | grep -Ev '^#|^--|^$|^-r' | while read PACKAGE; do pip install "$PACKAGE"; done

# Run ci:
bash format.sh ci
```

#### Submit the commit

```bash
# Commit changed files using `-s`
git commit -sm "your commit info"
```

🎉 Congratulations! You have completed the development environment setup.

### Test locally

You can refer to [Testing](./testing.md) doc to help you setup testing environment and running tests locally.

## DCO and Signed-off-by

When contributing changes to this project, you must agree to the DCO. Commits must include a `Signed-off-by:` header which certifies agreement with the terms of the DCO.

Using `-s` with `git commit` will automatically add this header.

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

:::{note}
If the PR spans more than one category, please include all relevant prefixes.
:::

## Others

You may find more information about contributing to vLLM Ascend backend plugin on [<u>docs.vllm.ai</u>](https://docs.vllm.ai/en/latest/contributing/overview.html).
If you find any problem when contributing, you can feel free to submit a PR to improve the doc to help other developers.

:::{toctree}
:caption: Index
:maxdepth: 1
testing
:::
