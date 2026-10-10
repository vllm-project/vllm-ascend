# Contributing

## Building and Testing

It's recommended to set up a local development environment to build vllm-ascend and run tests
before you submit a PR.

### Set up a development environment

Theoretically, the vllm-ascend build is only supported on Linux because
`vllm-ascend` dependency `torch_npu` only supports Linux.

But you can still set up a development environment on Linux/Windows/macOS for linting and running basic
tests.

#### Run lint locally

```bash
# Choose a base dir (~/vllm-project/) and set up venv
cd ~/vllm-project/
python3 -m venv .venv
source ./.venv/bin/activate

# Clone vllm-ascend and install
git clone --branch main https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend

# Install lint requirement and enable pre-commit hook
pip install -r requirements-lint.txt

# Run lint (You need to install pre-commits deps via proxy network at first time)
bash format.sh
```

#### Run CI locally

After completing "Run lint" setup, you can run CI (Continuous integration) locally:

```bash
cd ~/vllm-project/

# Install the vLLM commit verified by the main-branch plugin checkout.
VLLM_COMMIT=$(tr -d '[:space:]' < vllm-ascend/.github/vllm-main-verified.commit)
git init vllm
git -C vllm fetch --depth 1 https://github.com/vllm-project/vllm.git "$VLLM_COMMIT"
git -C vllm checkout --detach FETCH_HEAD
cd vllm
VLLM_TARGET_DEVICE="empty" pip install .
cd ..

# Install requirements
cd vllm-ascend
# For Linux:
pip install -r requirements-dev.txt
# For non-Linux:
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

### Testing locally

You can refer to [Testing](./testing.md)  to set up a testing environment and running tests locally.

### Local native build cache

When a source build compiles native csrc actions, their reusable local entries are
stored in the ignored `csrc/build_cache` directory. Set
`VLLM_ASCEND_BUILD_CACHE_DIR` to an absolute path to use a different directory.
This is local build state, not a remote OBS snapshot; deleting it while no
build is running only makes the next relevant source build cold. Builds that
disable custom-kernel compilation do not exercise the custom-operator action
cache.

If you change a generated operator input, compiler command, or toolchain
dependency, review the [cache identity and CMake integration contract](../Design_Documents/persistent_csrc_build_cache.md#cmake-integration-contract)
in the design document.

## DCO and Signed-off-by

When contributing changes to this project, you must agree to the DCO. Commits must include a `Signed-off-by:` header which certifies agreement with the terms of the DCO (Developer Certificate of Origin).

Using `-s` with `git commit` will automatically add this header.

## PR Title and Classification

Only specific types of PRs will be reviewed. The PR title must contain one of the following type prefixes. This is enforced by CI in [pr_test.yaml](https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/pr_test.yaml): a PR whose title contains none of them fails before any test runs.

- `[BugFix]` for bug fixes.
- `[Performance]` for performance optimization.
- `[Feature]` for new features.
- `[Refactor]` for refactoring that does not change behavior.
- `[Test]` for tests (such as unit tests).
- `[CI]` for build or continuous integration improvements.
- `[Doc]` for documentation fixes and improvements.
- `[Community]` for community related changes.
- `[Misc]` for PRs that do not fit the above categories. Please use this sparingly.

A module prefix may be added alongside the type prefix to indicate the affected area, such as `[BugFix][Attention]` or `[Feature][Worker]`. Module prefixes are free-form and are not checked by CI. Commonly used ones are:

- `[Attention]` for attention.
- `[Communicator]` for communicators.
- `[ModelRunner]` for model runner.
- `[Platform]` for platform.
- `[Worker]` for worker.
- `[Core]` for the core vllm-ascend logic (such as platform, attention, communicators, model runner).
- `[Kernel]` for compute kernels and ops.

!!! note

    If the PR spans more than one category, please include all relevant prefixes. At least one of the type prefixes listed above must be present, otherwise CI rejects the PR.

## Others

You may find more information about contributing to vLLM Ascend backend plugin on [<u>docs.vllm.ai</u>](https://docs.vllm.ai/en/latest/contributing).
If you encounter any problems while contributing, feel free to submit a PR to improve the documentation to help other developers.
