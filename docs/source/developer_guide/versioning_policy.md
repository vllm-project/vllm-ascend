# Versioning policy

Starting with vLLM 0.7.x, the vLLM Ascend Plugin ([vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)) project follows the [PEP 440](https://peps.python.org/pep-0440/) to publish matching with vLLM ([vllm-project/vllm](https://github.com/vllm-project/vllm)).

## vLLM Ascend Plugin versions

Each vllm-ascend release will be versioned: `v[major].[minor].[micro][rcN][.postN]` (such as
`v0.7.3rc1`, `v0.7.3`, `v0.7.3.post1`)

- **Final releases**: will typically be released every **3 months**, will take the vLLM upstream release plan and Ascend software product release plan into comprehensive consideration.
- **Pre releases**: will typically be released **on demand**, ending with rcN, represents the Nth release candidate version, to support early testing by our users prior to a final release.
- **Post releases**: will typically be released **on demand** to support to address minor errors in a final release. It's different from [PEP-440 post release note](https://peps.python.org/pep-0440/#post-releases) suggestion, it will contain actual bug fixes considering that the final release version should be matched strictly with the vLLM final release version (`v[major].[minor].[micro]`). The post version has to be published as a patch version of the final release.

For example:
- `v0.7.x`: it's the first final release to match the vLLM `v0.7.x` version.
- `v0.7.3rc1`: will be the first pre version of vllm-ascend.
- `v0.7.3.post1`: will be the post release if the `v0.7.3` release has some minor errors.

## Branch policy

vllm-ascend has main branch and dev branch.

- **main**: main branch，corresponds to the vLLM main branch and latest 1 or 2 release version. It is continuously monitored for quality through Ascend CI.
- **vX.Y.Z-dev**: development branch, created with part of new releases of vLLM. For example, `v0.7.3-dev` is the dev branch for vLLM `v0.7.3` version.

Usually, a commit should be ONLY first merged in the main branch, and then backported to the dev branch to reduce maintenance costs as much as possible.

### Maintenance branch and EOL:
The branch status will be in one of the following states:

| Branch            | Time frame                       | Summary                                                              |
|-------------------|----------------------------------|----------------------------------------------------------------------|
| Maintained        | Approximately 2-3 minor versions | All bugfixes are appropriate. Releases produced, CI commitment.      |
| Unmaintained      | Community interest driven        | All bugfixes are appropriate. No Releases produced, No CI commitment |
| End of Life (EOL) | N/A                              | Branch no longer accepting changes                                   |

### Branch state

Note that vllm-ascend will only be released for a certain vLLM release version rather than all versions. Hence, You might see only part of versions have dev branches (such as only `0.7.1-dev` / `0.7.3-dev` but no `0.7.2-dev`), this is as expected.

Usually, each minor version of vLLM (such as 0.7) will correspond to a vllm-ascend version branch and support its latest version (for example, we plan to support version 0.7.3) as following shown:

| Branch     | Status       | Note                                 |
|------------|--------------|--------------------------------------|
| main       | Maintained   | CI commitment for vLLM main branch and vLLM 0.8.x branch   |
| v0.7.3-dev | Maintained   | CI commitment for vLLM 0.7.3 version |
| v0.7.1-dev | Unmaintained | Replaced by v0.7.3-dev               |

### Backward compatibility

For main branch, vllm-ascend should works with vLLM main branch and latest 1 or 2 release version. So to ensure the backward compatibility, we will do the following:
- Both main branch and target vLLM release is tested by Ascend E2E CI. For example, currently, vLLM main branch and vLLM 0.8.4 are tested now.
- For code changes, we will make sure that the changes are compatible with the latest 1 or 2 vLLM release version as well. In this case, vllm-ascend introduced a version check machinism inner the code. It'll check the version of installed vLLM pacakge first to decide which code logic to use. If users hit the `InvalidVersion` error, it sometimes means that they have installed an dev/editable version of vLLM package. In this case, we provide the env variable `VLLM_VERSION` to let users specify the version of vLLM package to use.
- For documentation changes, we will make sure that the changes are compatible with the latest 1 or 2 vLLM release version as well. Note should be added if there are any breaking changes.

## Document Branch Policy
To reduce maintenance costs, **all branch documentation content should remain consistent, and version differences can be controlled via variables in [docs/source/conf.py](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/conf.py)**. While this is not a simple task, it is a principle we should strive to follow.

| Version | Purpose | Code Branch |
|-----|-----|---------|
| latest | Doc for the latest dev branch | vX.Y.Z-dev (Will be `main` after the first final release) |
| version | Doc for historical released versions | Git tags, like vX.Y.Z[rcN] |
| stable（not yet released） | Doc for latest final release branch | Will be `vX.Y.Z-dev` after the first official release |

As shown above:

- `latest` documentation: Matches the current maintenance branch `vX.Y.Z-dev` (Will be `main` after the first final release). Continuously updated to ensure usability for the latest release.
- `version` documentation: Corresponds to specific released versions (e.g., `v0.7.3`, `v0.7.3rc1`). No further updates after release.
- `stable` documentation (**not yet released**): Official release documentation. Updates are allowed in real-time after release, typically based on vX.Y.Z-dev. Once stable documentation is available, non-stable versions should display a header warning: `You are viewing the latest developer preview docs. Click here to view docs for the latest stable release.`.

## Software Dependency Management
- `torch-npu`: Ascend Extension for PyTorch (torch-npu) releases a stable version to [PyPi](https://pypi.org/project/torch-npu)
  every 3 months, a development version (aka the POC version) every month, and a nightly version every day.
  The PyPi stable version **CAN** be used in vLLM Ascend final version, the monthly dev version **ONLY CANN** be used in
  vLLM Ascend RC version for rapid iteration, the nightly version **CANNOT** be used in vLLM Ascend any version and branches.

## Release Compatibility Matrix

Following is the Release Compatibility Matrix for vLLM Ascend Plugin:

| vllm-ascend  | vLLM         | Python         | Stable CANN | PyTorch/torch_npu |
|--------------|--------------|----------------| --- | --- |
| v0.8.4rc1 | v0.8.4 | >= 3.9, < 3.12 | 8.0.0   |  2.5.1 / 2.5.1.dev20250320 |
| v0.7.3rc2 | v0.7.3 | >= 3.9, < 3.12 | 8.0.0   |  2.5.1 / 2.5.1.dev20250320 |
| v0.7.3rc1 | v0.7.3 | >= 3.9, < 3.12 | 8.0.0   |  2.5.1 / 2.5.1.dev20250308 |
| v0.7.1rc1 | v0.7.1 | >= 3.9, < 3.12 | 8.0.0   |  2.5.1 / 2.5.1.dev20250218 |

## Release cadence

### release window

| Date       | Event                                     |
|------------|-------------------------------------------|
| End of 2025.04 | v0.7.x Final release, v0.7.3          |
| 2025.04.18 | Release candidates, v0.8.4rc1             |
| 2025.03.28 | Release candidates, v0.7.3rc2             |
| 2025.03.14 | Release candidates, v0.7.3rc1             |
| 2025.02.19 | Release candidates, v0.7.1rc1             |
