# Versioning policy

Starting with vLLM 0.7.x, the vLLM Ascend Plugin ([vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)) project follows the [PEP 440](https://peps.python.org/pep-0440/) to publish matching with vLLM ([vllm-project/vllm](https://github.com/vllm-project/vllm)).

## vLLM Ascend Plugin versions

Each vllm-ascend release will be versioned: `v[major].[minor].[micro][rcN][.postN]` (such as
`v0.7.1rc1`, `v0.7.1`, `v0.7.1.post1`)

- **Final releases**: will typically be released every **3 months**, will take the vLLM upstream release plan and Ascend software product release plan into comprehensive consideration.
- **Pre releases**: will typically be released **on demand**, ending with rcN, represents the Nth release candidate version, to support early testing by our users prior to a final release.
- **Post releases**: will typically be released **on demand** to support to address minor errors in a final release. It's different from [PEP-440 post release note](https://peps.python.org/pep-0440/#post-releases) suggestion, it will contain actual bug fixes considering that the final release version should be matched strictly with the vLLM final release version (`v[major].[minor].[micro]`). The post version has to be published as a patch version of the final release.

For example:
- `v0.7.x`: it's the first final release to match the vLLM `v0.7.x` version.
- `v0.7.1rc1`: will be the first pre version of vllm-ascend.
- `v0.7.1.post1`: will be the post release if the `v0.7.1` release has some minor errors.

## Branch policy

vllm-ascend has main branch and dev branch.

- **main**: main branch，corresponds to the vLLM main branch, and is continuously monitored for quality through Ascend CI.
- **vX.Y.Z-dev**: development branch, created with part of new releases of vLLM. For example, `v0.7.1-dev` is the dev branch for vLLM `v0.7.1` version.

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

| Branch    | Status     | Note                                 |
|-----------|------------|--------------------------------------|
| main      | Maintained | CI commitment for vLLM main branch   |
| v0.7.1-dev | Maintained | CI commitment for vLLM 0.7.1 version |

## Release Compatibility Matrix

Following is the Release Compatibility Matrix for vLLM Ascend Plugin:

| vllm-ascend  | vLLM         | Python | Stable CANN | PyTorch/torch_npu |
|--------------|--------------| --- | --- | --- |
| v0.7.1.rc1 | v0.7.1 | 3.9 - 3.12 | 8.0.0   |  2.5.1 / 2.5.1.dev20250218 |

## Release cadence

### Next final release (`v0.7.x`) window

| Date | Event |
|---------|-----------|
| February 2025 | Release candidates (RC1), v0.7.1rc1 |
| March 2025 | Release candidates (RC2), v0.7.1rc2 or v0.7.3rc1 |
| March 2025 | Final release passes, match vLLM v0.7.x latest: v0.7.1 or v0.7.3 |
