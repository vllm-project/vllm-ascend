# Versioning Policy

Starting with vLLM 0.7.x, the vLLM Ascend Plugin ([vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)) project follows [PEP 440](https://peps.python.org/pep-0440/) to publish versions matching vLLM ([vllm-project/vllm](https://github.com/vllm-project/vllm)).

## vLLM Ascend Plugin versions

Each vLLM Ascend release is versioned as `v[major].[minor].[micro][rcN][.postN]` (such as
`v0.7.3rc1`, `v0.7.3`, `v0.7.3.post1`)

- **Final releases**: Typically scheduled every three months, with careful alignment to the vLLM upstream release cycle and the Ascend software product roadmap.
- **Pre releases**: Typically issued **on demand**, labeled with rcN to indicate the Nth release candidate. They are intended to support early testing by users ahead of the final release.
- **Post releases**: Typically issued **on demand** to address minor errors in a final release. Different from [PEP-440 post release note](https://peps.python.org/pep-0440/#post-releases) convention, these versions include actual bug fixes, as the final release version must strictly align with the vLLM final release format (`v[major].[minor].[micro]`). Any post version must be published as a patch version of the final release.

For example:

- `v0.7.x`: first final release to match the vLLM `v0.7.x` version.
- `v0.7.3rc1`: first pre version of vLLM Ascend.
- `v0.7.3.post1`: post release for the `v0.7.3` release if it has some minor errors.

## Release compatibility matrix

The table below is the release compatibility matrix for vLLM Ascend release.

| vLLM Ascend | vLLM              | Python          | Stable CANN |        PyTorch/torch_npu        | Triton Ascend |
|-------------|-------------------|-----------------|-------------|---------------------------------|---------------|
| v0.15.0rc1  | v0.15.0           | >= 3.10, < 3.12 | 8.5.0       | 2.9.0  / 2.9.0                  | 3.2.0         |
| v0.14.0rc1  | v0.14.1           | >= 3.10, < 3.12 | 8.5.0       | 2.9.0  / 2.9.0                  | 3.2.0         |
| v0.13.0     | v0.13.0           | >= 3.10, < 3.12 | 8.5.0       | 2.9.0  / 2.8.0.post2            | 3.2.0         |
| v0.13.0rc2  | v0.13.0           | >= 3.10, < 3.12 | 8.5.0       | 2.8.0  / 2.8.0.post1            | 3.2.0         |
| v0.13.0rc1  | v0.13.0           | >= 3.10, < 3.12 | 8.3.RC2     | 2.8.0  / 2.8.0                  |               |
| v0.12.0rc1  | v0.12.0           | >= 3.10, < 3.12 | 8.3.RC2     | 2.8.0  / 2.8.0                  |               |
| v0.11.0     | v0.11.0           | >= 3.9, < 3.12 | 8.3.RC2     | 2.7.1 / 2.7.1.post1             |               |
| v0.11.0rc3  | v0.11.0           | >= 3.9, < 3.12  | 8.3.RC2     | 2.7.1 / 2.7.1.post1             |               |
| v0.11.0rc2  | v0.11.0           | >= 3.9, < 3.12  | 8.3.RC2     | 2.7.1 / 2.7.1                   |               |
| v0.11.0rc1  | v0.11.0           | >= 3.9, < 3.12  | 8.3.RC1     | 2.7.1 / 2.7.1                   |               |
| v0.11.0rc0  | v0.11.0rc3        | >= 3.9, < 3.12  | 8.2.RC1     | 2.7.1 / 2.7.1.dev20250724       |               |
| v0.10.2rc1  | v0.10.2           | >= 3.9, < 3.12  | 8.2.RC1     | 2.7.1 / 2.7.1.dev20250724       |               |
| v0.10.1rc1  | v0.10.1/v0.10.1.1 | >= 3.9, < 3.12  | 8.2.RC1     | 2.7.1 / 2.7.1.dev20250724       |               |
| v0.10.0rc1  | v0.10.0           | >= 3.9, < 3.12  | 8.2.RC1     | 2.7.1 / 2.7.1.dev20250724       |               |
| v0.9.2rc1   | v0.9.2            | >= 3.9, < 3.12  | 8.1.RC1     | 2.5.1 / 2.5.1.post1.dev20250619 |               |
| v0.9.1      | v0.9.1            | >= 3.9, < 3.12  | 8.2.RC1     | 2.5.1 / 2.5.1.post1             |               |
| v0.9.1rc3   | v0.9.1            | >= 3.9, < 3.12  | 8.2.RC1     | 2.5.1 / 2.5.1.post1             |               |
| v0.9.1rc2   | v0.9.1            | >= 3.9, < 3.12  | 8.2.RC1     | 2.5.1 / 2.5.1.post1             |               |
| v0.9.1rc1   | v0.9.1            | >= 3.9, < 3.12  | 8.1.RC1     | 2.5.1 / 2.5.1.post1.dev20250528 |               |
| v0.9.0rc2   | v0.9.0            | >= 3.9, < 3.12  | 8.1.RC1     | 2.5.1 / 2.5.1                   |               |
| v0.9.0rc1   | v0.9.0            | >= 3.9, < 3.12  | 8.1.RC1     | 2.5.1 / 2.5.1                   |               |
| v0.8.5rc1   | v0.8.5.post1      | >= 3.9, < 3.12  | 8.1.RC1     | 2.5.1 / 2.5.1                   |               |
| v0.8.4rc2   | v0.8.4            | >= 3.9, < 3.12  | 8.0.0       | 2.5.1 / 2.5.1                   |               |
| v0.7.3.post1| v0.7.3            | >= 3.9, < 3.12  | 8.1.RC1     | 2.5.1 / 2.5.1                   |               |
| v0.7.3      | v0.7.3            | >= 3.9, < 3.12  | 8.1.RC1     | 2.5.1 / 2.5.1                   |               |

:::{note}
If you're using v0.7.3, don't forget to install [mindie-turbo](https://pypi.org/project/mindie-turbo) as well.
:::

For main branch of vLLM Ascend, we usually make it compatible with the latest vLLM release and a newer commit hash of vLLM. Please note that this table is usually updated. Please check it regularly.

| vLLM Ascend | vLLM         | Python           | Stable CANN | PyTorch/torch_npu  |
|-------------|--------------|------------------|-------------|--------------------|
|     main    | 83b47f67b1dfad505606070ae4d9f83e50ad4ebd, v0.15.0 tag | >= 3.10, < 3.12   | 8.5.0 | 2.9.0 / 2.9.0 |

## Release cadence

### Release window

| Date       | Event                                     |
|------------|-------------------------------------------|
| 2026.02.26 | Release candidates, v0.15.0rc1            |
| 2026.02.06 | v0.13.0 Final release, v0.13.0            |
| 2026.01.26 | Release candidates, v0.14.0rc1            |
| 2026.01.24 | Release candidates, v0.13.0rc2            |
| 2025.12.27 | Release candidates, v0.13.0rc1            |
| 2025.12.16 | v0.11.0 Final release, v0.11.0            |
| 2025.12.13 | Release candidates, v0.12.0rc1            |
| 2025.12.03 | Release candidates, v0.11.0rc3            |
| 2025.11.21 | Release candidates, v0.11.0rc2            |
| 2025.11.10 | Release candidates, v0.11.0rc1            |
| 2025.09.30 | Release candidates, v0.11.0rc0            |
| 2025.09.16 | Release candidates, v0.10.2rc1            |
| 2025.09.04 | Release candidates, v0.10.1rc1            |
| 2025.09.03 | v0.9.1 Final release, v0.9.1              |
| 2025.08.22 | Release candidates, v0.9.1rc3             |
| 2025.08.07 | Release candidates, v0.10.0rc1            |
| 2025.08.04 | Release candidates, v0.9.1rc2             |
| 2025.07.11 | Release candidates, v0.9.2rc1             |
| 2025.06.22 | Release candidates, v0.9.1rc1             |
| 2025.06.10 | Release candidates, v0.9.0rc2             |
| 2025.06.09 | Release candidates, v0.9.0rc1             |
| 2025.05.29 | v0.7.3 post release, v0.7.3.post1         |
| 2025.05.08 | v0.7.3 Final release, v0.7.3              |
| 2025.05.06 | Release candidates, v0.8.5rc1             |
| 2025.04.28 | Release candidates, v0.8.4rc2             |
| 2025.04.18 | Release candidates, v0.8.4rc1             |
| 2025.03.28 | Release candidates, v0.7.3rc2             |
| 2025.03.14 | Release candidates, v0.7.3rc1             |
| 2025.02.19 | Release candidates, v0.7.1rc1             |

## Branch policy

vLLM Ascend includes two branches: main and dev.

- **main**: corresponds to the vLLM main branch and latest 1 or 2 release version. It is continuously monitored for quality through Ascend CI.
- **releases/vX.Y.Z**: development branch, created with part of new releases of vLLM. For example, `releases/v0.13.0` is the dev branch for vLLM `v0.13.0` version.

Commits should typically be merged into the main branch first, and only then backported to the dev branch, to reduce maintenance costs as much as possible.

### Maintenance branch and EOL

The table below lists branch states.

| Branch            | Time Frame                       | Summary                                                   |
| ----------------- | -------------------------------- | --------------------------------------------------------- |
| Maintained        | Approximately 2-3 minor versions | Bugfixes received; releases produced; CI commitment       |
| Unmaintained      | Community-interest driven        | Bugfixes received; no releases produced; no CI commitment |
| End of Life (EOL) | N/A                              | Branch no longer accepting changes                        |

### Branch states

Note that vLLM Ascend will only be released for a certain vLLM release version, not for every version. Hence, you may notice that some versions have corresponding dev branches (e.g. `releases/v0.13.0`), while others do not (e.g. `releases/v0.12.0`). The vLLM Ascend release branch now follows the `releases/vX.Y.Z` naming convention, replacing the previous `vX.Y.Z-dev` format to align with vLLM's branch naming standards.

Usually, each minor version of vLLM (such as 0.7) corresponds to a vLLM Ascend version branch and supports its latest version (such as 0.7.3), as shown below:

| Branch     | State        | Note                                                     |
| ---------- | ------------ | -------------------------------------------------------- |
| main       | Maintained   | CI commitment for vLLM main branch and vLLM 0.13.0 tag |
| releases/v0.13.0 | Maintained | CI commitment for vLLM 0.13.0 version                |
| v0.11.0-dev| Maintained   | CI commitment for vLLM 0.11.0 version |
| v0.9.1-dev | Maintained   | CI commitment for vLLM 0.9.1 version                     |
| v0.7.3-dev | Maintained   | CI commitment for vLLM 0.7.3 version                     |
| v0.7.1-dev | Unmaintained | Replaced by v0.7.3-dev                                   |

### Feature branches

| Branch     | State       | RFC Link                             | Scheduled Merge Time | Mentor |
|------------|--------------|---------------------------------------|------------|--------|
|rfc/long_seq_optimization|Maintained|<https://github.com/vllm-project/vllm/issues/22693>|930|wangxiyuan|

- Branch: The feature branch should be created with a prefix `rfc/` followed by the feature name, such as `rfc/feature-name`.
- State: The state of the feature branch is `Maintained` until it is merged into the main branch or deleted.
- RFC Link: The feature branch should be created with a corresponding RFC issue. The creation of a feature branch requires an RFC and approval from at least two maintainers.
- Scheduled Merge Time: The final goal of a feature branch is to be merged into the main branch. If it remains unmerged for more than three months, the mentor maintainer should evaluate whether to delete the branch.
- Mentor: The mentor should be a vLLM Ascend maintainer who is responsible for the feature branch.

### Backward compatibility

For main branch, vLLM Ascend should work with vLLM main branch and latest 1 or 2 releases. To ensure backward compatibility, do as follows:

- Both main branch and target vLLM release, such as the vLLM main branch and vLLM 0.8.4, are tested by Ascend E2E CI.
- To make sure that code changes are compatible with the latest 1 or 2 vLLM releases, vLLM Ascend introduces a version check mechanism inside the code. It checks the version of the installed vLLM package first to decide which code logic to use. If users hit the `InvalidVersion` error, it may indicate that they have installed a dev or editable version of vLLM package. In this case, we provide the env variable `VLLM_VERSION` to let users specify the version of vLLM package to use.
- Document changes should be compatible with the latest 1 or 2 vLLM releases. Notes should be added if there are any breaking changes.

## Document branch policy

To reduce maintenance costs, **all branch documentation content should remain consistent, and version differences can be controlled via variables in [docs/source/conf.py](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/conf.py)**. While this is not a simple task, it is a principle we should strive to follow.

| Version | Purpose | Code Branch |
|-----|-----|---------|
| latest | Doc for the latest rc release of main branch | `main` branch |
| rc version | Doc for RC released versions | `vX.Y.ZrcN` --> `vX.Y.ZrcN` tag |
| version | Doc for historical released versions | `vX.Y.Z` --> `releases/vX.Y.Z` branch |

Notes:

- `latest` documentation: always points to latest rc release of main branch.
- `rc version` documentation: there are no further updates after release.
- `version` documentation: keep updating the `releases/vX.Y.Z` branch documentation to fix doc bugs.

## Software dependency management

- `torch-npu`: Ascend Extension for PyTorch (torch-npu) releases a stable version to [PyPi](https://pypi.org/project/torch-npu)
  every 3 months, a development version (aka the POC version) every month, and a nightly version every day.
  The PyPi stable version **CAN** be used in vLLM Ascend final version, the monthly dev version **ONLY CAN** be used in
  vLLM Ascend RC version for rapid iteration, and the nightly version **CANNOT** be used in vLLM Ascend any version and branch.
