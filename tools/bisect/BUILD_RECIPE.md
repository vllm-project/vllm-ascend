# Bisect dependency recipe

The bisect environment comes from the image build, not from a single version
file. For each candidate commit and hardware target, resolve the following
inputs in build order:

| Layer | Source | What it determines |
| --- | --- | --- |
| Base runtime | Root `Dockerfile` for the target (for example `Dockerfile.a3`) | CANN base image, vLLM checkout, Python install commands, extra indexes, prebuilt wheels and source-built packages |
| vLLM Ascend package | `pyproject.toml` and `requirements.txt` at the candidate commit | Build isolation requirements and runtime Python dependencies; both files must be checked |
| Nightly test image | `.github/workflows/dockerfiles/Dockerfile.nightly.<target>` and `requirements-nightly.txt` | Test dependencies, pip index used for the nightly install, AISBench repository and tag |
| Actual container | Installed package metadata, source checkout commits, base image and compiled extensions | Whether the container already matches the candidate recipe |

The image build workflow chooses the root Dockerfile for each chip and then
builds the matching nightly Dockerfile on top. For A3 Ubuntu these are
`Dockerfile.a3` and `Dockerfile.nightly.a3`. The nightly Dockerfile installs
`requirements-nightly.txt` and AISBench after the base image is built.

Dependency adaptation should compare the complete candidate recipe with the
currently installed environment before every trial. A matching distribution
version is insufficient for an editable source install: the vLLM source
checkout must also match the selected tag or commit. Reinstall changed Python
dependencies in build order, using the indexes from the relevant build step.
Native extensions must be rebuilt or replaced with a compatible wheel when
their source or build inputs change. If the CANN base image changes, the
candidate needs a matching image; a pip reinstall cannot reproduce that layer.

The current adapter selects the root Dockerfile from the bisect `--soc` value
and the running container's Ubuntu/openEuler variant. It reinstalls the
vLLM Ascend package when `requirements.txt` or `pyproject.toml` changes,
reinstalls `requirements-nightly.txt` when it changes, and switches AISBench
when its tag changes. It also checks the vLLM source checkout against the
Dockerfile's selected ref. Changes to other active Dockerfile steps are
reported as unsupported instead of being misclassified as model failures.

Do not infer an active install from an `ARG` alone. In particular, the
standalone torch-npu download block in the current root Dockerfiles is
commented out. `requirements.txt` and `pyproject.toml` still constrain
torch-npu during the vLLM Ascend package install.
