"""Semantic intermediate representation of a Docker-built bisect environment."""

from dataclasses import dataclass

import tomllib

from tools.bisect.build_recipe import BuildRecipe


@dataclass(frozen=True)
class EnvironmentEffect:
    name: str
    value: str
    action: str


def compile_environment(recipe: BuildRecipe) -> tuple[EnvironmentEffect, ...]:
    """Project the ordered Docker build into effects the in-place adapter knows."""
    project = tomllib.loads(recipe.pyproject)
    build_metadata = (project.get("build-system"), project.get("project"), project.get("tool", {}).get("setuptools"))
    return (
        EnvironmentEffect("CANN base image", recipe.cann_image, "matching image required"),
        EnvironmentEffect("vLLM source", recipe.vllm_ref, "switch vLLM"),
        EnvironmentEffect("PyTorch index", recipe.pytorch_index_url, "use for pip installs"),
        EnvironmentEffect("Ascend index", recipe.ascend_index_url, "use for pip installs"),
        EnvironmentEffect("nightly index", recipe.nightly_index_url, "use for nightly pip installs"),
        EnvironmentEffect("vLLM Ascend requirements", recipe.requirements, "rebuild vLLM Ascend"),
        EnvironmentEffect("package build metadata", repr(build_metadata), "rebuild vLLM Ascend"),
        EnvironmentEffect("nightly requirements", recipe.requirements_nightly, "reinstall nightly requirements"),
        EnvironmentEffect("runtime wheels", repr(recipe.runtime_wheels), "reinstall runtime wheels"),
        EnvironmentEffect("flash-attention-npu source", recipe.flash_attention_ref or "", "rebuild extension"),
        EnvironmentEffect("AISBench source", recipe.ais_bench_url, "matching image required"),
        EnvironmentEffect("AISBench tag", recipe.ais_bench_tag, "switch AISBench"),
    )


def changed_effects(previous: BuildRecipe, target: BuildRecipe) -> list[tuple[EnvironmentEffect, EnvironmentEffect]]:
    """Compare semantic outputs after parsing the two hardware build chains."""
    return [
        (old, new)
        for old, new in zip(compile_environment(previous), compile_environment(target), strict=True)
        if old.value != new.value
    ]
