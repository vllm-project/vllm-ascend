import os
from typing import Dict, List
from setuptools import setup

ROOT_DIR = os.path.dirname(__file__)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line.startswith("--"):
                continue
            else:
                resolved_requirements.append(line)
        return resolved_requirements
    
    try:
        requirements = _read_requirements("requirements.txt")
    except ValueError:
        print("Failed to read requirements.txt in vllm_ascend_plugin.")
    return requirements


setup(name='vllm_ascend_plugin',
      version='0.1',
      packages=['vllm_ascend_plugin'],
      install_requires=get_requirements(),
      entry_points={
          'vllm.platform_plugins':
          ["ascend_plugin = vllm_ascend_plugin:register"]
      })
