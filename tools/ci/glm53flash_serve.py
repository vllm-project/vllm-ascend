# SPDX-License-Identifier: Apache-2.0
"""Test-only vLLM CLI wrapper registering the fixed synthetic Flash loader."""

from vllm.entrypoints.cli.main import main

from tools.ci.glm53flash_collect import FlashDummyLoader  # noqa: F401

if __name__ == "__main__":
    main()
