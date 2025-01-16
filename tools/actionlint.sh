#!/bin/bash

if command -v actionlint &> /dev/null; then
    # NOTE: avoid check .github/workflows/vllm_ascend_test.yaml becase sel-hosted runner `npu-arm64` is unknown
    actionlint .github/workflows/*.yml .github/workflows/mypy.yaml
    exit 0
elif [ -x ./actionlint ]; then
    ./actionlint .github/workflows/*.yml .github/workflows/mypy.yaml
    exit 0
fi

# download a binary to the current directory - v1.7.3
bash <(curl https://raw.githubusercontent.com/rhysd/actionlint/aa0a7be8e566b096e64a5df8ff290ec24fa58fbc/scripts/download-actionlint.bash)
./actionlint  .github/workflows/*.yml .github/workflows/mypy.yaml
