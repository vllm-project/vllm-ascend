# SPDX-License-Identifier: Apache-2.0


def pytest_addoption(parser):
    parser.addoption(
        "--expert-offload-model",
        default=None,
        help=(
            "Local DeepSeek V4 checkpoint for full target/DSpark offload acceptance; "
            "requires sufficient pinned host memory"
        ),
    )
