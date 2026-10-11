# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Opt-in startup hook; place this directory first on PYTHONPATH."""

from bounded_replay_import import install

try:
    install()
except ImportError as error:
    # Python normally prints and ignores exceptions from sitecustomize. Stop
    # startup instead of silently serving without the requested replay feature.
    raise SystemExit(str(error)) from error
