# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fields used by the shared Lite Actions Smoke catalog."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OverrideDefinitions:
    test_name: str = "default"
    test_descr: str = "default"
    ngpu: int = 4
    nnodes: int = 1
    timeout: int | None = None
    env_vars: dict[str, str] | None = None
    ci_kind: str = "train"
    ci_overrides: dict[str, str] = field(default_factory=dict)
    disabled: bool = False
    check_loss: bool = True
    device: str | None = None
    required_env: tuple[str, ...] = ()
    override_args: tuple[tuple[str, ...], ...] = ((" ",),)
