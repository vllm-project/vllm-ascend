#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#


def register_rforkloader() -> None:
    """Register the RFork model loader plugin."""
    from .rfork_loader import RForkModelLoader  # noqa: F401
