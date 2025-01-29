#!/bin/bash

#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from https://github.com/vllm-project/vllm/tree/main/tools
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Ensure that *.excalidraw.png files have the excalidraw metadata
# embedded in them. This ensures they can be loaded back into
# the tool and edited in the future.

find . -iname '*.excalidraw.png' | while read -r file; do
	if git check-ignore -q "$file"; then
		continue
	fi
	if ! grep -q "excalidraw+json" "$file"; then
		echo "$file was not exported from excalidraw with 'Embed Scene' enabled."
		exit 1
	fi
done
