# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

curl -X POST http://127.0.0.1:8089/v1/completions  \
     -H "Content-Type: application/json" \
     -d '{
     	 "model": "deepseek",
     	 "prompt": ["Hello, my name is", "The future of AI is", "This is a", "It is known that", "How is the", "If we want to", "He does this", "Maybe it is"],
     	 "max_tokens": 100,
     	 "temperature": 0.0,
      	 "top_p": 1,
     	 "top_k": -1
 	 }' \
	 -m 99999