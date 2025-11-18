import base64
import os

import requests
from modelscope import snapshot_download  # type: ignore

mm_dir = snapshot_download("vllm-ascend/mm_request", repo_type='dataset')
image_path = os.path.join(mm_dir, "test_mm2.jpg")
with open(image_path, 'rb') as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

data = {
    "messages": [{
        "role":
        "user",
        "content": [{
            "type": "text",
            "text": "What is the content of this image?"
        }, {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_data}"
            }
        }]
    }],
    "eos_token_id": [1, 106],
    "pad_token_id":
    0,
    "top_k":
    64,
    "top_p":
    0.95,
    "max_tokens":
    8192,
    "stream":
    False
}

headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}


def send_image_request(model, server):
    data["model"] = model
    url = server.url_for("v1", "chat", "completions")
    response = requests.post(url, headers=headers, json=data)
    print("Status Code:", response.status_code)
    response_json = response.json()
    print("Response:", response_json)
    assert response_json["choices"][0]["message"]["content"], "empty response"
