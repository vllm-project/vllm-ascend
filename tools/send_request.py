from typing import Any

import requests


def send_v1_completions(prompt, model, server, request_args=None):
    data: dict[str, Any] = {"model": model, "prompt": prompt}
    if request_args:
        data.update(request_args)
    url = server.url_for("v1", "completions")
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    response_json = response.json()
    print(f"Response json: {response_json}")
    response_text = response_json["choices"][0]["text"]
    print(f"Response: {response_text}")
    assert response_text, "empty response"


def send_v1_chat_completions(prompt, model, server, request_args=None):
    data: dict[str, Any] = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt,
        }],
    }
    if request_args:
        data.update(request_args)
    url = server.url_for("v1", "chat", "completions")
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    response_json = response.json()
    print(f"Response json: {response_json}")
    response_text = response_json["choices"][0]["message"]["content"]
    print(f"Response: {response_text}")
    assert response_text, "empty response"
