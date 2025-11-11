from typing import Any

import requests

data: dict[str, Any] = {
    "messages": [{
        "role": "user",
        "content": "",
    }],
}


def send_text_request(prompt, model, server, request_args=None):
    data["messages"][0]["content"] = prompt
    data["model"] = model
    url = server.url_for("v1", "chat", "completions")
    if request_args:
        data.update(request_args)
    response = requests.post(url, json=data)
    print("Status Code:", response.status_code)
    response_json = response.json()
    print("Response:", response_json)
    assert response_json["choices"][0]["message"]["content"], "empty response"
