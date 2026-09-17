CHAT_URI = "/v1/chat/completions"
COMPLETIONS_URI = "/v1/completions"
DEFAULT_MESSAGES = [{"role": "user", "content": "Say hello."}]
JSON_HEADERS = {"Content-Type": "application/json"}


def chat_request_body(**overrides):
    body = {
        "model": "auto",
        "messages": DEFAULT_MESSAGES,
        "stream": False,
    }
    if "max_tokens" not in overrides and "max_completion_tokens" not in overrides:
        body["max_tokens"] = 32
    body.update(overrides)
    if body.get("max_tokens") is None:
        body.pop("max_tokens")
    return body


def send_chat_request(api_client, **overrides):
    return send_request(api_client, CHAT_URI, chat_request_body(**overrides))


def send_completion_request(api_client, **overrides):
    body = {
        "model": "auto",
        "prompt": "Say hello.",
        "max_tokens": 32,
        "stream": False,
    }
    body.update(overrides)
    return send_request(api_client, COMPLETIONS_URI, body)


def send_request(api_client, uri, request_body, headers=None):
    request_headers = dict(JSON_HEADERS)
    if headers:
        request_headers.update(headers)
    return api_client.post(uri, json=request_body, headers=request_headers)
