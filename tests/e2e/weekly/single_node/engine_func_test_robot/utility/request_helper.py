def send_request(api_client, uri, request_body):
    """发送请求，返回响应对象"""
    return api_client.post(uri, json=request_body, headers={"Content-Type": "application/json"})
