def send_request(api_client, uri, request_body):
    """Send request and return response object"""
    return api_client.post(uri, json=request_body, headers={"Content-Type": "application/json"})
