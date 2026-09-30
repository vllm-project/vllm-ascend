import requests
from requests.exceptions import RequestException


class HTTPClient:
    def __init__(self, base_url=None, timeout=36000):
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.timeout = timeout

    def get(self, endpoint, params=None, headers=None):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response
        except RequestException as e:
            raise AssertionError(f"GET {url} failed: {str(e)}")

    def post(self, endpoint, json=None, data=None, files=None, headers=None):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = requests.post(url, json=json, data=data, files=files, headers=headers, timeout=self.timeout)
            return response
        except RequestException as e:
            raise AssertionError(f"POST {url} failed: {str(e)}")
