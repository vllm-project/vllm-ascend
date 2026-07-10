import requests
import json

url = "http://localhost:8000/v1/chat/completions"

payload = {
    "model": "/workspace/shared_assets/GeekCamp/Infer/Model/zouchangjiang/Qwen3-VL-8B-Instruct",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
                {"type": "text", "text": "What is the text in the illustration?"}
            ]
        }
    ]
}

response = requests.post(url, json=payload)
print("Response Status:", response.status_code)
print("Response Text:", response.text[:2000])