import json
import time
import asyncio
import random
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import uvicorn
from multiprocessing import Process

app = FastAPI(title="OpenAI API 模拟服务")

class Message(BaseModel):
    role: str
    content: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False
    stream_options: Optional[Dict] = None
    ignore_eos: Optional[bool] = False
    kv_transfer_params: Optional[Dict] = None

async def stream_response_generator(max_tokens: int = 1000):
    # 模拟的 token 列表
    tokens = ["这是", "一", "个", "由", "OpenAI", "API", "模拟", "服务", "生成", "的", "流式", "响应", "示例", "在", "真实", "场景", "中", "这", "将", "是", "来自", "大型", "语言", "模型", "的", "实时", "生成", "内容"]
    
    total_tokens = []
    current_token_count = 0
    prompt_len = 4096 - max_tokens

    while current_token_count < max_tokens:
        num_tokens_to_return =  random.choice([1, 2])
        tokens_to_return = random.sample(tokens, min(num_tokens_to_return, len(tokens)))
        
        for token in tokens_to_return:
            if current_token_count < max_tokens:
                total_tokens.append(token)
                current_token_count += len(token)
                usage = {
                    "prompt_tokens": prompt_len, 
                    "completion_tokens": current_token_count,
                    "total_tokens": 3185 + current_token_count
                }

                resp = {
                    #"id": f"chatcmpl-6789",
                    "id": f"chatcmpl-{''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=28))}",
                    "object": "chat.completion.chunk",
                    #"created": 0,
                    "created": int(time.time()),
                    "model": "deepseek",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "logprobs": None,
                            "finish_reason": "stop" if current_token_count >= max_tokens else None
                        }
                    ],
                    "usage": usage
                }

                #print(f"Streaming token: {token}")
                # 按 SSE 格式发送
                yield f"data: {json.dumps(resp, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)

    # 结束标志
    yield "data: [DONE]\n\n"

@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request):
    #print(f"Body:{await request.body()}")
    body = await request.json()
    print("Request Headers:")
    for key, value in request.headers.items():
        print(f"{key}: {value}")

    req = ChatCompletionRequest(**body)

    if not req.stream or not req.max_tokens:
        raise HTTPException(status_code=400, detail="Streaming only")

    max_tokens = req.max_tokens or 1000
    generator = stream_response_generator(max_tokens=max_tokens)
    return StreamingResponse(generator, media_type="text/event-stream")

def run_server(port):
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    ports = [9000, 9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008, 9009, 9010, 9011, 9012, 9013, 9014, 9015]
    processes = []

    for port in ports:
        p = Process(target=run_server, args=(port,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
