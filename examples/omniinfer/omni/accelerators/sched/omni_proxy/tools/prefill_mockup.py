import asyncio
from multiprocessing import Process

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()


@app.api_route("/v1/chat/completions", methods=["POST", "GET"])
async def generate(request: Request):
    print("Request Headers:")
    for key, value in request.headers.items():
        print(f"{key}: {value}")

    #print(f"Body:{await request.body()}")
    #await asyncio.sleep(5)

    response_data = {
        "node": "node1",
        "block_list": [1, 2, 3, 4, 5],
        "kv_transfer_params": {
            "blocks": [2, 3, 4]
        }
    }
    return JSONResponse(content=response_data)


def run_server(port):
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    ports = [8000, 8001, 8002, 8003]
    processes = []

    for port in ports:
        p = Process(target=run_server, args=(port, ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
