# SPDX-License-Identifier: Apache-2.0

import json
import os

import aiohttp
from quart import Quart, make_response, request  # type: ignore

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)

PREFILL_ENDPOINT = "localhost:8100"
DECODE_ENDPOINT = "localhost:8200"


async def forward_request(url, data, headers: dict):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers.update({
            "Authorization":
            f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        })

        async with session.post(url=url, json=data,
                                headers=headers) as response:
            if response.status == 200:
                async for chunk_bytes in response.content.iter_chunked(1024):
                    yield chunk_bytes


@app.route("/v1/completions", methods=["POST"])
async def handle_request():
    try:
        original_request_data = await request.get_json()
        print(f"{request.headers.get('X-Request-ID')=}")

        prefill_request = original_request_data.copy()
        # Change max_tokens = 1 to let it only do prefill
        prefill_request["max_tokens"] = 1

        # Finish prefill
        async for prefill_result in forward_request(
            f"http://{PREFILL_ENDPOINT}/v1/completions",
            prefill_request,
            headers={
                "X-Request-ID": request.headers.get("X-Request-ID"),
            },
        ):
            # Print the prefill result
            print("===== Prefill result =====")
            print(prefill_result.decode("utf-8"))
            print("==========================")
            response = json.loads(prefill_result.decode("utf-8"))
            continue

        # Get the prefill result token, and add it to the decoding request
        decode_request = original_request_data.copy()
        for idx, choices in enumerate(response.get("choices")):
            decode_request["prompt"][idx] += choices.get("text")

        # Return the decoding result
        generator = forward_request(
            f"http://{DECODE_ENDPOINT}/v1/completions",
            decode_request,
            headers={
                "X-Request-ID": request.headers.get("X-Request-ID"),
            },
        )
        response = await make_response(generator)
        response.timeout = None

        return response

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))


if __name__ == "__main__":
    app.run(port=8000)
