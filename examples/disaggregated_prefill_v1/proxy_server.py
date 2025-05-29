import os

import aiohttp
from quart import Quart, make_response, request

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)


async def forward_request(url, data):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }
        async with session.post(url=url, json=data,
                                headers=headers) as response:
            if response.status == 200:
                if True:
                    async for chunk_bytes in response.content.iter_chunked(
                            1024):
                        yield chunk_bytes
                else:
                    content = await response.read()
                    yield content


@app.route('/v1/completions', methods=['POST'])
async def handle_request():
    try:
        original_request_data = await request.get_json()

        prefill_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
        prefill_request['max_tokens'] = 1
        prefill_request['kv_transfer_params'] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None
        }

        original_request_data['kv_transfer_params'] = {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None
        }

        # finish prefill
        # Configure the IP and port to your own settings.
        async for _ in forward_request('http://localhost:8100/v1/completions',
                                       prefill_request):
            continue

        # return decode
        # Configure the IP and port to your own settings.
        generator = forward_request('http://192.168.0.139:8200/v1/completions',
                                    # Be sure to change the IP address for your machine
                                    original_request_data)
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


if __name__ == '__main__':
    # Set the host configuration to your own IP.
    app.run(host="0.0.0.0", port=8000)
