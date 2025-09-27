# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import io

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
import torch
# downloading lora to test lora requests
from openai import BadRequestError
from transformers import AutoConfig

from ..utils import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "facebook/opt-125m"

CONFIG = AutoConfig.from_pretrained(MODEL_NAME)

class RemoteOpenAIServer:
    DUMMY_API_KEY = "token-abc123"  # vLLM's OpenAI server does not need API key

    def _start_server(self, model: str, vllm_serve_args: list[str],
                      env_dict: Optional[dict[str, str]]) -> None:
        """Subclasses override this method to customize server process launch
        """
        env = os.environ.copy()
        # the current process might initialize cuda,
        # to be safe, we should use spawn method
        env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        if env_dict is not None:
            env.update(env_dict)
        self.proc: subprocess.Popen = subprocess.Popen(
            ["vllm", "serve", model, *vllm_serve_args],
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    def __init__(self,
                 model: str,
                 vllm_serve_args: list[str],
                 *,
                 env_dict: Optional[dict[str, str]] = None,
                 seed: Optional[int] = 0,
                 auto_port: bool = True,
                 max_wait_seconds: Optional[float] = None,
                 override_hf_configs: Optional[dict[str, Any]] = None) -> None:
        if auto_port:
            if "-p" in vllm_serve_args or "--port" in vllm_serve_args:
                raise ValueError("You have manually specified the port "
                                 "when `auto_port=True`.")

            # No need for a port if using unix sockets
            if "--uds" not in vllm_serve_args:
                # Don't mutate the input args
                vllm_serve_args = vllm_serve_args + [
                    "--port", str(get_open_port())
                ]
        if seed is not None:
            if "--seed" in vllm_serve_args:
                raise ValueError("You have manually specified the seed "
                                 f"when `seed={seed}`.")

            vllm_serve_args = vllm_serve_args + ["--seed", str(seed)]

        if override_hf_configs is not None:
            vllm_serve_args = vllm_serve_args + [
                "--hf-overrides",
                json.dumps(override_hf_configs)
            ]

        parser = FlexibleArgumentParser(
            description="vLLM's remote OpenAI server.")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        parser = ServeSubcommand().subparser_init(subparsers)
        args = parser.parse_args(["--model", model, *vllm_serve_args])
        self.uds = args.uds
        if args.uds:
            self.host = None
            self.port = None
        else:
            self.host = str(args.host or 'localhost')
            self.port = int(args.port)

        self.show_hidden_metrics = \
            args.show_hidden_metrics_for_version is not None

        # download the model before starting the server to avoid timeout
        is_local = os.path.isdir(model)
        if not is_local:
            engine_args = AsyncEngineArgs.from_cli_args(args)
            model_config = engine_args.create_model_config()
            load_config = engine_args.create_load_config()

            model_loader = get_model_loader(load_config)
            model_loader.download_model(model_config)

        self._start_server(model, vllm_serve_args, env_dict)
        max_wait_seconds = max_wait_seconds or 240
        self._wait_for_server(url=self.url_for("health"),
                              timeout=max_wait_seconds)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.proc.terminate()
        try:
            self.proc.wait(8)
        except subprocess.TimeoutExpired:
            # force kill if needed
            self.proc.kill()

    def _poll(self) -> Optional[int]:
        """Subclasses override this method to customize process polling"""
        return self.proc.poll()

    def _wait_for_server(self, *, url: str, timeout: float):
        # run health check
        start = time.time()
        client = (httpx.Client(transport=httpx.HTTPTransport(
            uds=self.uds)) if self.uds else requests)
        while True:
            try:
                if client.get(url).status_code == 200:
                    break
            except Exception:
                # this exception can only be raised by requests.get,
                # which means the server is not ready yet.
                # the stack trace is not useful, so we suppress it
                # by using `raise from None`.
                result = self._poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from None

                time.sleep(0.5)
                if time.time() - start > timeout:
                    raise RuntimeError(
                        "Server failed to start in time.") from None

    @property
    def url_root(self) -> str:
        return (f"http://{self.uds.split('/')[-1]}"
                if self.uds else f"http://{self.host}:{self.port}")

    def url_for(self, *parts: str) -> str:
        return self.url_root + "/" + "/".join(parts)

    def get_client(self, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return openai.OpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
            **kwargs,
        )

    def get_async_client(self, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return openai.AsyncOpenAI(base_url=self.url_for("v1"),
                                  api_key=self.DUMMY_API_KEY,
                                  max_retries=0,
                                  **kwargs)


@pytest.fixture(scope="module")
def default_server_args() -> list[str]:
    return [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
        # Prompt Embeds server args
        "--enable-prompt-embeds",
    ]


EXAMPLE_PROMPTS = [
    "Hello, my name is",
    "What is an LLM?",
]


def _encode_embeds(embeds: torch.Tensor):
    buffer = io.BytesIO()
    torch.save(embeds, buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@pytest.fixture(scope="module")
def example_prompt_embeds(hf_runner):
    """Create example embeddings and return them as base64 encoded string."""
    with hf_runner(MODEL_NAME) as hf_model:
        example_embeddings = hf_model.get_prompt_embeddings(EXAMPLE_PROMPTS)

    return [_encode_embeds(item) for item in example_embeddings]


@pytest.fixture(scope="module",
                params=["", "--disable-frontend-multiprocessing"])
def server_with_prompt_embeds(default_server_args, request):
    if request.param:
        default_server_args.append(request.param)

    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client_with_prompt_embeds(server_with_prompt_embeds):
    async with server_with_prompt_embeds.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_completions_with_prompt_embeds(
    example_prompt_embeds,
    client_with_prompt_embeds: openai.AsyncOpenAI,
    model_name: str,
):
    encoded_embeds, encoded_embeds2 = example_prompt_embeds

    # Test case: Single prompt embeds input
    completion = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="",  # Add empty prompt as required parameter
        max_tokens=5,
        temperature=0.0,
        extra_body={"prompt_embeds": encoded_embeds})
    assert len(completion.choices[0].text) >= 1
    assert completion.choices[0].prompt_logprobs is None

    # Test case: batch completion with prompt_embeds
    completion = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="",  # Add empty prompt as required parameter
        max_tokens=5,
        temperature=0.0,
        extra_body={"prompt_embeds": [encoded_embeds, encoded_embeds2]})
    assert len(completion.choices) == 2
    assert len(completion.choices[0].text) >= 1
    assert len(completion.choices[1].text) >= 1

    # Test case: streaming with prompt_embeds
    single_completion = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="",  # Add empty prompt as required parameter
        max_tokens=5,
        temperature=0.0,
        extra_body={"prompt_embeds": encoded_embeds})
    single_output = single_completion.choices[0].text

    stream = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="",  # Add empty prompt as required parameter
        max_tokens=5,
        temperature=0.0,
        stream=True,
        extra_body={"prompt_embeds": encoded_embeds})
    chunks = []
    finish_reason_count = 0
    async for chunk in stream:
        chunks.append(chunk.choices[0].text)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
    assert finish_reason_count == 1
    assert chunk.choices[0].finish_reason == "length"
    assert chunk.choices[0].text
    assert "".join(chunks) == single_output

    # Test case: batch streaming with prompt_embeds
    stream = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="",  # Add empty prompt as required parameter
        max_tokens=5,
        temperature=0.0,
        stream=True,
        extra_body={"prompt_embeds": [encoded_embeds, encoded_embeds2]})
    chunks_stream_embeds: list[list[str]] = [[], []]
    finish_reason_count = 0
    async for chunk in stream:
        chunks_stream_embeds[chunk.choices[0].index].append(
            chunk.choices[0].text)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
    assert finish_reason_count == 2
    assert chunk.choices[0].finish_reason == "length"
    assert chunk.choices[0].text
    assert len(chunks_stream_embeds[0]) > 0
    assert len(chunks_stream_embeds[1]) > 0

    # Test case: mixed text and prompt_embeds
    completion_mixed = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="This is a prompt",
        max_tokens=5,
        temperature=0.0,
        extra_body={"prompt_embeds": encoded_embeds})
    assert len(completion.choices) == 2
    completion_text_only = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="This is a prompt",
        max_tokens=5,
        temperature=0.0,
    )
    completion_embeds_only = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="",
        max_tokens=5,
        temperature=0.0,
        extra_body={"prompt_embeds": encoded_embeds})
    # Embeddings responses should be handled first
    assert completion_mixed.choices[0].text == completion_embeds_only.choices[
        0].text
    assert completion_mixed.choices[1].text == completion_text_only.choices[
        0].text


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_completions_errors_with_prompt_embeds(
        client_with_prompt_embeds: openai.AsyncOpenAI, model_name: str):
    # Test error case: invalid prompt_embeds
    with pytest.raises(BadRequestError):
        await client_with_prompt_embeds.completions.create(
            prompt="",
            model=model_name,
            max_tokens=5,
            temperature=0.0,
            extra_body={"prompt_embeds": "invalid_base64"})


@pytest.mark.asyncio
@pytest.mark.parametrize("logprobs_arg", [1, 0])
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_completions_with_logprobs_and_prompt_embeds(
    example_prompt_embeds,
    client_with_prompt_embeds: openai.AsyncOpenAI,
    logprobs_arg: int,
    model_name: str,
):
    encoded_embeds, encoded_embeds2 = example_prompt_embeds

    # Test case: Logprobs using prompt_embeds
    completion = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="",  # Add empty prompt as required parameter
        max_tokens=5,
        temperature=0.0,
        echo=False,
        logprobs=logprobs_arg,
        extra_body={"prompt_embeds": encoded_embeds})

    logprobs = completion.choices[0].logprobs
    assert logprobs is not None
    assert len(logprobs.text_offset) == 5
    assert len(logprobs.token_logprobs) == 5
    assert len(logprobs.top_logprobs) == 5
    for top_logprobs in logprobs.top_logprobs[1:]:
        assert max(logprobs_arg, 1) <= len(top_logprobs) <= logprobs_arg + 1
    assert len(logprobs.tokens) == 5

    # Test case: Log probs with batch completion and prompt_embeds
    completion = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="",  # Add empty prompt as required parameter
        max_tokens=5,
        temperature=0.0,
        echo=False,
        logprobs=logprobs_arg,
        extra_body={"prompt_embeds": [encoded_embeds, encoded_embeds2]})

    assert len(completion.choices) == 2
    for choice in completion.choices:
        logprobs = choice.logprobs
        assert logprobs is not None
        assert len(logprobs.text_offset) == 5
        assert len(logprobs.token_logprobs) == 5
        assert len(logprobs.top_logprobs) == 5
        for top_logprobs in logprobs.top_logprobs[1:]:
            assert max(logprobs_arg,
                       1) <= len(top_logprobs) <= logprobs_arg + 1
        assert len(logprobs.tokens) == 5


@pytest.mark.asyncio
async def test_prompt_logprobs_raises_error(
    example_prompt_embeds,
    client_with_prompt_embeds: openai.AsyncOpenAI,
):
    encoded_embeds, _ = example_prompt_embeds

    with pytest.raises(BadRequestError, match="not compatible"):
        await client_with_prompt_embeds.completions.create(
            model=MODEL_NAME,
            prompt="",
            max_tokens=5,
            temperature=0.0,
            extra_body={
                "prompt_embeds": encoded_embeds,
                "prompt_logprobs": True
            },
        )
