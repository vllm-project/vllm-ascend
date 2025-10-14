#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/blob/main/tests/conftest.py
#

import contextlib
import gc
import json
import os
import subprocess
import sys
import time
from typing import Any, List, Optional, Tuple, TypeVar, Union

import httpx
import numpy as np
import openai
import pytest
import requests
import torch
from modelscope import snapshot_download  # type: ignore[import-untyped]
from PIL import Image
from torch import nn
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BatchEncoding, BatchFeature)
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from vllm import LLM, SamplingParams
from vllm.config.model import TaskOption, _get_and_verify_dtype
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.inputs import TextPrompt
from vllm.model_executor.model_loader import get_model_loader
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils import FlexibleArgumentParser, get_open_port

from tests.e2e.model_utils import (TokensTextLogprobs,
                                   TokensTextLogprobsPromptLogprobs)
from vllm_ascend.ascend_config import clear_ascend_config
# TODO: remove this part after the patch merged into vllm, if
# we not explicitly patch here, some of them might be effectiveless
# in pytest scenario
from vllm_ascend.utils import adapt_patch  # noqa E402

adapt_patch(True)
adapt_patch(False)

from vllm.distributed.parallel_state import (  # noqa E402
    destroy_distributed_environment, destroy_model_parallel)

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature, dict)
_M = TypeVar("_M")

_PromptMultiModalInput = Union[List[_M], List[List[_M]]]

PromptImageInput = _PromptMultiModalInput[Image.Image]
PromptAudioInput = _PromptMultiModalInput[Tuple[np.ndarray, int]]
PromptVideoInput = _PromptMultiModalInput[np.ndarray]

_TEST_DIR = os.path.dirname(__file__)


def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    if shutdown_ray:
        import ray  # Lazy import Ray
        ray.shutdown()
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


class RemoteOpenAIServer:
    DUMMY_API_KEY = "token-abc123"  # vLLM's OpenAI server does not need API key

    def _start_server(self, model: str, vllm_serve_args: list[str],
                      env_dict: Optional[dict[str, str]]) -> None:
        """Subclasses override this method to customize server process launch
        """
        env = os.environ.copy()
        # the current process might initialize npu,
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
                 server_host: str = "0.0.0.0",
                 server_port: int = 8080,
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
        args = parser.parse_args([*vllm_serve_args])
        self.uds = args.uds
        if args.uds:
            self.host = None
            self.port = None
        else:
            self.host = str(server_host)
            self.port = int(server_port)

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
        max_wait_seconds = max_wait_seconds or 7200
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

    def hang_until_terminated(self) -> None:
        """
        Wait until the server process terminates.
        This is for headless mode, where the api server
        process only exists in the leader node.
        """
        if self.uds:
            client = httpx.Client(transport=httpx.HTTPTransport(uds=self.uds))
        else:
            client = requests

        try:
            while True:
                try:
                    resp = client.get(self.url_for("health"), timeout=5)
                    if resp.status_code != 200:
                        break
                    time.sleep(5)
                except Exception:
                    break
        finally:
            if isinstance(client, httpx.Client):
                client.close()

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

                time.sleep(1)
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


class VllmRunner:

    def __init__(
        self,
        model_name: str,
        task: TaskOption = "auto",
        tokenizer_name: Optional[str] = None,
        tokenizer_mode: str = "auto",
        # Use smaller max model length, otherwise bigger model cannot run due
        # to kv cache size limit.
        max_model_len: int = 1024,
        dtype: str = "auto",
        disable_log_stats: bool = True,
        tensor_parallel_size: int = 1,
        block_size: int = 16,
        enable_chunked_prefill: bool = False,
        swap_space: int = 4,
        enforce_eager: Optional[bool] = False,
        quantization: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.model = LLM(
            model=model_name,
            task=task,
            tokenizer=tokenizer_name,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=True,
            dtype=dtype,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            disable_log_stats=disable_log_stats,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            block_size=block_size,
            enable_chunked_prefill=enable_chunked_prefill,
            quantization=quantization,
            **kwargs,
        )

    def get_inputs(
        self,
        prompts: List[str],
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> List[TextPrompt]:
        if images is not None:
            assert len(prompts) == len(images)

        if videos is not None:
            assert len(prompts) == len(videos)

        if audios is not None:
            assert len(prompts) == len(audios)

        inputs = [TextPrompt(prompt=prompt) for prompt in prompts]
        if images is not None:
            for i, image in enumerate(images):
                if image is not None:
                    inputs[i]["multi_modal_data"] = {"image": image}

        if videos is not None:
            for i, video in enumerate(videos):
                if video is not None:
                    inputs[i]["multi_modal_data"] = {"video": video}

        if audios is not None:
            for i, audio in enumerate(audios):
                if audio is not None:
                    inputs[i]["multi_modal_data"] = {"audio": audio}

        return inputs

    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> List[Tuple[List[List[int]], List[str]]]:
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

        req_outputs = self.model.generate(inputs,
                                          sampling_params=sampling_params)

        outputs: List[Tuple[List[List[int]], List[str]]] = []
        for req_output in req_outputs:
            prompt_str = req_output.prompt
            prompt_ids = req_output.prompt_token_ids
            req_sample_output_ids: List[List[int]] = []
            req_sample_output_strs: List[str] = []
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = list(sample.token_ids)
                req_sample_output_ids.append(prompt_ids + output_ids)
                req_sample_output_strs.append(prompt_str + output_str)
            outputs.append((req_sample_output_ids, req_sample_output_strs))
        return outputs

    @staticmethod
    def _final_steps_generate_w_logprobs(
        req_outputs: List[RequestOutput],
    ) -> List[TokensTextLogprobsPromptLogprobs]:
        outputs: List[TokensTextLogprobsPromptLogprobs] = []
        for req_output in req_outputs:
            assert len(req_output.outputs) > 0
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = list(sample.token_ids)
                output_logprobs = sample.logprobs
            outputs.append((output_ids, output_str, output_logprobs,
                            req_output.prompt_logprobs))
        return outputs

    def generate_w_logprobs(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        images: Optional[PromptImageInput] = None,
        audios: Optional[PromptAudioInput] = None,
        videos: Optional[PromptVideoInput] = None,
    ) -> Union[List[TokensTextLogprobs],
               List[TokensTextLogprobsPromptLogprobs]]:
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

        req_outputs = self.model.generate(inputs,
                                          sampling_params=sampling_params)

        toks_str_logsprobs_prompt_logprobs = (
            self._final_steps_generate_w_logprobs(req_outputs))
        # Omit prompt logprobs if not required by sampling params
        return ([x[0:-1] for x in toks_str_logsprobs_prompt_logprobs]
                if sampling_params.prompt_logprobs is None else
                toks_str_logsprobs_prompt_logprobs)

    def generate_greedy(
        self,
        prompts: List[str],
        max_tokens: int,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> List[Tuple[List[int], str]]:
        greedy_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = self.generate(prompts,
                                greedy_params,
                                images=images,
                                videos=videos,
                                audios=audios)
        return [(output_ids[0], output_str[0])
                for output_ids, output_str in outputs]

    def generate_greedy_logprobs(
        self,
        prompts: List[str],
        max_tokens: int,
        num_logprobs: int,
        num_prompt_logprobs: Optional[int] = None,
        images: Optional[PromptImageInput] = None,
        audios: Optional[PromptAudioInput] = None,
        videos: Optional[PromptVideoInput] = None,
        stop_token_ids: Optional[List[int]] = None,
        stop: Optional[List[str]] = None,
    ) -> Union[List[TokensTextLogprobs],
               List[TokensTextLogprobsPromptLogprobs]]:
        greedy_logprobs_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=num_logprobs,
            prompt_logprobs=num_prompt_logprobs,
            stop_token_ids=stop_token_ids,
            stop=stop)

        return self.generate_w_logprobs(prompts,
                                        greedy_logprobs_params,
                                        images=images,
                                        audios=audios,
                                        videos=videos)

    def encode(
        self,
        prompts: List[str],
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> List[List[float]]:
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

        req_outputs = self.model.embed(inputs)
        return [req_output.outputs.embedding for req_output in req_outputs]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        clear_ascend_config()
        cleanup_dist_env_and_memory()


class HfRunner:

    def get_default_device(self):

        return ("cpu"
                if current_platform.is_cpu() else current_platform.device_type)

    def wrap_device(self, x: _T, device: Optional[str] = None) -> _T:
        if x is None or isinstance(x, (bool, )):
            return x

        if device is None:
            device = self.device

        if isinstance(x, dict):
            return {k: self.wrap_device(v, device) for k, v in x.items()}

        if hasattr(x, "device") and x.device.type == device:
            return x

        return x.to(device)

    def __init__(
        self,
        model_name: str,
        dtype: str = "auto",
        *,
        model_kwargs: Optional[dict[str, Any]] = None,
        trust_remote_code: bool = True,
        is_sentence_transformer: bool = False,
        is_cross_encoder: bool = False,
        skip_tokenizer_init: bool = False,
        auto_cls: type[_BaseAutoModelClass] = AutoModelForCausalLM,
    ) -> None:
        model_name = maybe_model_redirect(model_name)
        self.model_name = model_name

        self.config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.device = self.get_default_device()
        self.dtype = torch_dtype = _get_and_verify_dtype(
            self.model_name,
            self.config,
            dtype=dtype,
            is_pooling_model=is_sentence_transformer or is_cross_encoder,
        )

        model_kwargs = model_kwargs if model_kwargs is not None else {}
        model_kwargs.setdefault("torch_dtype", torch_dtype)

        if is_sentence_transformer:
            # Lazy init required for AMD CI
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                model_kwargs=model_kwargs,
                trust_remote_code=trust_remote_code,
            )
        elif is_cross_encoder:
            # Lazy init required for AMD CI
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                model_name,
                device=self.device,
                automodel_args=model_kwargs,
                trust_remote_code=trust_remote_code,
            )
        else:
            model = auto_cls.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )

            # in case some unquantized custom models are not in same dtype
            if (getattr(model, "quantization_method", None) is None
                    and any(p.dtype != self.dtype
                            for p in model.parameters())):
                model = model.to(dtype=self.dtype)

            if (getattr(model, "quantization_method", None) != "bitsandbytes"
                    and len({p.device
                             for p in model.parameters()}) < 2):
                model = model.to(device=self.device)

            self.model = model

        if not skip_tokenizer_init:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )

        # don't put this import at the top level
        # it will call torch.cuda.device_count()
        from transformers import AutoProcessor  # noqa: F401
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        if skip_tokenizer_init:
            self.tokenizer = self.processor.tokenizer

    def encode(self, prompts: list[str], *args,
               **kwargs) -> list[list[torch.Tensor]]:
        return self.model.encode(prompts, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="session")
def ilama_lora_files():
    return snapshot_download(repo_id="vllm-ascend/ilama-text2sql-spider")


def qwen_prompt(questions: List[str]) -> List[str]:
    placeholder = "<|image_pad|>"
    return [("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
             f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
             f"{q}<|im_end|>\n<|im_start|>assistant\n") for q in questions]


PROMPT_TEMPLATES = {
    "qwen2.5vl": qwen_prompt,
}


@pytest.fixture(params=list(PROMPT_TEMPLATES.keys()))
def prompt_template(request):
    return PROMPT_TEMPLATES[request.param]
