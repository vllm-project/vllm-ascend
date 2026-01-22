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
import functools
import gc
import json
import logging
import multiprocessing
import os
import shlex
import subprocess
import sys
import time
from typing import Any, Optional, Tuple, TypeVar, Union

import numpy as np
import openai
import pytest
import requests
import torch
from modelscope import snapshot_download  # type: ignore[import-untyped]
from PIL import Image
from requests.exceptions import RequestException
from torch import nn
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BatchEncoding, BatchFeature)
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from vllm import LLM, SamplingParams
from vllm.config.model import (ConvertOption, RunnerOption,
                               _get_and_verify_dtype)
from vllm.inputs import TextPrompt
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_port

from tests.e2e.model_utils import (TokensTextLogprobs,
                                   TokensTextLogprobsPromptLogprobs)
from tests.e2e.nightly.multi_node.scripts.multi_node_config import (
    DisaggregatedPrefillCfg, NodeInfo)
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

_PromptMultiModalInput = Union[list[_M], list[list[_M]]]

PromptImageInput = _PromptMultiModalInput[Image.Image]
PromptAudioInput = _PromptMultiModalInput[Tuple[np.ndarray, int]]
PromptVideoInput = _PromptMultiModalInput[np.ndarray]
logger = logging.getLogger(__name__)

_TEST_DIR = os.path.dirname(__file__)


def _check_npu_memory_worker(target_free_percentage: float, max_wait_seconds: float):
    import torch_npu  # type: ignore
    
    # We can try to clean up memory in this subprocess, though it mostly affects this process.
    # But if there are any lingering contexts in this process (unlikely for a fresh spawn), it helps.
    gc.collect()
    torch.npu.empty_cache()
    
    _, total_npu_memory = torch.npu.mem_get_info()
    start_time = time.time()

    while True:
        free_bytes, _ = torch.npu.mem_get_info()
        if free_bytes / total_npu_memory >= target_free_percentage:
            print(f'check_npu_memory_worker: npu free memory decreased target value.')
            return  # Success

        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            # Print to stderr so it's visible in test logs even if captured
            print(
                f"Timeout: NPU memory free size did not reach "
                f"{target_free_percentage} of total npu memory within {max_wait_seconds} seconds.",
                file=sys.stderr
            )
            sys.exit(1)  # Failure

        print(
            f"Waiting for NPU memory to be free: "
            f"{free_bytes / 1024**3:.2f} GB available, "
            f"Elapsed time: {elapsed:.2f} s."
        )
        # Try to clean up
        gc.collect()
        torch.npu.empty_cache()
        time.sleep(1)


def wait_until_npu_memory_free(target_free_percentage: float = 0.5, max_wait_seconds: float = 50):
    """Decorator to wait until the NPU memory free size is above target_free_percentage.

    Args:
        target_free_percentage (float): Target free memory percentage of total.
        max_wait_seconds (float): Maximum wait time in seconds.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Clean up non-NPU resources in the main process
            cleanup_dist_env_and_memory()
            
            # Use a spawned subprocess to check NPU memory to avoid initializing NPU in the main process
            ctx = multiprocessing.get_context("spawn")
            p = ctx.Process(
                target=_check_npu_memory_worker,
                args=(target_free_percentage, max_wait_seconds)
            )
            p.start()
            p.join()
            
            if p.exitcode != 0:
                raise TimeoutError(
                    f"Timeout: NPU memory free size did not reach "
                    f"{target_free_percentage} of total npu memory within {max_wait_seconds} seconds."
                )

            return func(*args, **kwargs)
        return wrapper
    return decorator


def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    if shutdown_ray:
        import ray  # Lazy import Ray
        ray.shutdown()
    gc.collect()
    
    # Only clean NPU cache if NPU is already initialized/available in this process.
    # This prevents accidental initialization of NPU context in the main process,
    # which would break subsequent forks.
    if hasattr(torch, "npu") and torch.npu.is_initialized():
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()


class RemoteOpenAIServer:
    DUMMY_API_KEY = "token-abc123"  # vLLM's OpenAI server does not need API key

    def _start_server(self, model: str, server_cmd: list[str],
                      env_dict: Optional[dict[str, str]]) -> None:
        """Subclasses override this method to customize server process launch
        """
        env = os.environ.copy()
        # the current process might initialize npu,
        # to be safe, we should use spawn method
        env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        if env_dict is not None:
            env.update(env_dict)
        logger.info(f"Starting server with command: {' '.join(server_cmd)}")
        self.proc: subprocess.Popen = subprocess.Popen(
            server_cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    def __init__(
            self,
            model: str,
            vllm_serve_args: Union[list[str], str],
            *,
            server_host: str = '0.0.0.0',
            server_port: int = 8080,
            env_dict: Optional[dict[str, str]] = None,
            seed: Optional[int] = None,
            auto_port: bool = True,
            nodes_info: Optional[list[NodeInfo]] = None,
            disaggregated_prefill: Optional[DisaggregatedPrefillCfg] = None,
            proxy_port: Optional[int] = None,
            max_wait_seconds: Optional[float] = None,
            override_hf_configs: Optional[dict[str, Any]] = None) -> None:
        if isinstance(vllm_serve_args, str):
            vllm_serve_args = shlex.split(vllm_serve_args)
        else:
            vllm_serve_args = ["vllm", "serve", model, *vllm_serve_args]
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

        self.host = str(server_host)
        self.port = int(server_port)
        # for multi-nodes test
        self.nodes_info = nodes_info
        self.disaggregated_prefill = disaggregated_prefill
        self.cur_index = os.getenv("LWS_WORKER_INDEX", 0)
        self.proxy_port = proxy_port

        self._start_server(model, vllm_serve_args, env_dict)
        max_wait_seconds = max_wait_seconds or 2800
        if self.disaggregated_prefill:
            assert proxy_port is not None, "for disaggregated_prefill, proxy port must be provided"
            self._wait_for_server_pd(timeout=max_wait_seconds)
        else:
            self._wait_for_multiple_servers(
                [(self.host, self.url_for("health"))],
                timeout=max_wait_seconds)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._terminate_server()

    def _poll(self) -> Optional[int]:
        """Subclasses override this method to customize process polling"""
        return self.proc.poll()

    def hang_until_terminated(self, url) -> None:
        """
        Wait until the server process terminates.
        This is for headless mode, where the api server
        process only exists in the leader node.
        """
        logger.info("Hanging until server process terminates...")
        client = requests
        try:
            while True:
                try:
                    resp = client.get(url, timeout=5)
                    if resp.status_code != 200:
                        break
                    time.sleep(5)
                except Exception:
                    break
        finally:
            self._terminate_server()

    def _wait_for_server_pd(self, timeout: float):
        # Wait for all api_server nodes ready
        assert self.nodes_info is not None, "cluster info must be provided"
        proxy_port = self.proxy_port

        def url_health(ip: str, port: int) -> str:
            return f"http://{ip}:{port}/health"

        targets = [(node_info.ip, url_health(node_info.ip, self.port))
                   for node_info in self.nodes_info if not node_info.headless]

        # Wait for proxy ready
        master_node = self.nodes_info[0]
        url_proxy = f"http://{master_node.ip}:{proxy_port}/healthcheck"

        # Wait for master node proxy first
        self._wait_for_multiple_servers([(master_node.ip, url_proxy)],
                                        timeout=timeout)

        # Then wait for all api_server nodes
        self._wait_for_multiple_servers(targets=targets, timeout=timeout)

    def _wait_for_multiple_servers(self,
                                   targets,
                                   timeout: float,
                                   log_interval: float = 30.0):
        """
        targets: List[(node_ip, url)]
        log_interval
        """
        start = time.time()
        client = requests

        ready = {node_ip: False for node_ip, _ in targets}

        last_log_time = 0.0

        while True:
            now = time.time()
            all_ready = True
            should_log = (now - last_log_time) >= log_interval

            for node_ip, url in targets:
                if ready[node_ip]:
                    continue

                try:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        ready[node_ip] = True
                        logger.info(f"[READY] Node {node_ip} is ready.")
                except RequestException:
                    all_ready = False
                    if should_log:
                        logger.debug(f"[WAIT] {url}: connection failed")

                    # check unexpected exit
                    result = self._poll()
                    if result is not None and result != 0:
                        raise RuntimeError(
                            f"Server at {node_ip} exited unexpectedly."
                        ) from None

            if should_log:
                last_log_time = now

            if all_ready:
                break

            if now - start > timeout:
                not_ready_nodes = [n for n, ok in ready.items() if not ok]
                self._terminate_server()
                raise RuntimeError(
                    f"Timeout: these nodes did not become ready: {not_ready_nodes} in time: {timeout}s"
                ) from None

            time.sleep(5)

    @property
    def url_root(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _terminate_server(self) -> None:
        """Subclasses override this method to customize server process termination"""
        self.proc.terminate()
        try:
            self.proc.wait(8)
        except subprocess.TimeoutExpired:
            # force kill if needed
            self.proc.kill()

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
        runner: RunnerOption = "auto",
        convert: ConvertOption = "auto",
        tokenizer_name: Optional[str] = None,
        tokenizer_mode: str = "auto",
        max_model_len: Optional[int] = 1024,
        dtype: str = "auto",
        disable_log_stats: bool = True,
        tensor_parallel_size: int = 1,
        block_size: int = 16,
        enable_chunked_prefill: bool = True,
        swap_space: int = 4,
        enforce_eager: Optional[bool] = False,
        quantization: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.model = LLM(
            model=model_name,
            runner=runner,
            convert=convert,
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
        prompts: Union[list[str], list[torch.Tensor], list[int]],
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> list[TextPrompt]:

        if any(x is not None and len(x) != len(prompts)
               for x in [images, videos, audios]):
            raise ValueError(
                "All non-None multimodal inputs must have the same length as "
                "prompts")

        inputs = []
        for i, prompt in enumerate(prompts):
            multi_modal_data = {}
            if images is not None and (image := images[i]) is not None:
                multi_modal_data["image"] = image
            if videos is not None and (video := videos[i]) is not None:
                multi_modal_data["video"] = video # type: ignore
            if audios is not None and (audio := audios[i]) is not None:
                multi_modal_data["audio"] = audio # type: ignore

            text_prompt_kwargs: dict[str, Any] = {
                "multi_modal_data": multi_modal_data or None
            }
            if isinstance(prompt, str):
                text_prompt_kwargs["prompt"] = prompt
            elif isinstance(prompt, list):
                text_prompt_kwargs["prompt_token_ids"] = prompt
            else:
                text_prompt_kwargs["prompt_embeds"] = prompt

            inputs.append(TextPrompt(**text_prompt_kwargs))

        return inputs

    def generate(
        self,
        prompts: Union[list[str], list[torch.Tensor]],
        sampling_params: SamplingParams,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
        **kwargs: Any,
    ) -> list[tuple[list[list[int]], list[str]]]:
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

        req_outputs = self.model.generate(inputs,
                                          sampling_params=sampling_params,
                                          **kwargs)

        outputs: list[tuple[list[list[int]], list[str]]] = []
        for req_output in req_outputs:
            prompt_str = req_output.prompt
            prompt_ids = req_output.prompt_token_ids
            req_sample_output_ids: list[list[int]] = []
            req_sample_output_strs: list[str] = []
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = list(sample.token_ids)
                req_sample_output_ids.append(prompt_ids + output_ids)
                req_sample_output_strs.append((prompt_str or "") + output_str)
            outputs.append((req_sample_output_ids, req_sample_output_strs))
        return outputs

    @staticmethod
    def _final_steps_generate_w_logprobs(
        req_outputs: list[RequestOutput],
    ) -> list[TokensTextLogprobsPromptLogprobs]:
        outputs: list[TokensTextLogprobsPromptLogprobs] = []
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
        prompts: list[str],
        sampling_params: SamplingParams,
        images: Optional[PromptImageInput] = None,
        audios: Optional[PromptAudioInput] = None,
        videos: Optional[PromptVideoInput] = None,
        **kwargs: Any,
    ) -> Union[list[TokensTextLogprobs],
               list[TokensTextLogprobsPromptLogprobs]]:
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

        req_outputs = self.model.generate(inputs,
                                          sampling_params=sampling_params,
                                          **kwargs)

        toks_str_logsprobs_prompt_logprobs = (
            self._final_steps_generate_w_logprobs(req_outputs))
        # Omit prompt logprobs if not required by sampling params
        return ([x[0:-1] for x in toks_str_logsprobs_prompt_logprobs]
                if sampling_params.prompt_logprobs is None else
                toks_str_logsprobs_prompt_logprobs)

    def generate_greedy(
        self,
        prompts: Union[list[str], list[torch.Tensor]],
        max_tokens: int,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
        **kwargs: Any,
    ) -> list[tuple[list[int], str]]:
        greedy_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = self.generate(prompts,
                                greedy_params,
                                images=images,
                                videos=videos,
                                audios=audios,
                                **kwargs)
        return [(output_ids[0], output_str[0])
                for output_ids, output_str in outputs]

    def generate_greedy_logprobs(
        self,
        prompts: list[str],
        max_tokens: int,
        num_logprobs: Optional[int],
        num_prompt_logprobs: Optional[int] = None,
        images: Optional[PromptImageInput] = None,
        audios: Optional[PromptAudioInput] = None,
        videos: Optional[PromptVideoInput] = None,
        stop_token_ids: Optional[list[int]] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Union[list[TokensTextLogprobs],
               list[TokensTextLogprobsPromptLogprobs]]:
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
                                        videos=videos,
                                        **kwargs)

    def classify(self, prompts: list[str]) -> list[list[float]]:
        req_outputs = self.model.classify(prompts)
        return [req_output.outputs.probs for req_output in req_outputs]

    def embed(self,
              prompts: list[str],
              images: Optional[PromptImageInput] = None,
              videos: Optional[PromptVideoInput] = None,
              audios: Optional[PromptAudioInput] = None,
              *args,
              **kwargs) -> list[list[float]]:
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

        req_outputs = self.model.embed(inputs, *args, **kwargs)
        return [req_output.outputs.embedding for req_output in req_outputs]

    def encode(self, prompts: list[str]) -> list[list[float]]:
        req_outputs = self.model.encode(prompts)
        return [req_output.outputs.data for req_output in req_outputs]

    def reward(self, prompts: list[str]) -> list[list[float]]:
        req_outputs = self.model.reward(prompts)
        return [req_output.outputs.data for req_output in req_outputs]

    def score(
        self,
        text_1: Union[str, list[str]],
        text_2: Union[str, list[str]],
        *args,
        **kwargs,
    ) -> list[float]:
        req_outputs = self.model.score(text_1, text_2, *args, **kwargs)
        return [req_output.outputs.score for req_output in req_outputs]

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

    def get_inputs(
        self,
        prompts: list[str],
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> list[Union[BatchFeature, BatchEncoding]]:
        if images is not None:
            assert len(prompts) == len(images)

        if videos is not None:
            assert len(prompts) == len(videos)

        if audios is not None:
            assert len(prompts) == len(audios)

        all_inputs: list[Union[BatchFeature, BatchEncoding]] = []
        for i, prompt in enumerate(prompts):
            processor_kwargs: dict[str, Any] = {
                "text": prompt,
                "return_tensors": "pt",
            }
            if images is not None and (image := images[i]) is not None:
                processor_kwargs["images"] = image
            if videos is not None and (video := videos[i]) is not None:
                processor_kwargs["videos"] = video
            if audios is not None and (audio_inputs := audios[i]) is not None:
                # HACK - not all processors take sampling_rate; we should
                # clean this up in the future.
                if len(audio_inputs) == 2:
                    audio, sr = audio_inputs
                    processor_kwargs["audio"] = audio
                    processor_kwargs["sampling_rate"] = sr
                else:
                    processor_kwargs["audio"] = audio_inputs

            inputs = self.processor(**processor_kwargs)
            if isinstance(inputs, BatchFeature):
                inputs = inputs.to(dtype=self.dtype)

            all_inputs.append(inputs)

        return all_inputs

    def classify(self, prompts: list[str]) -> list[str]:
        # output is final logits
        all_inputs = self.get_inputs(prompts)
        outputs = []
        problem_type = getattr(self.config, "problem_type", "")

        for inputs in all_inputs:
            output = self.model(**self.wrap_device(inputs))
            if problem_type == "regression":
                logits = output.logits[0].tolist()
            elif problem_type == "multi_label_classification":
                logits = output.logits.sigmoid()[0].tolist()
            else:
                logits = output.logits.softmax(dim=-1)[0].tolist()
            outputs.append(logits)

        return outputs

    def encode(self, prompts: list[str], *args,
               **kwargs) -> list[list[torch.Tensor]]:
        return self.model.encode(prompts, *args, **kwargs)

    def predict(self, prompts: list[list[str]], *args,
                **kwargs) -> torch.Tensor:
        return self.model.predict(prompts,
                                  *args,
                                  convert_to_tensor=True,
                                  **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="session")
def ilama_lora_files():
    return snapshot_download(repo_id="vllm-ascend/ilama-text2sql-spider")


@pytest.fixture(scope="session")
def llama32_lora_files():
    return snapshot_download(repo_id="vllm-ascend/llama32-3b-text2sql-spider")


def qwen_prompt(questions: list[str]) -> list[str]:
    placeholder = "<|image_pad|>"
    return [("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
             f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
             f"{q}<|im_end|>\n<|im_start|>assistant\n") for q in questions]


def hunyuan_prompt(questions: list[str]) -> list[str]:
    placeholder = "<｜hy_place▁holder▁no▁100｜><｜hy_place▁holder▁no▁102｜><｜hy_place▁holder▁no▁101｜>"  # noqa: E501
    return [
        f"<｜hy_begin▁of▁sentence｜>{placeholder}{question}<｜hy_User｜>"
        for question in questions
    ]


PROMPT_CONFIGS = {
    "qwen-vl": {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "prompt_fn": qwen_prompt,
        "mm_processor_kwargs": {
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
    },
    "hunyuan-vl": {
        "model": "Tencent-Hunyuan/HunyuanOCR",
        "prompt_fn": hunyuan_prompt,
        "mm_processor_kwargs": {},
    },
}


@pytest.fixture(params=PROMPT_CONFIGS.keys())
def vl_config(request):
    return PROMPT_CONFIGS[request.param]
