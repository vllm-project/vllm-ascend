#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/blob/main/tests/conftest.py
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
#

import contextlib
import gc
import os
from typing import (Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar,
                    Union)

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (AutoModelForCausalLM, AutoTokenizer, BatchEncoding,
                          BatchFeature)
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from vllm import LLM, SamplingParams
from vllm.config import TaskOption
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from vllm.inputs import (ExplicitEncoderDecoderPrompt, TextPrompt,
                         TokensPrompt, to_enc_dec_tuple_list)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, identity, is_list_of

from tests.model_utils import (TokensTextLogprobs,
                               TokensTextLogprobsPromptLogprobs)

logger = init_logger(__name__)

_TEST_DIR = os.path.dirname(__file__)
_TEST_PROMPTS = [os.path.join(_TEST_DIR, "prompts", "example.txt")]

_M = TypeVar("_M")
_PromptMultiModalInput = Union[List[_M], List[List[_M]]]

PromptImageInput = _PromptMultiModalInput[Image.Image]
PromptAudioInput = _PromptMultiModalInput[Tuple[np.ndarray, int]]
PromptVideoInput = _PromptMultiModalInput[np.ndarray]

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature, dict)


def _read_prompts(filename: str) -> List[str]:
    with open(filename) as f:
        prompts = f.readlines()
        return prompts


@pytest.fixture
def example_prompts() -> List[str]:
    prompts = []
    for filename in _TEST_PROMPTS:
        prompts += _read_prompts(filename)
    return prompts


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


class HfRunner:

    def wrap_device(self, x: _T, device: Optional[str] = None) -> _T:
        from vllm.platforms import current_platform
        if x is None or isinstance(x, (bool, )):
            return x

        if device is None:
            device = "cpu" if current_platform.is_cpu() else "npu"

        if isinstance(x, dict):
            return {k: self.wrap_device(v, device) for k, v in x.items()}

        if hasattr(x, "device") and x.device.type == device:
            return x

        return x.to(device)

    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        *,
        model_kwargs: Optional[Dict[str, Any]] = None,
        is_sentence_transformer: bool = False,
        is_cross_encoder: bool = False,
        skip_tokenizer_init: bool = False,
        auto_cls: Type[_BaseAutoModelClass] = AutoModelForCausalLM,
        postprocess_inputs: Callable[..., BatchEncoding] = identity,
    ) -> None:
        torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]

        self.model_name = model_name
        model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.model = self.wrap_device(
            auto_cls.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                **model_kwargs,
            ))

        if not skip_tokenizer_init:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )

        # don't put this import at the top level
        # it will call torch.cuda.device_count()
        from transformers import AutoProcessor  # noqa: F401
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        if skip_tokenizer_init:
            self.tokenizer = self.processor.tokenizer

        self.dtype = dtype
        self.postprocess_inputs = postprocess_inputs

    def get_inputs(
        self,
        prompts: List[str],
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> List[BatchEncoding]:
        if images is not None:
            assert len(prompts) == len(images)

        if videos is not None:
            assert len(prompts) == len(videos)

        if audios is not None:
            assert len(prompts) == len(audios)

        all_inputs: List[BatchEncoding] = []
        for i, prompt in enumerate(prompts):
            processor_kwargs: Dict[str, Any] = {
                "text": prompt,
                "return_tensors": "pt",
            }
            if images is not None and (image := images[i]) is not None:
                processor_kwargs["images"] = image
            if videos is not None and (video := videos[i]) is not None:
                processor_kwargs["videos"] = video
            if audios is not None and (audio_tuple := audios[i]) is not None:
                audio, sr = audio_tuple
                processor_kwargs["audio"] = audio
                processor_kwargs["sampling_rate"] = sr

            inputs = self.processor(**processor_kwargs)
            inputs = self.postprocess_inputs(inputs, dtype=self.dtype)

            all_inputs.append(inputs)

        return all_inputs

    def classify(self, prompts: List[str]) -> List[str]:
        # output is final logits
        all_inputs = self.get_inputs(prompts)
        outputs = []
        for inputs in all_inputs:
            output = self.model(**self.wrap_device(inputs))
            logits = output.logits.softmax(dim=-1)[0].tolist()
            outputs.append(logits)

        return outputs

    def generate(
        self,
        prompts: List[str],
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
        **kwargs: Any,
    ) -> List[Tuple[List[List[int]], List[str]]]:
        all_inputs = self.get_inputs(prompts,
                                     images=images,
                                     videos=videos,
                                     audios=audios)

        outputs: List[Tuple[List[List[int]], List[str]]] = []
        for inputs in all_inputs:
            output_ids = self.model.generate(
                **self.wrap_device(inputs, device=self.model.device.type),
                use_cache=True,
                **kwargs,
            )
            output_str = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            output_ids = output_ids.cpu().tolist()
            outputs.append((output_ids, output_str))
        return outputs

    def generate_greedy(
        self,
        prompts: List[str],
        max_tokens: int,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
        **kwargs: Any,
    ) -> List[Tuple[List[int], str]]:
        outputs = self.generate(prompts,
                                do_sample=False,
                                max_new_tokens=max_tokens,
                                images=images,
                                videos=videos,
                                audios=audios,
                                **kwargs)

        return [(output_ids[0], output_str[0])
                for output_ids, output_str in outputs]

    def generate_beam_search(
        self,
        prompts: List[str],
        beam_width: int,
        max_tokens: int,
    ) -> List[Tuple[List[List[int]], List[str]]]:
        outputs = self.generate(prompts,
                                do_sample=False,
                                max_new_tokens=max_tokens,
                                num_beams=beam_width,
                                num_return_sequences=beam_width)
        for i in range(len(outputs)):
            output_ids, output_str = outputs[i]
            for j in range(len(output_ids)):
                output_ids[j] = [
                    x for x in output_ids[j]
                    if x != self.tokenizer.pad_token_id
                ]
            outputs[i] = (output_ids, output_str)
        return outputs

    def generate_greedy_logprobs(
        self,
        prompts: List[str],
        max_tokens: int,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
        **kwargs: Any,
    ) -> List[List[torch.Tensor]]:
        all_inputs = self.get_inputs(prompts,
                                     images=images,
                                     videos=videos,
                                     audios=audios)

        all_logprobs: List[List[torch.Tensor]] = []
        for inputs in all_inputs:
            output = self.model.generate(
                **self.wrap_device(inputs, device=self.model.device.type),
                use_cache=True,
                do_sample=False,
                max_new_tokens=max_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs,
            )
            seq_logprobs = self._hidden_states_to_seq_logprobs(
                output.hidden_states)
            all_logprobs.append(seq_logprobs)
        return all_logprobs

    def _hidden_states_to_seq_logprobs(
        self,
        hidden_states: Tuple[Tuple[torch.Tensor, ...], ...],
    ) -> List[torch.Tensor]:
        output_embeddings = self.model.get_output_embeddings()

        seq_logprobs: List[torch.Tensor] = []
        for _, hidden_state in enumerate(hidden_states):
            last_hidden_states = hidden_state[-1][0]
            logits = torch.matmul(
                last_hidden_states.to(output_embeddings.weight.device),
                output_embeddings.weight.t(),
            )
            if getattr(output_embeddings, "bias", None) is not None:
                logits += output_embeddings.bias.unsqueeze(0)
            logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            seq_logprobs.append(logprobs)

        return seq_logprobs

    def _hidden_states_to_logprobs(
        self,
        hidden_states: Tuple[Tuple[torch.Tensor, ...], ...],
        num_logprobs: int,
    ) -> Tuple[List[Dict[int, float]], int]:
        seq_logprobs = self._hidden_states_to_seq_logprobs(hidden_states)
        output_len = len(hidden_states)

        # convert to dict
        seq_logprobs_lst: List[Dict[int, float]] = []
        for tok_idx, tok_logprobs in enumerate(seq_logprobs):
            # drop prompt logprobs
            if tok_idx == 0:
                tok_logprobs = tok_logprobs[-1, :].reshape(1, -1)
            topk = tok_logprobs.topk(num_logprobs)

            tok_logprobs_dct = {}
            for token_id, logprob in zip(topk.indices[0], topk.values[0]):
                tok_logprobs_dct[token_id.item()] = logprob.item()

            seq_logprobs_lst.append(tok_logprobs_dct)

        return (
            seq_logprobs_lst,
            output_len,
        )

    def generate_greedy_logprobs_limit(
        self,
        prompts: List[str],
        max_tokens: int,
        num_logprobs: int,
        images: Optional[PromptImageInput] = None,
        audios: Optional[PromptAudioInput] = None,
        videos: Optional[PromptVideoInput] = None,
        **kwargs: Any,
    ) -> List[TokensTextLogprobs]:
        all_inputs = self.get_inputs(prompts,
                                     images=images,
                                     videos=videos,
                                     audios=audios)

        all_logprobs: List[List[Dict[int, float]]] = []
        all_output_ids: List[List[int]] = []
        all_output_strs: List[str] = []

        for inputs in all_inputs:
            output = self.model.generate(
                **self.wrap_device(inputs, device=self.model.device.type),
                use_cache=True,
                do_sample=False,
                max_new_tokens=max_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs,
            )

            (
                seq_logprobs_lst,
                output_len,
            ) = self._hidden_states_to_logprobs(output.hidden_states,
                                                num_logprobs)

            all_logprobs.append(seq_logprobs_lst)
            seq_ids = output.sequences[0]
            output_len = len(seq_logprobs_lst)
            output_ids = seq_ids[-output_len:]
            all_output_ids.append(output_ids.tolist())
            all_output_strs.append(self.tokenizer.decode(output_ids))

        outputs = zip(all_output_ids, all_output_strs, all_logprobs)
        return [(output_ids, output_str, output_logprobs)
                for output_ids, output_str, output_logprobs in outputs]

    def generate_encoder_decoder_greedy_logprobs_limit(
        self,
        encoder_decoder_prompts: List[ExplicitEncoderDecoderPrompt[str, str]],
        max_tokens: int,
        num_logprobs: int,
        images: Optional[PromptImageInput] = None,
        **kwargs: Any,
    ) -> List[TokensTextLogprobs]:
        '''
        Greedy logprobs generation for vLLM encoder/decoder models
        '''

        all_logprobs: List[List[Dict[int, float]]] = []
        all_output_ids: List[List[int]] = []
        all_output_strs: List[str] = []

        for i, (encoder_prompt, decoder_prompt) in enumerate(
                to_enc_dec_tuple_list(encoder_decoder_prompts)):
            processor_kwargs: Dict[str, Any] = {
                "text": encoder_prompt,
                "return_tensors": "pt",
            }
            if images is not None and images[i] is not None:
                processor_kwargs["images"] = images[i]

            encoder_input_ids = self.wrap_device(
                self.processor(**processor_kwargs).input_ids,
                device=self.model.device.type,
            )

            if decoder_prompt is None:
                decoder_input_ids = None
            else:
                decoder_input_ids = self.wrap_device(
                    self.tokenizer(decoder_prompt,
                                   return_tensors="pt").input_ids,
                    device=self.model.device.type,
                )

            output = self.model.generate(
                encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                use_cache=True,
                do_sample=False,
                max_new_tokens=max_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs,
            )

            (
                seq_logprobs_lst,
                output_len,
            ) = self._hidden_states_to_logprobs(output.decoder_hidden_states,
                                                num_logprobs)

            all_logprobs.append(seq_logprobs_lst)
            seq_ids = output.sequences[0]
            output_ids = seq_ids[-output_len:]
            all_output_ids.append(output_ids.tolist())
            all_output_strs.append(self.tokenizer.decode(output_ids))

        outputs = zip(all_output_ids, all_output_strs, all_logprobs)
        return [(output_ids, output_str, output_logprobs)
                for output_ids, output_str, output_logprobs in outputs]

    def encode(self, prompts: List[str]) -> List[List[torch.Tensor]]:
        return self.model.encode(prompts)

    def predict(self, prompts: List[List[str]]) -> torch.Tensor:
        return self.model.predict(prompts, convert_to_tensor=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="session")
def hf_runner():
    return HfRunner


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
        dtype: str = "half",
        disable_log_stats: bool = True,
        tensor_parallel_size: int = 1,
        block_size: int = 16,
        enable_chunked_prefill: bool = False,
        swap_space: int = 4,
        enforce_eager: Optional[bool] = False,
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

    def generate_encoder_decoder_w_logprobs(
        self,
        encoder_decoder_prompts: List[ExplicitEncoderDecoderPrompt[str, str]],
        sampling_params: SamplingParams,
    ) -> Union[List[TokensTextLogprobs],
               List[TokensTextLogprobsPromptLogprobs]]:
        '''
        Logprobs generation for vLLM encoder/decoder models
        '''

        assert sampling_params.logprobs is not None
        req_outputs = self.model.generate(encoder_decoder_prompts,
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

    def generate_encoder_decoder_greedy_logprobs(
        self,
        encoder_decoder_prompts: List[ExplicitEncoderDecoderPrompt[str, str]],
        max_tokens: int,
        num_logprobs: int,
        num_prompt_logprobs: Optional[int] = None,
    ) -> Union[List[TokensTextLogprobs],
               List[TokensTextLogprobsPromptLogprobs]]:
        greedy_logprobs_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=num_logprobs,
            prompt_logprobs=(num_prompt_logprobs),
        )
        '''
        Greedy logprobs generation for vLLM encoder/decoder models
        '''

        return self.generate_encoder_decoder_w_logprobs(
            encoder_decoder_prompts, greedy_logprobs_params)

    def generate_beam_search(
        self,
        prompts: Union[List[str], List[List[int]]],
        beam_width: int,
        max_tokens: int,
    ) -> List[Tuple[List[List[int]], List[str]]]:
        if is_list_of(prompts, str, check="all"):
            prompts = [TextPrompt(prompt=prompt) for prompt in prompts]
        else:
            prompts = [
                TokensPrompt(prompt_token_ids=tokens) for tokens in prompts
            ]
        outputs = self.model.beam_search(
            prompts,
            BeamSearchParams(beam_width=beam_width, max_tokens=max_tokens))
        returned_outputs = []
        for output in outputs:
            token_ids = [x.tokens for x in output.sequences]
            texts = [x.text for x in output.sequences]
            returned_outputs.append((token_ids, texts))
        return returned_outputs

    def classify(self, prompts: List[str]) -> List[List[float]]:
        req_outputs = self.model.classify(prompts)
        return [req_output.outputs.probs for req_output in req_outputs]

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

    def score(
        self,
        text_1: Union[str, List[str]],
        text_2: Union[str, List[str]],
    ) -> List[float]:
        req_outputs = self.model.score(text_1, text_2)
        return [req_output.outputs.score for req_output in req_outputs]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="session")
def vllm_runner():
    return VllmRunner
