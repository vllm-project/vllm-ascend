# Test Guide
This page explains how to write e2e tests and unit tests to verify the implementation of your feature.

## Unit test

There are several principles to follow when writing unit tests:

- The overall file tree should be consistent with vllm_ascend
- The file name should be the original file name with a prefix `test_`
- Use unittest framework, make good use of mock
- The UTs are all running on cpu node, mock the device-related function to host

#### Example
`unittest.TestCase` is recommended when writing UTs, please refer to the existing `test_ascend_config.py` to get how to use it, or find more info about how to use unittest in [unittest doc](https://docs.python.org/3/library/unittest.html#module-unittest)

Full example: [`tests/ut/test_ascend_config.py`](https://github.com/vllm-project/vllm-ascend/blob/main/tests/ut/test_ascend_config.py)

## E2E test

### Setting up offline llm instance

We offer a util class [`VllmRunner`](https://github.com/vllm-project/vllm-ascend/blob/main/tests/conftest.py) for developers to use, which offers the initialization of a llm instance, and the apis, e.g., `generate`, `generate_w_logprobs`, find more details in the following table.

All the e2e offline test should use **`with`** to setup llm instance and do the inference, e.g., `with VllmRunner(...) as vllm_model: ` or `with vllm_runner(...) as vllm_model`. This ensures the resource could be cleaned up clearly when the test is tearing down.

| function name                              | input agrs                                                   | return value                                                 | function instruction                                         |
| ------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `__init__`                                 | `model_name: str`,<br/> `task: TaskOption`, <br/>`tokenizer_name: Optional[str]`,<br/> `tokenizer_mode: str`,<br/> `max_model_len: int`,<br/> `dtype: str`, <br/>`disable_log_stats: bool`, <br/>`tensor_parallel_size: int`, <br/>`block_size: int`, <br/>`enable_chunked_prefill: bool`, <br/>`swap_space: int`,<br/> `enforce_eager: Optional[bool]`, <br/>`quantization: Optional[str]`, <br/>`**kwargs` | `None`                                                       | initialize llm instance                                      |
| `get_inputs`                               | `prompts: List[str]`, <br/>`images: Optional[PromptImageInput]`, <br/>`videos: Optional[PromptVideoInput]`, <br/>`audios: Optional[PromptAudioInput]` | `List[TextPrompt]`                                           | Constructs a prompt list that supports multi-modal (text, image, video, audio) input. |
| `generate`                                 | `prompts: List[str]`, <br/>`sampling_params: SamplingParams`, <br/>`images`, <br/>`videos`,<br/> `audios` | `List[Tuple[List[List[int]], List[str]]]`                    | Generate text and corresponding token IDs, supporting multimodal input. |
| `_final_steps_generate_w_logprobs`         | `req_outputs: List[RequestOutput]`                           | `List[TokensTextLogprobsPromptLogprobs]`                     | Process the generated results and return ouput logprobs.     |
| `generate_w_logprobs`                      | `prompts: List[str]`, <br/>`sampling_params: SamplingParams`, <br/>`images`,<br/> `videos`, <br/>`audios` | `Union[List[TokensTextLogprobs], List[TokensTextLogprobsPromptLogprobs]]` | Generate output with logprobs.                               |
| `generate_encoder_decoder_w_logprobs`      | `encoder_decoder_prompts: List[ExplicitEncoderDecoderPrompt[str, str]]`, <br/>`sampling_params: SamplingParams` | 同上                                                         | The logprobs output for the encoder-decoder architecture model. |
| `generate_greedy`                          | `prompts: List[str]`,<br/> `max_tokens: int`, <br/>`images`, <br/>`videos`, <br/>`audios` | `List[Tuple[List[int], str]]`                                | Generate the first candidate for each prompt using a greedy strategy. |
| `generate_greedy_logprobs`                 | `prompts`,<br/> `max_tokens`,<br/> `num_logprobs`, <br/>`num_prompt_logprobs`, <br/>`images`, `audios`, `videos`, <br/>`stop_token_ids`, `stop` | `Union[List[TokensTextLogprobs], List[TokensTextLogprobsPromptLogprobs]]` | Get the probability distribution of each token under the greedy strategy. |
| `generate_encoder_decoder_greedy_logprobs` | `encoder_decoder_prompts`,<br/> `max_tokens`, <br/>`num_logprobs`, <br/>`num_prompt_logprobs` | `Union[List[TokensTextLogprobs], List[TokensTextLogprobsPromptLogprobs]]` | Greedy + logprobs output of the encoder-decoder architecture model. |
| `generate_beam_search`                     | `prompts: Union[List[str], List[List[int]]]`,<br/> `beam_width: int`, <br/>`max_tokens: int` | `List[Tuple[List[List[int]], List[str]]]`                    | Use the beam search strategy to generate multiple candidate answers. |
| `classify`                                 | `prompts: List[str]`                                         | `List[List[float]]`                                          | Returns the probability of each class for the classification model. |
| `encode`                                   | `prompts: List[str]`, <br/>`images`, <br/>`videos`, <br/>`audios` | `List[List[float]]`                                          | Get the embedding vector of prompt (text/multimodal vector representation) |
| `score`                                    | `text_1: Union[str, List[str]]`, <br/>`text_2: Union[str, List[str]]` | `List[float]`                                                | Returns the similarity score between two texts               |
| `__enter__`                                | /                                                            | `self`                                                       | Make the class support the `with` syntax, returning the instance itself. |
| `__exit__`                                 | `exc_type`, `exc_value`, `traceback`                         | `None`                                                       | Exit and clean up model and distributed environment resources. |

Full Example: [`tests/e2e/singlecard/test_offline_inference.py`](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/singlecard/test_offline_inference.py)

### Setting up online llm server

The util class to setup a llm server is offered in [tests/utils.py](https://github.com/vllm-project/vllm-ascend/blob/main/tests/utils.py).

All the e2e online test should use **`with`** to setup llm server and then send request to it, e.g., `with RemoteOpenAIServer(...) as remote_server: ` . This ensures the resource could be cleaned up clearly when the test is tearing down.

| function name      | input args                                                   | return value         | function instruction                                         |
| ------------------ | ------------------------------------------------------------ | -------------------- | ------------------------------------------------------------ |
| `__init__`         | `model: str`, <br/>`vllm_serve_args: list[str]`, <br/>`env_dict: Optional[dict[str, str]] = None`, <br/>`seed: Optional[int] = 0`, <br/>`auto_port: bool = True`, <br/>`max_wait_seconds: Optional[float] = None` | `None`               | Initialize and start a vLLM OpenAI compatible server process; support automatic port, automatic model download, and setting environment variables. |
| `__enter__`        | /                                                            | `self`               | Make the class support the `with` syntax, returning the instance itself. |
| `__exit__`         | `exc_type, exc_value, traceback`                             | `None`               | Automatically terminate and clean up the vLLM background server process on exit. |
| `_wait_for_server` | `url: str`, <br/>`timeout: float`                            | `None`               | Wait for the OpenAI server to start successfully (health check). If it times out or the process exits abnormally, an error will be reported. |
| `url_root`         | /                                                            | `str`                | Returns the full server address                              |
| `url_for`          | `*parts: str`                                                | `str`                | Concatenate the full URL of the API, for example, `url_for("v1", "chat", "completions")`. |
| `get_client`       | `**kwargs`                                                   | `openai.OpenAI`      | Returns a synchronous OpenAI Python client that connects to the local vLLM server with a default timeout of 600 seconds. |
| `get_async_client` | `**kwargs`                                                   | `openai.AsyncOpenAI` | Returns an asynchronous OpenAI Python client that connects to the local vLLM server with a default timeout of 600 seconds. |

Full Examples: [`tests/e2e/singlecard/test_prompt_embedding.py`](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/singlecard/test_prompt_embedding.py)

### Checking the output results

The util functions to check the correctness of output text or logprobs results of `VllmRunner` are offered in [tests/model_utils.py](https://github.com/vllm-project/vllm-ascend/blob/main/tests/model_utils.py).

| function name          | input agrs                                                   | return value | function instruction                                         |
| ---------------------- | ------------------------------------------------------------ | ------------ | ------------------------------------------------------------ |
| `check_outputs_equal`  | `*`,<br/>`outputs_0_lst: Sequence[TokensText]`,<br/>`outputs_1_lst: Sequence[TokensText]`,<br/>`name_0: str`,<br/>`name_1: str`, | None         | Compare the two sequences generated by different models,     |
| `check_logprobs_close` | `*`,<br/> `outputs_0_lst: Sequence[Union[TokensTextLogprobs, TokensTextLogprobsPromptLogprobs,TextTextLogprobs]]`,<br/> `outputs_1_lst: Sequence[Union[TokensTextLogprobs, TokensTextLogprobsPromptLogprobs, TextTextLogprobs]]`,<br/>`name_0: str`,<br/>`name_1: str`,<br/>`num_outputs_0_skip_tokens: int = 0`,<br/>`warn_on_mismatch: bool = True`,<br/>`always_check_logprobs: bool = False`, | `None`       | which should be equal.Compare the logprobs of two sequences generated by different models, which should be similar but not necessarily equal. |

Full Example: [`tests/e2e/singlecard/test_aclgraph.py`](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/singlecard/test_aclgraph.py)
