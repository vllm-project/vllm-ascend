import base64
import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

os.environ["VLLM_PLUGINS"] = ""
os.environ["RAYON_NUM_THREADS"] = os.environ.get("RAYON_NUM_THREADS", "2")
os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM",
                                                      "true")
os.environ["RAYON_MIN_CHUNK_SIZE"] = os.environ.get("RAYON_MIN_CHUNK_SIZE",
                                                    "1024")

if os.getenv("PYTHONHASHSEED") == None:
    raise ValueError("PYTHONHASHSEED must be set to use APC")

print(f"Using {os.environ['PYTHONHASHSEED']} for block hash seed")
print(f"Using {os.environ['RAYON_NUM_THREADS']} threads for Rayon parallelism")
print(f"Tokenizers parallelism set to: {os.environ['TOKENIZERS_PARALLELISM']}")
print(f"Rayon minimum chunk size: {os.environ['RAYON_MIN_CHUNK_SIZE']}")

from transformers import AutoTokenizer, PreTrainedTokenizer

# from vllm.transformers_utils.tokenizer import get_tokenizer
# from vllm.utils import sha256
# from vllm.v1.core import kv_cache_utils


@dataclass
class PreprocessResult:
    """Container for preprocessing results matching vLLM format"""
    conversation: List[Dict[str, Any]]
    prompt: str
    input_ids: List[int]
    multi_modal_data: Optional[Any] = None


def load_tokenizer(model_path: str) -> PreTrainedTokenizer:
    """
    Load tokenizer identical to vLLM's implementation
    
    Args:
        model_path: Path to model containing tokenizer files
        
    Returns:
        PreTrainedTokenizer: Loaded tokenizer instance
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True,
                                              local_files_only=True)

    # Set padding token identical to vLLM
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return tokenizer


def parse_tools_and_tool_choice(
    request: Dict[str, Any]
) -> Tuple[Optional[List[Dict]], Optional[Union[str, Dict]]]:
    """
    Exact replica of vLLM's tool parsing logic
    
    Args:
        request: Request dictionary containing tools and tool_choice
        
    Returns:
        Tuple of (tools, tool_choice) as parsed by vLLM
    """
    tools = request.get("tools")
    tool_choice = request.get("tool_choice")

    # vLLM's validation logic
    if tools is not None:
        if not isinstance(tools, list):
            raise ValueError("tools must be a list")

        for tool in tools:
            if not isinstance(tool, dict):
                raise ValueError("each tool must be a dictionary")
            if "type" not in tool:
                raise ValueError("each tool must have a 'type' field")
            if tool["type"] != "function":
                raise ValueError("only function tools are supported")
            if "function" not in tool:
                raise ValueError("each tool must have a 'function' field")

            function = tool["function"]
            if not isinstance(function, dict):
                raise ValueError("function must be a dictionary")
            if "name" not in function:
                raise ValueError("function must have a 'name' field")
            if "parameters" not in function:
                raise ValueError("function must have a 'parameters' field")

    # vLLM's tool_choice validation
    if tool_choice is not None:
        if isinstance(tool_choice, str):
            if tool_choice not in ["none", "auto"]:
                raise ValueError(
                    "tool_choice must be 'none', 'auto', or a specific tool")
        elif isinstance(tool_choice, dict):
            if "type" not in tool_choice or tool_choice["type"] != "function":
                raise ValueError("tool_choice dict must have type 'function'")
            if "function" not in tool_choice:
                raise ValueError("tool_choice must have a 'function' field")
            function = tool_choice["function"]
            if "name" not in function:
                raise ValueError("tool_choice must specify a function name")
        else:
            raise ValueError("tool_choice must be a string or dictionary")

    return tools, tool_choice


def extract_multi_modal_data(
    messages: List[Dict[str,
                        Any]]) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
    """
    Extract multi-modal data from messages (vLLM's mm_data handling)
    
    Args:
        messages: List of messages that may contain multi-modal content
        
    Returns:
        Tuple of (processed_messages, multi_modal_data)
    """
    processed_messages = []
    multi_modal_data = None

    for message in messages:
        # Check if message content is a list (multi-modal format)
        if isinstance(message.get("content"), list):
            # Create a copy of the message
            processed_message = message.copy()
            content_list = processed_message["content"]

            # Extract text parts and multi-modal data
            text_parts = []
            mm_data_parts = []

            for content_item in content_list:
                if isinstance(content_item, dict):
                    if content_item.get("type") == "text":
                        text_parts.append(content_item.get("text", ""))
                    elif content_item.get("type") == "image_url":
                        # Extract image data (vLLM handles this as mm_data)
                        image_url = content_item.get("image_url", {})
                        if isinstance(image_url, dict):
                            url = image_url.get("url", "")
                            if url.startswith("data:image"):
                                # Base64 encoded image
                                try:
                                    # Extract base64 data
                                    header, data = url.split(",", 1)
                                    mm_data_parts.append({
                                        "type":
                                        "image",
                                        "data":
                                        base64.b64decode(data),
                                        "format":
                                        header.split(";")[0].split("/")[1]
                                    })
                                except:
                                    # Fallback: keep as URL
                                    mm_data_parts.append({
                                        "type": "image_url",
                                        "url": url
                                    })
                            else:
                                # Regular URL
                                mm_data_parts.append({
                                    "type": "image_url",
                                    "url": url
                                })

            # Join text parts and update message
            if text_parts:
                processed_message["content"] = " ".join(text_parts)
            else:
                processed_message["content"] = ""

            # Set multi-modal data if any
            if mm_data_parts:
                multi_modal_data = mm_data_parts

            processed_messages.append(processed_message)
        else:
            # Regular text message
            processed_messages.append(message)

    return processed_messages, multi_modal_data


def _apply_chat_template(tokenizer: PreTrainedTokenizer,
                         messages: List[Dict[str, Any]],
                         add_generation_prompt: bool = False,
                         tools: Optional[List[Dict]] = None,
                         tool_choice: Optional[Union[str, Dict]] = None,
                         multi_modal_data: Optional[Any] = None) -> str:
    """
    Exact replica of vLLM's _apply_chat_template function with mm_data support
    
    Args:
        tokenizer: Loaded tokenizer instance
        messages: List of messages in conversation
        add_generation_prompt: Whether to add generation prompt
        tools: List of available tools
        tool_choice: Tool selection strategy
        multi_modal_data: Multi-modal data for the conversation
        
    Returns:
        str: Formatted prompt text
    """
    # vLLM's implementation: try tokenizer.apply_chat_template first
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            # vLLM passes tools parameter if available
            apply_kwargs = {
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": False
            }

            if tools is not None:
                apply_kwargs["tools"] = tools

            # Some tokenizers support multi_modal_data parameter
            if multi_modal_data is not None and hasattr(
                    tokenizer, 'handle_multi_modal_data'):
                apply_kwargs["multi_modal_data"] = multi_modal_data

            prompt = tokenizer.apply_chat_template(**apply_kwargs)
            return prompt
        except (TypeError, AttributeError):
            # Fallback if tokenizer doesn't support these parameters
            pass
        except Exception as e:
            # vLLM logs this but continues with fallback
            print(f"Tokenizer apply_chat_template failed: {e}")

    # vLLM's fallback: manually handle tools and multi-modal data
    processed_messages = messages.copy()

    # Handle tools (same as before)
    if tools is not None:
        tool_info = json.dumps(tools, ensure_ascii=False)
        tool_choice_info = json.dumps(
            tool_choice, ensure_ascii=False) if tool_choice else None

        tool_message = f"Available tools: {tool_info}"
        if tool_choice_info:
            tool_message += f"\nTool choice: {tool_choice_info}"

        # Append to last system message or add new one
        system_message_index = -1
        for i, msg in enumerate(processed_messages):
            if msg.get("role") == "system":
                system_message_index = i
                break

        if system_message_index >= 0:
            processed_messages[system_message_index]["content"] = (
                f"{processed_messages[system_message_index]['content']}\n\n{tool_message}"
            )
        else:
            processed_messages.insert(0, {
                "role": "system",
                "content": tool_message
            })

    # Handle multi-modal data by adding special tokens or markers
    if multi_modal_data is not None:
        # Add multi-modal markers to the prompt
        mm_markers = []
        for mm_item in multi_modal_data:
            if mm_item.get("type") == "image":
                mm_markers.append("[IMAGE]")
            elif mm_item.get("type") == "image_url":
                mm_markers.append(f"[IMAGE_URL:{mm_item.get('url')}]")

        if mm_markers:
            # Find the best place to insert multi-modal markers
            # Usually after the last user message that references images
            last_user_idx = -1
            for i, msg in enumerate(processed_messages):
                if msg.get("role") == "user":
                    last_user_idx = i

            if last_user_idx >= 0:
                processed_messages[last_user_idx]["content"] = (
                    f"{processed_messages[last_user_idx]['content']} {' '.join(mm_markers)}"
                )

    # Final fallback: use tokenizer's chat template without special parameters
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            return tokenizer.apply_chat_template(
                processed_messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False)
        except Exception as e:
            print(f"Fallback apply_chat_template failed: {e}")

    # Ultimate fallback: manual template application
    return _apply_chat_template_fallback(tokenizer, processed_messages,
                                         add_generation_prompt)


def _apply_chat_template_fallback(tokenizer: PreTrainedTokenizer,
                                  messages: List[Dict[str, Any]],
                                  add_generation_prompt: bool) -> str:
    """
    vLLM's ultimate fallback template application
    """
    # Try to get template from config
    chat_template = getattr(tokenizer, 'chat_template', None)
    if chat_template is None:
        try:
            config_path = os.path.join(tokenizer.name_or_path,
                                       'tokenizer_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    chat_template = config.get('chat_template')
        except Exception:
            pass

    # If template exists, try to render it (vLLM uses jinja2)
    if chat_template:
        try:
            from jinja2 import Template
            template = Template(chat_template)
            return template.render(messages=messages,
                                   add_generation_prompt=add_generation_prompt)
        except Exception:
            pass

    # Final manual templates (vLLM's last resort)
    model_name = tokenizer.name_or_path.lower()

    if any(x in model_name for x in ['llama', 'mistral', 'codellama']):
        return _render_llama_template(messages, add_generation_prompt)
    elif any(x in model_name for x in ['chatml', 'openai', 'gpt']):
        return _render_chatml_template(messages, add_generation_prompt)
    else:
        return _render_generic_template(messages, add_generation_prompt)


def _render_llama_template(messages: List[Dict[str, Any]],
                           add_generation_prompt: bool) -> str:
    """vLLM's exact Llama template rendering"""
    prompt = ""
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    # Find system message
    system_message = None
    other_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            system_message = msg["content"]
        else:
            other_messages.append(msg)

    if system_message:
        prompt += f"{B_INST} {B_SYS}{system_message}{E_SYS}"
    else:
        prompt += f"{B_INST} "

    # Process conversation turns
    for i, message in enumerate(other_messages):
        role, content = message["role"], message["content"]

        if role == "user":
            if i > 0:
                prompt += f"{B_INST} {content} {E_INST}"
            else:
                prompt += f"{content} {E_INST}"
        elif role == "assistant":
            prompt += f" {content}</s>"

    if add_generation_prompt and other_messages and other_messages[-1][
            "role"] != "assistant":
        prompt += " "

    return prompt


def _render_chatml_template(messages: List[Dict[str, Any]],
                            add_generation_prompt: bool) -> str:
    """vLLM's exact ChatML template rendering"""
    prompt = ""

    for message in messages:
        role, content = message["role"], message["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    if add_generation_prompt:
        prompt += "<|im_start|>assistant\n"

    return prompt


def _render_generic_template(messages: List[Dict[str, Any]],
                             add_generation_prompt: bool) -> str:
    """vLLM's exact generic template rendering"""
    prompt = ""

    for message in messages:
        role, content = message["role"], message["content"]
        prompt += f"{role}: {content}\n"

    if add_generation_prompt:
        prompt += "assistant: "

    return prompt


def _tokenize_batch_optimized(tokenizer: PreTrainedTokenizer,
                              texts: List[str],
                              multi_modal_data_list: Optional[
                                  List[Any]] = None,
                              **kwargs) -> List[List[int]]:
    """
    Optimized batch tokenization using Rust backend with multi-modal support
    
    Args:
        tokenizer: Loaded tokenizer instance
        texts: List of texts to tokenize
        multi_modal_data_list: List of multi-modal data for each text
        **kwargs: Additional arguments for tokenization
        
    Returns:
        List[List[int]]: Tokenized sequences
    """
    # Use the most efficient batch method available
    if hasattr(tokenizer, '__call__'):
        # Direct call method (most common in modern tokenizers)
        if multi_modal_data_list and hasattr(tokenizer, 'encode_with_images'):
            # Special handling for multi-modal data
            return tokenizer(texts,
                             images=multi_modal_data_list,
                             add_special_tokens=True,
                             return_tensors=None,
                             **kwargs)['input_ids']
        else:
            # Standard text-only encoding
            return tokenizer(texts,
                             add_special_tokens=True,
                             return_tensors=None,
                             **kwargs)['input_ids']
    elif hasattr(tokenizer, 'encode_batch'):
        # encode_batch is often optimized for Rust parallelism
        # Handle multi-modal data if needed
        if multi_modal_data_list and hasattr(tokenizer,
                                             'encode_batch_with_images'):
            # Special handling for multi-modal data
            return tokenizer.encode_batch_with_images(
                texts,
                images=multi_modal_data_list,
                add_special_tokens=True,
                **kwargs)
        else:
            # Standard text-only batch encoding
            return tokenizer.encode_batch(texts,
                                          add_special_tokens=True,
                                          **kwargs)
    elif hasattr(tokenizer, 'batch_encode_plus'):
        # Fallback to batch_encode_plus
        encoding = tokenizer.batch_encode_plus(texts,
                                               padding=False,
                                               truncation=False,
                                               add_special_tokens=True,
                                               return_attention_mask=False,
                                               return_tensors=None,
                                               **kwargs)
        return encoding['input_ids']
    else:
        # Ultimate fallback (shouldn't happen with modern tokenizers)
        results = []
        for i, text in enumerate(texts):
            if multi_modal_data_list and i < len(
                    multi_modal_data_list) and multi_modal_data_list[i]:
                # Handle multi-modal data individually
                if hasattr(tokenizer, 'encode_with_images'):
                    results.append(
                        tokenizer.encode_with_images(
                            text,
                            images=multi_modal_data_list[i],
                            add_special_tokens=True,
                            **kwargs))
                else:
                    # Fallback: just encode text
                    results.append(
                        tokenizer.encode(text,
                                         add_special_tokens=True,
                                         **kwargs))
            else:
                # Standard text encoding
                results.append(
                    tokenizer.encode(text, add_special_tokens=True, **kwargs))
        return results


def _preprocess_chat_batch(
        tokenizer: PreTrainedTokenizer,
        requests: List[Dict[str, Any]]) -> List[PreprocessResult]:
    """
    Batch version of vLLM's _preprocess_chat function with 100% identical functionality
    
    Args:
        tokenizer: Loaded tokenizer instance
        requests: List of request dictionaries
        
    Returns:
        List[PreprocessResult]: Preprocessing results for each request
    """
    results = []
    formatted_prompts = []
    multi_modal_data_list = []

    # Phase 1: Template application (sequential to maintain exact vLLM behavior)
    for request in requests:
        messages = request["messages"]
        add_generation_prompt = request.get("add_generation_prompt", True)
        tools, tool_choice = parse_tools_and_tool_choice(request)

        # Extract multi-modal data from messages (vLLM's mm_data handling)
        processed_messages, multi_modal_data = extract_multi_modal_data(
            messages)

        # Apply chat template exactly as vLLM does
        prompt = _apply_chat_template(tokenizer, processed_messages,
                                      add_generation_prompt, tools,
                                      tool_choice, multi_modal_data)
        formatted_prompts.append(prompt)
        multi_modal_data_list.append(multi_modal_data)

        # Store conversation for result
        results.append(
            PreprocessResult(conversation=processed_messages,
                             prompt=prompt,
                             input_ids=[],
                             multi_modal_data=multi_modal_data))

    # Phase 2: Batch tokenization (where we gain performance)
    try:
        # Use optimized batch tokenization with multi-modal support
        all_input_ids = _tokenize_batch_optimized(tokenizer, formatted_prompts,
                                                  multi_modal_data_list)

        # Assign tokenized results
        for i, input_ids in enumerate(all_input_ids):
            model_name = requests[i].get("model", "")
            if "deepseek" in model_name.lower():
                if input_ids and len(input_ids) > 0:
                    input_ids = input_ids[1:]
            results[i].input_ids = input_ids

    except Exception as e:
        # Fallback to individual tokenization if batch fails
        print(f"Batch tokenization failed, falling back to individual: {e}")
        for i, (prompt, mm_data) in enumerate(
                zip(formatted_prompts, multi_modal_data_list)):
            if mm_data and hasattr(tokenizer, 'encode_with_images'):
                results[i].input_ids = tokenizer.encode_with_images(
                    prompt, images=mm_data, add_special_tokens=True)
            else:
                results[i].input_ids = tokenizer.encode(
                    prompt, add_special_tokens=True)

    return results


tokenizer: PreTrainedTokenizer = None


def init_tokenizer(model_path: str) -> int:
    """
    Initialize tokenizer from model path
    
    Args:
        model_path: Path to model containing tokenizer files
        
    Returns:
        int: 0 for success, -1 for error
    """
    global tokenizer
    try:
        tokenizer = load_tokenizer(model_path)
        return 0
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        return -1


def batch_chat_encode(
        texts: List[str]) -> Tuple[List[str], List[List[int]], List[int]]:
    """
    Public batch encoding function
    
    Args:
        tokenizer: Loaded tokenizer instance
        texts: List of text of json requests to tokenize
        **kwargs: Additional arguments for tokenization
        
    Returns:
        List[List[int]]: Tokenized sequences
    """
    print(texts[0])
    requests = [json.loads(text) for text in texts]
    results = _preprocess_chat_batch(tokenizer, requests)

    # conversations = [result.conversation for result in results]
    prompts = [result.prompt for result in results]
    input_ids = [result.input_ids for result in results]
    multi_modal_size = [
        0 if result.multi_modal_data is None else len(result.multi_modal_data)
        for result in results
    ]

    return prompts, input_ids, multi_modal_size


def sha256(input) -> int:
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    return int.from_bytes(hashlib.sha256(input_bytes).digest(),
                          byteorder="big")


def hash_block_tokens(token_ids: List[int],
                      block_size: int = 128) -> List[int]:
    block_hashes = []
    NONE_HASH = int(sha256(os.environ.get("PYTHONHASHSEED")))

    parent_block_hash_value = NONE_HASH
    # print(f"parent_block_hash_value is {parent_block_hash_value}")
    hash_function = hash

    for start in range(0, len(token_ids), block_size):
        end = start + block_size
        block_token_ids = token_ids[start:end]
        if len(block_token_ids) < block_size:
            break

        block_hash = hash_function(
            (parent_block_hash_value, tuple(block_token_ids), 0))
        block_hashes.append(block_hash)
        parent_block_hash_value = block_hash

    return block_hashes


def batch_chat_encode_bytes(
    texts_bytes: List[bytes]
) -> Tuple[List[bytes], List[List[int]], List[int]]:
    global tokenizer
    if tokenizer is None:
        raise ValueError("Tokenizer not initialized")

    try:
        requests = []
        for text_bytes in texts_bytes:
            try:
                text_bytes.decode('utf-8')
                request = json.loads(text_bytes)
                requests.append(request)
            except (json.JSONDecodeError, UnicodeDecodeError):
                text_str = text_bytes.decode('utf-8')
                request = json.loads(text_str)
                requests.append(request)

        results = _preprocess_chat_batch(tokenizer, requests)

        prompts = []
        for result in results:
            if isinstance(result.prompt, str):
                prompts.append(result.prompt.encode('utf-8'))
            else:
                prompts.append(str(result.prompt).encode('utf-8'))

        input_ids = [result.input_ids for result in results]
        block_hashes = [
            hash_block_tokens(token_ids, block_size=128)
            for token_ids in input_ids
        ]

        multi_modal_sizes = [
            0 if result.multi_modal_data is None else len(
                result.multi_modal_data) for result in results
        ]

        return prompts, input_ids, block_hashes, multi_modal_sizes

    except Exception as e:
        print(f"Error in batch_chat_encode_bytes: {e}")
        raise


# C interface functions
def c_init_tokenizer(model_path: str) -> int:
    """C-friendly wrapper for init_tokenizer"""
    return init_tokenizer(model_path)


def c_batch_chat_encode(texts: List[str]) -> tuple:
    """C-friendly wrapper for batch_chat_encode"""
    return batch_chat_encode(texts)


def c_batch_chat_encode_bytes(texts_bytes: List[bytes]) -> tuple:
    return batch_chat_encode_bytes(texts_bytes)


if __name__ == "__main__":
    try:
        model_path = "/root/nginx-1.26.0/omni_proxy/deepseek"
        msgs = [
            '{"messages":[{"role":"user","content":"赵女士买了一些水果和小食品准备去看望一个朋友，士买了一些水果和小食品准备去看望一个朋友，谁知，这些水果和小食品被他士买了一些水果和小食品准备去看望一个朋友，谁知，这些水果和小食品被他士买了一些水果和小食品准备去看望一个朋友，谁知，这些水果和小食品被他士买了一些水果和小食品准备去看望一个朋友，谁知，这些水果和小食品被他士买了一些水果和小食品准备去看望一个朋友，谁知，这些水果和小食品被他士买了一些水果和小食品准备去看望一个朋友，谁知品被他士买了一些水果和小食品准备去看谁知品被他士买了一些水果和小食品准备去看谁知品被他士买了一些水果和小食品准备去看谁知品被他士买士买了一些水果和小食品准备去看谁知品被他士买士买了一些水果和小食品准备去看谁知品被他士买士买了一些水果和小食品准备去看谁知品被他士买士买了一些水果和小食品准备去看谁知品被他士买了一些水果和小食品准备去看谁知品被他士买了一些水果和小食品准备去看望一个朋友，谁知品被他士买了一些水果和小食品准备去看望一个朋友，谁知，这些水果和小食品被他谁知，这些水果和小食品被他的儿子们偷吃了，二说道：“是老四偷吃的。”老三人说了实话，其他的3个都在撒谎。那么，到底是谁偷吃了这些水果和小食 品？"}],"model":"deepseek","temperature":0,"max_tokens":20, "stream":true,"stream_options": {"include_usage": true,"continuous_usage_stats": true}}',
            '{"messages":[{"role":"user","content":"老四说道：“老二在说谎。”这4个儿子中只有一个人说了实话，其他的3个都在撒谎。那么，到底是谁偷吃了这些水果和小食 品？"}],"model":"deepseek","temperature":0,"max_tokens":20, "stream":true,"stream_options": {"include_usage": true,"continuous_usage_stats": true}}',
        ]

        msgs = [s.encode('utf-8') for s in msgs]
        tokenizer = load_tokenizer(model_path)

        prompts, input_ids, block_hashes, multi_modal_data = batch_chat_encode_bytes(
            msgs)

        print(f"Multi-modal data found: {any(multi_modal_data)}")

        for prompt, ids, block_hash, mm_data in zip(prompts, input_ids,
                                                    block_hashes,
                                                    multi_modal_data):
            # print(f"Conversation: {conv}")
            print(f"Prompt: {prompt}")
            print(f"  Input IDs ({len(ids)}): {ids}")
            print(f"  Block Hashes ({len(ids)}): {block_hash}")
            if mm_data:
                print(f"Multi-modal data: {mm_data}")
            print("-" * 40)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
