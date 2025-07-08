# Structured Output Guide

## Overview

### What is Structured Output?

LLMs can be unpredictable when you need output in specific formats. Think of asking a model to generate JSON - without guidance, it might produce valid text that breaks JSON specification. **Structured Output (also called Guided Decoding)** enables LLMs to generate outputs that follow a desired structure while preserving the non-deterministic nature of the system.

In simple terms, structured decoding gives LLMs a “template” to follow. Users provide a schema that “influences” the model’s output, ensuring compliance with the desired structure.

![structured decoding](./images/structured_output_1.png)

### Structured Output in vllm-ascend

Currently, vllm-ascend supports **xgrammar** and **guidance** backend for structured output with vllm v1 engine.

XGrammar introduces a new technique that batch constrained decoding via pushdown automaton (PDA). You can think of a PDA as a “collection of FSMs, and each FSM represents a context-free grammar (CFG).” One significant advantage of PDA is its recursive nature, allowing us to execute multiple state transitions. They also include additional optimisation (for those who are interested) to reduce grammar compilation overhead. Besides, you can also find more details about guidance by yourself.

## How to Use Structured Output?

### Online Inference

You can also generate structured outputs using the OpenAI's Completions and Chat API. The following parameters are supported, which must be added as extra parameters:

- `guided_choice`: the output will be exactly one of the choices.
- `guided_regex`: the output will follow the regex pattern.
- `guided_json`: the output will follow the JSON schema.
- `guided_grammar`: the output will follow the context free grammar.

Structured outputs are supported by default in the OpenAI-Compatible Server. You can choose to specify the backend to use by setting the `--guided-decoding-backend` flag to vllm serve. The default backend is `auto`, which will try to choose an appropriate backend based on the details of the request. You may also choose a specific backend, along with some options.

Now let´s see an example for each of the cases, starting with the guided_choice, as it´s the easiest one:

```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="-",
)

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}
    ],
    extra_body={"guided_choice": ["positive", "negative"]},
)
print(completion.choices[0].message.content)
```

The next example shows how to use the guided_regex. The idea is to generate an email address, given a simple regex template:

```python
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Generate an example email address for Alan Turing, who works in Enigma. End in .com and new line. Example result: alan.turing@enigma.com\n",
        }
    ],
    extra_body={"guided_regex": r"\w+@\w+\.com\n", "stop": ["\n"]},
)
print(completion.choices[0].message.content)
```

One of the most relevant features in structured text generation is the option to generate a valid JSON with pre-defined fields and formats. For this we can use the guided_json parameter in two different ways:

- Using a JSON Schema.
- Defining a Pydantic model and then extracting the JSON Schema from it.

The next example shows how to use the guided_json parameter with a Pydantic model:

```python
from pydantic import BaseModel
from enum import Enum

class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"

class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType

json_schema = CarDescription.model_json_schema()

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
        }
    ],
    extra_body={"guided_json": json_schema},
)
print(completion.choices[0].message.content)
```

Finally we have the guided_grammar option, which is probably the most difficult to use, but it´s really powerful. It allows us to define complete languages like SQL queries. It works by using a context free EBNF grammar. As an example, we can use to define a specific format of simplified SQL queries:

```python
simplified_sql_grammar = """
    root ::= select_statement

    select_statement ::= "SELECT " column " from " table " where " condition

    column ::= "col_1 " | "col_2 "

    table ::= "table_1 " | "table_2 "

    condition ::= column "= " number

    number ::= "1 " | "2 "
"""

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Generate an SQL query to show the 'username' and 'email' from the 'users' table.",
        }
    ],
    extra_body={"guided_grammar": simplified_sql_grammar},
)
print(completion.choices[0].message.content)
```

Find more examples [here](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/structured_outputs.py).

### Offline Inference

To use Structured Output, we'll need to configure the guided decoding using the class `GuidedDecodingParams` inside `SamplingParams`. The main available options inside `GuidedDecodingParams` are:

- json
- regex
- choice
- grammar

One example for the usage of the choice parameter is shown below:

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct",
          guided_decoding_backend="xgrammar")

guided_decoding_params = GuidedDecodingParams(choice=["Positive", "Negative"])
sampling_params = SamplingParams(guided_decoding=guided_decoding_params)
outputs = llm.generate(
    prompts="Classify this sentiment: vLLM is wonderful!",
    sampling_params=sampling_params,
)
print(outputs[0].outputs[0].text)
```

Find more examples of other usages [here](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/structured_outputs.py).
