from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from logits_processor_zoo.vllm import GenLengthLogitsProcessor, CiteFromPromptLogitsProcessor, ForceLastPhraseLogitsProcessor


model_name = 'facebook/opt-125m'
tokenizer = AutoTokenizer.from_pretrained(model_name)

logits_processors = [
    CiteFromPromptLogitsProcessor(tokenizer, boost_factor=2.0),
    GenLengthLogitsProcessor(tokenizer, boost_factor=-0.2, p=1),
    ForceLastPhraseLogitsProcessor("\n\nReferences:\n", tokenizer)
]

model = LLM(model_name,
            trust_remote_code=True,
            dtype="half",
            logits_processors=logits_processors,
            enforce_eager=True
        )

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

samplint_params = SamplingParams(n=1,
                temperature=0,
                seed=0,
                skip_special_tokens=True,
                max_tokens=64,)

gen_output = model.generate(
            prompts,
            samplint_params,
        )