"""A simple script for generating an output with a normal Llama-3-8B-Instruct model 
or a non-finetuned EELlama model for early-exit. Note: You should fine-tune a model
with early-exit before using the early exits to obtain reasonable results."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers_override.models.llama.configuration_eellama import EeLlamaConfig
from transformers_override.models.llama.modeling_eellama import EeLlamaForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
model = EeLlamaForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=0,

    exit_layers=[7, 15, 23, 31]
)

messages = [
    # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Please explain why a candle is warm."},
]

print("Device: ", model.device)

tokenizer_result = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
).to(model.device)

input_ids = tokenizer_result["input_ids"]
attention_mask = tokenizer_result["attention_mask"]

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.9,
    top_p=0.9,
)
# Will use generation function _sample (https://github.com/huggingface/transformers/blob/v4.57-release/src/transformers/generation/utils.py#L2686)

response = outputs[0][input_ids.shape[-1]:]

print()
print(tokenizer.decode(response, skip_special_tokens=True))
