"""A simple script for generating an output with a normal Llama-2-7b-hf model."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers_override.models.llama.configuration_eellama import EeLlamaConfig
from transformers_override.models.llama.modeling_eellama import EeLlamaForCausalLM

# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = EeLlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map=0
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

prompt = "Hey, what is another word for happiness?\n"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print("Model output:")
print(output)

