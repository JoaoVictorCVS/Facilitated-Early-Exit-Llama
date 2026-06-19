"""
Script for instantiating an EELlama model for inference and generating outputs
with it.

The model is configured by setting the global variables at the beginning of the 
file or by passing arguments to the instantiate_model() function. instantiate_model() 
will set the global variables `model` and `tokenizer`, but also return them. The 
generate() function takes these as arguments to generate an output to a given prompt. 
Executing this file will run an example generation.

"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers_override.models.llama.configuration_eellama import EeLlamaConfig
from transformers_override.models.llama.modeling_eellama import EeLlamaForCausalLM
from peft import PeftModel
import torch

torch.manual_seed(0)

_sentinel_unset = object()

#model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
#adapter_path = "facilitated-early-exit-llama/models/llama-3-eellama-3-8B-instruct/llama-3-eellama-3-8B-instruct-adapter-cnn-dm/checkpoint-4830"
model_id = "facebook/layerskip-llama3.2-1B"
# CNN/DM fine-tuned adapter:
# adapter_path = "/home/joaosa/facilitated-early-exit-llama/models/llama-3-eellama-3p2-1B-layerskip/llama-3-eellama-3p2-1B-layerskip-adapter-cnn-dm/checkpoint-4760"
# SST-2 fine-tuned adapter:
adapter_path = "../models/llama-3-eellama-3p2-1B-layerskip/llama-3-eellama-3p2-1B-layerskip-adapter-sst2-final"
#adapter_path = None
is_chat_model = True
untied_heads = False
# exit_layers=[7, 15, 23, 31]
exit_layers=[i for i in range(1,16)]
# ee_softmax_threshold=0.9    # lowest exit layer for cnn-dm_validation_short_shuffled32-52 s.t. increase of perplexity <= 10% in log space
ee_softmax_threshold=[0.9836151599884033, 0.9710655212402344, 0.961453378200531, 0.9540911316871643, 0.9484521746635437, 0.9441331028938293, 0.9408249855041504, 0.9382911920547485, 0.9363504648208618, 0.9348640441894531, 0.9337255358695984, 0.9328535199165344, 0.9321855902671814, 0.9316740036010742, 0.9312821626663208]    # Found to be best exit point on cnn-dm_validation_short-shuffled52-102.json at a maximum PPL-increase of 10%
ee_entropy_threshold = _sentinel_unset
attention_weight_tuning = _sentinel_unset

# copy_chat_template_model_id = None
copy_chat_template_model_id = "meta-llama/Llama-3.2-1B-Instruct"
# tied_word_embeddings = _sentinel_unset
tied_word_embeddings = False
    
model = None
tokenizer = None

def instantiate_model(model_id=model_id, adapter_path=adapter_path, is_chat_model=is_chat_model, untied_heads=untied_heads, exit_layers=exit_layers, ee_softmax_threshold=ee_softmax_threshold, ee_entropy_threshold=ee_entropy_threshold, attention_weight_tuning=attention_weight_tuning, copy_chat_template_model_id=copy_chat_template_model_id, tied_word_embeddings=tied_word_embeddings):
    given_kwargs = dict(locals())
    model_args = ["untied_heads", "exit_layers", "ee_softmax_threshold", "ee_entropy_threshold", "attention_weight_tuning"]
    kwargs = {
        arg: given_kwargs[arg]
        for arg in model_args
        if given_kwargs[arg] != _sentinel_unset
    }

    global model, tokenizer

    model = EeLlamaForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=0,

        **kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path if adapter_path != None else model_id)
    if copy_chat_template_model_id is not None:
        # Add chat template to non-chat model
        tokenizer.eos_token_id = model.config.eos_token_id
        chat_template = AutoTokenizer.from_pretrained(copy_chat_template_model_id).chat_template
        chat_template = chat_template.replace("<|eot_id|>", tokenizer.decode(model.config.eos_token_id))  # Ensure that the chat template uses the correct eos token id
        tokenizer.chat_template = chat_template
    
    if model.model.config.untied_heads:
        # Initialize all lm heads to the same values as the final layers head
        for h in range(len(model.lm_heads)-1):
            model.lm_heads[h].load_state_dict(model.lm_head.state_dict(), assign=True)
    if tied_word_embeddings != _sentinel_unset:
        model.model.config.tie_word_embeddings = tied_word_embeddings
    if adapter_path != None:
        model = PeftModel.from_pretrained(model, adapter_path)

    # model and tokenizer are set globally, but are also returned for convenience
    return model, tokenizer

def to_subscript(num):
    """Helper function to convert a number to a string of corresponding subscript 
    figures.

    Parameters
    ----------
    num : int
        The number to convert.
    """
    subscript_map = str.maketrans("0123456789-", "₀₁₂₃₄₅₆₇₈₉₋")
    return str(num).translate(subscript_map)


def generate(model, tokenizer, is_chat_model, user_input, system_prompt=None, max_new_tokens=256, do_sample=True, use_cache=True, skip_special_tokens=False, eval_stats=[], print_output=True, tag_exits=False, return_output_string=True, return_input_ids=False, tokenization_func=None, output_format="std"):
    """Call the given model to generate an output to a prompt and print it.

    Parameters
    ----------
    model : torch.nn.Module
        The model to execute.
    tokenizer : transformers.PreTrainedTokenizerBase
        Will be used to encode the prompt and decode the model's output.
    is_chat_model : bool
        Whether this model is trained for chat purposes and the tokenizer has a
        chat template.
    user_input : string
        The text to be inserted into the prompt to the model.
    system_prompt : string, optional
        An optional system prompt to prepend to the prompt using role "system".
    max_new_tokens : int, optional
        See transformers.GenerationMixin.generate.
    do_sample : bool, optional
        See transformers.GenerationMixin.generate.
    use_cache : bool, optional
        See transformers.GenerationMixin.generate.
    skip_special_tokens : bool, optional
        Whether to skip special_tokens in the output.
    eval_stats : list of string, optional
        The list of statistics to make during generation. Possible keys:
            - "exit_layer": The layers at which the model exits. Will result in 
                a list of integers in the stats dict returned.
            - "exit_layer_attributions": The attributions of each prompt token 
                as computed at each exit layer. Will be computed as sum, but returned 
                as a dict of (exit-layer, mean attributions)-pairs in the stats 
                dict returned, where mean-attributions are of shape (batch_size, prompt_length).
    print_output : bool, optional
        Whether to print the generated output to the console.
    tag_exits : bool, optional
        Whether the output should be tagged with indices of the exit layer of 
        each token. Requires "exit_layer" to be in `eval_stats`
    return_input_ids : bool, optional
        Whether to return the input IDs along with the generated output token IDs.
    tokenization_func : callable, optional
        An optional function to use for tokenizing the input. Obtains as arguments
        (model, tokenizer, is_chat_model, user_input, model.device, system_prompt) 
        and should return input_ids and the attention_mask
    output_format : str, optional
        The format of the text string to return. Possible values are "std" for 
        normal text and "latex" for use in LaTeX.

    Returns
    -------
    response : torch.Tensor or tuple of (torch.Tensor, torch.Tensor)
        The generated output as token IDs or if `return_input_ids` is True, a tuple 
        of the input IDs and the generated outpus as token IDs.
    output_string : string
        The generated output as a decoded string.
    stats : dict
        The statistics collected during generation according to `eval_stats`.
    """

    if not tokenization_func:
        if is_chat_model:
            messages = [
                {"role": "user", "content": user_input},
            ]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            print("Device: ", model.device)

            tokenizer_result = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            ).to(model.device)
        else:
            if system_prompt: raise NotImplementedError
            tokenizer_result = tokenizer(
                f"Please summarize this article in 20 words or more:\n\nArticle to summarize:\n{user_input}\nSummary:\n",
                return_tensors="pt",
                add_special_tokens=True
            ).to(model.device)
            # Might also want to add repetition_penalty=1.3 to the generate call

        input_ids = tokenizer_result["input_ids"]
        attention_mask = tokenizer_result["attention_mask"]
    
    else:
        input_ids, attention_mask = tokenization_func(model, tokenizer, is_chat_model, user_input, model.device, system_prompt=system_prompt)

    model.eval()

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    stats = {}

    ctx = torch.autocast(device_type=model.device.type, dtype=model.dtype) \
        if model.dtype in [torch.float16, torch.bfloat16] else nullcontext()
    with ctx:  
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=do_sample,
            temperature=0.6,
            top_p=0.9,
            use_cache=use_cache,
            eval_stats=eval_stats,
            stats=stats
        )
        # Will use generation function _sample (https://github.com/huggingface/transformers/blob/v4.57-release/src/transformers/generation/utils.py#L2686)

    response = outputs[0][input_ids.shape[-1]:]

    output_string = ""
    if return_output_string or print_output:
        if not tag_exits:
            output_string = tokenizer.decode(response, skip_special_tokens=skip_special_tokens)
        else:
            assert "exit_layer" in eval_stats
            output_token_strings = tokenizer.convert_ids_to_tokens(response, skip_special_tokens=False)
            assert len(output_token_strings) == len(stats["exit_layer"])

            def format_layer_text(layer):
                if output_format == "latex":
                    return f"\\tss{{{layer}}}"  # For use with \newcommand{\tss}[1]{\textsubscript{#1}}
                else:
                    return to_subscript(layer)

            if "▁" in "".join(output_token_strings):
                # LLaMA 2 represents spaces with ▁
                tagged_token_strings = [f"{str_token}{format_layer_text(layer)}" for str_token, layer in zip(output_token_strings, stats['exit_layer'])]
                output_string = "".join([" " + token[1:] if token.startswith("▁") else token for token in tagged_token_strings])
            else:
                # LLaMA 3 handels spaces differently
                decoded_tokens = [tokenizer.decode([tid], skip_special_tokens=False) for tid in response]
                tagged_token_strings = [f"{str_token}{format_layer_text(layer)}" for str_token, layer in zip(decoded_tokens, stats['exit_layer'])]
                output_string = tokenizer.convert_tokens_to_string(tagged_token_strings)



    if print_output:
        print()
        print("Model output:")
        print()
        print(output_string)

        if "exit_layer" in eval_stats:
            print()
            print("Exit layers: ")
            print(stats["exit_layer"])
            print("Mean exit layer: ", torch.tensor(stats["exit_layer"], dtype=torch.float).mean().item())

        print()
        print("Number of tokens generated: ", len(response))
    if return_input_ids:
        response = (input_ids, response)

    return response, output_string, stats

SST2_SYSTEM_PROMPT = (
    "Classify the sentiment of the following sentence as 'positive' or 'negative'."
)

def example_run():
    instantiate_model()

    # SST-2 example sentences (label: negative, positive, negative)
    sentences = [
        "hide new secretions from the parental units",
        "it 's a charming and often affecting journey .",
        "a disappointingly superficial exercise .",
    ]

    for sentence in sentences:
        generate(
            model, tokenizer, is_chat_model,
            f"Sentence: {sentence}",
            system_prompt=SST2_SYSTEM_PROMPT,
            max_new_tokens=8,
            eval_stats=["exit_layer"],
            tag_exits=True,
            use_cache=False,
            do_sample=False,
        )
    


if __name__=="__main__":
    print(adapter_path)
    print(os.path.exists(adapter_path))
    example_run()
