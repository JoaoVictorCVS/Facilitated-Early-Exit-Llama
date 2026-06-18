"""Functions for analyzing attention weights and the impact of different attention weight tuning values."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm
from rich.console import Console
from rich.text import Text
from rich.table import Table
import warnings
from itertools import product

from transformers_override.models.llama.modeling_eellama import LayerSubsets
import inference_fine_tuned as inference
import model_evaluation

# PRINT_TO_FILE = False
# results_file_path = "./output_files/prompt_adaption_experiments.md"

data_dir = "../datasets/"
data_file = "cnn-dm_validation_short-shuffled32-52.json"

# job = "analyze_attributions"
job = "grid_search"
# job = "single_generation"
# job = "test"


print("Loading dataset from disk")
# dataset = load_from_disk("../datasets/cnn-dm_validation_very_short-shuffled12")
dataset = load_dataset("json", data_dir=data_dir, data_files={"validation": data_file}, split="validation")

dataset = dataset.rename_column("article", "prompt")
dataset = dataset.rename_column("highlights", "completion")

if inference.model: print("Device: ", inference.model.device)

def write_rich_text_to_file(text, file_path, mode="w", out_types=["ansi", "html"]):
    """Writes rich Text object to file in specified formats (ANSI, HTML).

    Parameters
    ----------
    text : rich.text.Text
        The rich Text object to write.
    file_path : str
        The base file path (without extension) to write to.
    mode : str, optional
        File mode, "w" for overwrite, "a" for append.
    out_types : list of str, optional
        List of output types to write, options are "ansi" and "html".
    """

    if "ansi" in out_types:
        # Export ANSI text for terminal
        console_ansi = Console(record=True, width=100000)
        console_ansi.print(text)
        ansi_text = console_ansi.export_text(styles=True)  # includes ANSI codes

        with open(file_path + ".ansi", mode) as f:
            f.write(ansi_text)

    if "html" in out_types:
        # Export HTML for Word/...
        from io import StringIO
        html_buffer = StringIO() # Used to not print the text to the terminal
        console_html = Console(file=html_buffer, record=True, force_terminal=True, width=100000)
        console_html.print(text)
        html_content = console_html.export_html(inline_styles=True)  # HTML with inline styles

        with open(file_path + ".html", mode) as f:
            f.write(html_content)


def print_saliencies(token_lists, score_lists, descriptors, appendices=None, file_path="plots/saliency", overwrite_file=True, color="red"):
    """
    Creates a text file with ANSI color codes and an html file to visualize token importance.
    
    Parameters
    ----------
    token_lists : list of list of strings
        A list of token (string) sequences to print.
    scores: list of list of float
        The scores to visualize for each of the tokens.
    """
    
    # Color gradient from white (low) to dark red (high)
    def get_color(score):
        # Interpolate from white (255, 255, 255) to dark red (139, 0, 0) (assuming
        # color=="red")
        # -> dark red for high score, white for low score
        r = int(255 - score * (255 - (0 if color!="red" else 139)))
        g = int(255 - score * (255 - (0 if color!="green" else 139)))
        b = int(255 - score * (255 - (0 if color!="blue" else 139)))
        return r, g, b

    # Build the colored text
    text = Text()

    for l in range(len(token_lists)):
        tokens = token_lists[l]
        scores = score_lists[l]

        # Normalize scores to 0-1 range
        min_score, max_score = min(scores), max(scores)
        score_range = (max_score - min_score)
        if score_range==0: score_range = float("inf")   # will make all normalized scores 0.
        normalized = [(s - min_score) / score_range for s in scores]

        text.append(descriptors[l] + "\n")
        
        # Add tokens with colors
        for token, score in zip(tokens, normalized):
            r, g, b = get_color(score)
            text.append(token, style=f"black on rgb({r},{g},{b})")

        text.append("\n")

        if appendices:
            text.append(appendices[l] + "\n\n")

    write_rich_text_to_file(text, file_path, mode="w" if overwrite_file else "a", out_types=["ansi", "html"])
    print("Saliency maps saved as .ansi and .html")


def analyze_attributions(dataset, model, tokenizer, out_file="plots/saliency", eval="normal", comparison_model=None, comparison_tokenizer=None, tokenization_func=None):
    """
    Collects attention weights computed during generation with the given model 
    and creates saliency maps for different analyses on these attention weights.

    Parameters
    ----------
    dataset : datasets.Dataset
        The dataset used as prompts for generation. Each instance should have 
        a "prompt" field.
    model : torch.nn.Module
        The language model to compute the attention attributions with.
    tokenizer : transformers.PreTrainedTokenizer
        Will be used to encode the prompts and decode the model outputs.
    out_file : string
        File path (without file extension) to use to save the saliency maps.
    eval : string
        Decides which analysis to perform. Possible values:
            - "normal" for evaluating normal attributions, 
            - "difference" for analyzing the difference of each layers attributions 
                to the last layers attributions
            - "model" for analyzing the difference of the model's attributions 
                to the `comparison_model`'s attributions
    comparison_model : torch.nn.Module
        Model to compute the reference attention weights for if `eval` is set to
        "model".
    comparison_tokenizer : transformers.PreTrainedTokenizer
        Tokenizer belonging to the `comparison_model`. Should be the same as the
        `model`'s tokenizer.
    tokenization_func : callable
        An optional function to use for tokenizing the input. See inference_fine_tuned.py:generate() 
        for more information.
    """
    if eval=="difference": 
        out_file += "_attribution_differences"
    elif eval=="model": 
        out_file += "_model_differences"
    else:
        if eval!="normal":
            raise Exception("Got an invalid value for argument `eval`")

    if eval=="model" and (tokenizer != comparison_tokenizer):
        warnings.warn("The tokenizers of both models and their special tokens are assumed to be the same, which does not seem to be the case!")

    # Clear possible old files
    open(out_file + ".ansi", "w").close()
    open(out_file + ".html", "w").close()

    with torch.no_grad():
        # For some special tokens, we want to collect attributions across all dataset instances
        special_tokens_to_collect = ["<|begin_of_text|>", "<|start_header_id|>", "user", "<|end_header_id|>", "<|eot_id|>", "assistant"]
        special_tokens_to_collect = [tokenizer.encode(strtoken, add_special_tokens=False) for strtoken in special_tokens_to_collect]
        special_tokens_to_collect = [token[0] for token in special_tokens_to_collect if len(token) == 1]    # if any token given as string is actually not a single token, ignore it.
        special_token_dict = {token: {layer: {"attribution_sum": 0, "num_occurences": 0} for layer in inference.exit_layers} for token in special_tokens_to_collect}

        for i in tqdm(range(len(dataset))):
            response, output_text, stats = inference.generate(model, tokenizer, True, dataset[i]["prompt"], system_prompt="", max_new_tokens=512, do_sample=False, use_cache=False, skip_special_tokens=False, eval_stats=["exit_layer_attributions"], print_output=False, return_input_ids=True, tokenization_func=tokenization_func)
            input_ids, _ = response
            all_models_stats = [stats]

            if eval=="model":
                _, _, stats_comparison = inference.generate(comparison_model, comparison_tokenizer, True, dataset[i]["prompt"], system_prompt="", max_new_tokens=512, do_sample=False, use_cache=False, skip_special_tokens=False, eval_stats=["exit_layer_attributions"], print_output=False, return_input_ids=False, tokenization_func=tokenization_func)
                all_models_stats.append(stats_comparison)
            
            for stats_of_model in all_models_stats:
                for layer, attributions in stats_of_model["exit_layer_attributions"]["attributions"].items():
                    stats_of_model["exit_layer_attributions"][layer] = attributions / stats_of_model["exit_layer_attributions"]["num_tokens_summed"][layer]
                del stats_of_model["exit_layer_attributions"]["attributions"]
                del stats_of_model["exit_layer_attributions"]["num_tokens_summed"]
                del stats_of_model["exit_layer_attributions"]["prompt_length"]

            highest_attribution_strings = []

            print("\n\nMost attributed tokens per exit layer:")
            for layer, attributions in stats["exit_layer_attributions"].items():
                k = 20

                if eval=="normal":
                    sorted_tokens = attributions[0].sort(descending=True)  # tuple of (attributions, indices)
                    decoded_tokens = [tokenizer.decode([input_ids[0, idx.item()].item()]) for idx in sorted_tokens.indices[:k]]
                    print(repr(f"{layer}: {', '.join([decoded_tokens[i] + '(' + str(idx.item()) + ', ' + str(round(sorted_tokens.values[i].item(), 5)) + ')' for i, idx in enumerate(sorted_tokens.indices[:k])])}"))
                    print()
                    highest_attribution_strings.append("Highest attribution: " + repr(', '.join([decoded_tokens[i] + '(' + str(idx.item()) + ', ' + str(round(sorted_tokens.values[i].item(), 5)) + ')' for i, idx in enumerate(sorted_tokens.indices[:k])])))

                if eval=="difference":
                    # Look for the tokens that have high attribution in high layers, but low attribution in low layers
                    if not layer == inference.exit_layers[-1]:
                        sorted_tokens = (stats["exit_layer_attributions"][inference.exit_layers[-1]][0] - attributions[0]).sort(descending=True)  # tuple of (attributions, indices)
                        decoded_tokens = [tokenizer.decode([input_ids[0, idx.item()].item()]) for idx in sorted_tokens.indices[:k]]
                        print(repr(f"{layer}: {', '.join([decoded_tokens[i] + '(' + str(idx.item()) + ', ' + str(round(sorted_tokens.values[i].item(), 5)) + ')' for i, idx in enumerate(sorted_tokens.indices[:k])])}"))
                        print()
                    highest_attribution_strings.append("Highest attribution difference: " + repr(', '.join([decoded_tokens[i] + '(' + str(idx.item()) + ', ' + str(round(sorted_tokens.values[i].item(), 5)) + ')' for i, idx in enumerate(sorted_tokens.indices[:k])])))

                if eval=="model":
                    sorted_tokens = (attributions[0] - stats_comparison["exit_layer_attributions"][layer][0]).sort(descending=True)  # tuple of (attributions, indices)
                    decoded_tokens = [tokenizer.decode([input_ids[0, idx.item()].item()]) for idx in sorted_tokens.indices[:k]]
                    print(repr(f"{layer}: {', '.join([decoded_tokens[i] + '(' + str(idx.item()) + ', ' + str(round(sorted_tokens.values[i].item(), 5)) + ')' for i, idx in enumerate(sorted_tokens.indices[:k])])}"))
                    print()
                    highest_attribution_strings.append("Highest attribution difference: " + repr(', '.join([decoded_tokens[i] + '(' + str(idx.item()) + ', ' + str(round(sorted_tokens.values[i].item(), 5)) + ')' for i, idx in enumerate(sorted_tokens.indices[:k])])))


                # Collect special token attributions
                for token in special_tokens_to_collect:
                    attributions_to_sum = attributions if not eval=="model" else stats_comparison["exit_layer_attributions"][layer]
                    token_indices = (input_ids[0] == token).nonzero().squeeze(-1)
                    token_attribution_sum = attributions_to_sum[0, token_indices].sum().item()
                    special_token_dict[token][layer]["attribution_sum"] += token_attribution_sum
                    special_token_dict[token][layer]["num_occurences"] += len(token_indices)

            if eval=="normal":
                decoded_tokens = [tokenizer.batch_decode(input_ids[0])] * len(stats["exit_layer_attributions"])
                attribution_lists = [attributions[0].log().tolist() for attributions in stats["exit_layer_attributions"].values()]
                descriptors = [f"layer {layer}:" for layer in stats["exit_layer_attributions"].keys()]
                print_saliencies(decoded_tokens, attribution_lists, descriptors, appendices=highest_attribution_strings, file_path=out_file, overwrite_file=False)
            elif eval=="difference":
                decoded_tokens = [tokenizer.batch_decode(input_ids[0])] * len(stats["exit_layer_attributions"])
                attribution_lists = [(stats["exit_layer_attributions"][inference.exit_layers[-1]][0].log() - attributions[0].log()).tolist() for attributions in stats["exit_layer_attributions"].values()]
                descriptors = [f"layer {layer}:" for layer in stats["exit_layer_attributions"].keys()]
                print_saliencies(decoded_tokens, attribution_lists, descriptors, appendices=highest_attribution_strings, file_path=out_file, overwrite_file=False)
            elif eval=="model":
                decoded_tokens = [tokenizer.batch_decode(input_ids[0])] * len(stats["exit_layer_attributions"])
                attribution_lists = [(attributions[0].log() - stats_comparison["exit_layer_attributions"][layer][0].log()).tolist() for layer, attributions in stats["exit_layer_attributions"].items()]
                descriptors = [f"layer {layer}:" for layer in stats["exit_layer_attributions"].keys()]
                print_saliencies(decoded_tokens, attribution_lists, descriptors, appendices=highest_attribution_strings, file_path=out_file, overwrite_file=False, color="green")

        for token in special_token_dict.keys():
            for layer in special_token_dict[token].keys():
                special_token_dict[token][layer]["average_attribution"] = special_token_dict[token][layer]["attribution_sum"] / max(1, special_token_dict[token][layer]["num_occurences"])  # avoid division by zero
        out_string = "Special token attributions across dataset:\n" if not eval=="model" else "Special token attributions of comparison model:\n"
        out_string += "Layer\t" + "\t".join([tokenizer.decode([token]) for token in special_token_dict.keys()]) + "\n"
        for layer in inference.exit_layers:
            out_string += f"{layer}:\t" + "\t".join([str(round(special_token_dict[token][layer]["average_attribution"], 6)) for token in special_token_dict.keys()]) + "\n"

        for file_path in [out_file + ".ansi", out_file + ".html"]:
            with open(file_path, "a") as f:
                f.write("\n")
                f.write(out_string)

        table = Table(title="Special token attributions across dataset" if not eval=="model" else "Special token attributions of comparison model")
        table.add_column("Layer", justify="left")
        for token in special_token_dict.keys():
            table.add_column(tokenizer.decode([token]), justify="left")
        for layer in inference.exit_layers:
            table.add_row(str(layer), *([str(round(special_token_dict[token][layer]["average_attribution"], 6)) for token in special_token_dict.keys()]))

        write_rich_text_to_file(table, out_file, mode="a")

def attn_weight_tuning_grid_search(model_instantiator, dataset, assess_outputs=False, tokenization_func=None, **kwargs):
    """Performs a grid search over given values of model parameters and computes 
    the mean exit layer obtained for the dataset.

    The outputs will be logged in a textfile and the evaluation results will be 
    exported as csv file.

    Parameters
    ----------
    model_instantiator : function
        A function that instantiates a model with the parameters given as keyword
        arguments and returns the model and its tokenizer.
    dataset : datasets.Dataset
        The dataset used to evaluate each model on. Each instance should have 
        a "prompt" field.
    assess_outputs : bool
        Whether to assess each output generated by the model using the llm-as-a-judge
        method in model_evaluation.py. If `True`, the generated outputs and the 
        summary scores will be saved in a file and the overall scores will be included
        in the csv file of the evaluation results.
    tokenization_func : callable
        An optional function to use for tokenizing the input. See inference_fine_tuned.py:generate() 
        for more information.
    **kwargs
        Should contain one or more parameters of the model to instantiate, each 
        with a list of values to use in the grid search.

    """
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='output_files/grid_search_attn_weight_tuning.log', level=logging.INFO, filemode="w")

    all_outputs = []

    # Iterate over all combinations of argument values
    for grid_args in tqdm([dict(zip(kwargs.keys(), values)) for values in list(product(*kwargs.values()))]):
        # grid_args is a dict with a single value for each of the given keys
        logger.info(f"Executing model with parameters {grid_args}")

        model, tokenizer = model_instantiator(**grid_args)
        file_appendix = None if not assess_outputs else str(grid_args).replace(".", "p").replace(":", "")
        all_exit_points, mean_summary_scores = eval_tuned_attn_weights(dataset, model, tokenizer, assess_outputs=assess_outputs, file_appendix=file_appendix, tokenization_func=tokenization_func)

        all_exit_points = torch.tensor(all_exit_points, dtype=torch.float)
        exit_frequencies = {exit_layer: round((all_exit_points == exit_layer).sum().item()/all_exit_points.shape[0], 4) for exit_layer in inference.exit_layers}
        exit_points_mean = round(all_exit_points.mean().item(), 4)
        exit_points_std = round(all_exit_points.std().item(), 4)

        logger.info(f"Exit points mean: {exit_points_mean}, std: {exit_points_std}")
        logger.info(f"Exit frequencies: {str(exit_frequencies)}")
        logger.info("")

        all_outputs.append({"args": str(grid_args), "exit_points_mean": exit_points_mean, "exit_points_std": exit_points_std})
        if assess_outputs:
            all_outputs[-1]["summary_score"] = mean_summary_scores["overall"]
    
    logger.info("Writing csv file.")
    import csv
    with open('output_files/attn_weight_tuning_grid_search_results.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, all_outputs[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(all_outputs)

    logger.info("Finished.")

def eval_tuned_attn_weights(dataset, model, tokenizer, assess_outputs=False, file_appendix="", tokenization_func=None):
    """
    Helper function to generate outputs for all `dataset` instances using the 
    given model and `tokenizer` and to compute exit layers and summary scores, 
    if `assess_outputs` is set to True.
    """

    mean_summary_scores = None

    if not assess_outputs:
        with torch.no_grad():
            all_exit_layers = torch.empty((0,), device=model.device)
            for i in tqdm(range(len(dataset))):
                # response, output_text, stats = inference.generate(model, tokenizer, True, dataset[i]["prompt"], system_prompt="", max_new_tokens=512, do_sample=False, use_cache=False, skip_special_tokens=False, eval_stats=["exit-layer", "exit_layer_attributions"], print_output=True, return_input_ids=True)
                response, output_text, stats = inference.generate(model, tokenizer, True, dataset[i]["prompt"], system_prompt="", max_new_tokens=512, do_sample=False, use_cache=False, skip_special_tokens=False, eval_stats=["exit_layer"], print_output=False, return_input_ids=True, tokenization_func=tokenization_func)
                input_ids, _ = response
                exit_layers = torch.tensor(stats["exit_layer"], dtype=torch.float, device=all_exit_layers.device)
                all_exit_layers = torch.cat([all_exit_layers, exit_layers])
            
    else:
        out_dir = "../datasets/softmax_threshold_0p9/attn_weight_tuning/"
        summary_scores, all_exit_points, mean_exit_points = model_evaluation.eval_generation_with_llm_as_a_judge(dataset, model, tokenizer, reference_dataset=None, system_prompt=None, out_dataset_file=f"{out_dir}{data_file.replace('.json', '_generated_completions_' + file_appendix + '.json')}", tokenization_func=tokenization_func)
        all_exit_layers = torch.tensor(all_exit_points, dtype=torch.float, device=model.device)
        mean_summary_scores = {key: torch.tensor(val).mean().item() for key, val in summary_scores.items()}

    return all_exit_layers, mean_summary_scores


if job=="analyze_attributions":
    COMPARE_MODELS = True
    if COMPARE_MODELS:
        ee_model, ee_tokenizer = inference.model, inference.tokenizer
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        original_model, original_tokenizer = inference.instantiate_model(model_id=model_id, adapter_path=None, is_chat_model=True, untied_heads=False)
        analyze_attributions(dataset, ee_model, ee_tokenizer, out_file="plots/softmax_threshold_0p9/saliency_" + data_file.replace(".json", ""), eval="model", comparison_model=original_model, comparison_tokenizer=original_tokenizer)
    else:
        analyze_attributions(dataset, inference.model, inference.tokenizer, out_file="plots/softmax_threshold_0p9/saliency_" + data_file.replace(".json", ""), eval="difference")
elif job=="grid_search":

    import spacy

    nlp = spacy.load("en_core_web_lg")
    # Spacy features coarse-grained and fine-grained pos tagging. Fine-grained 
    # tags can be obtained with nlp as token.tag and the labels can be found as 
    # follows:
    # pos_tag_list = nlp.get_pipe("tagger").labels    # tuple; desctiptions can be found using glossary.py: https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
    # pos_to_id = {pos: i for i, pos in enumerate(pos_tag_list)}  # map from pos tag string to its index in the list
    # Coarse-grained tags come from the "Universal POS tag set", and can be obtained 
    # with nlp as token.pos, which is an integer id/enum (pos_ for the string).
    # They can be converted to string with nlp.tokenizer.vocab.morphology.strings[<id>]
    pos_to_id = nlp.tokenizer.vocab.morphology.strings

    # We define a custom tokenization function for the inference.generate calls 
    # that computes and stores a pos tag for each token.
    def pos_tagging_tokenization_func(model, tokenizer, is_chat_model, user_input, device, system_prompt=None):
        assert is_chat_model==True
        messages = [
            {"role": "user", "content": user_input},
        ]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        print("Device: ", device)

        rendered_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        doc = nlp(rendered_text)
        spacy_words = ([(token.pos, token.idx, token.idx + len(token)) for token in doc])    # see https://spacy.io/api/token
    
        tokenizer_result = tokenizer(
            rendered_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=False    # Have already been added when applying the chat template
        ).to(device)

        input_ids = tokenizer_result["input_ids"]
        attention_mask = tokenizer_result["attention_mask"]

        all_special_token_ids = set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values())

        # Save the pos tag of each token in the model config
        # For efficiency, we save the pos ids used by spacy 
        # instead of the string representation
        unk_id = nlp.tokenizer.vocab.morphology.strings["X"]
        token_pos_tags = []
        word_idx = 0
        word_start = 0
        word_end = 0
        for index, (token_id, (tok_start, tok_end)) in enumerate(zip(input_ids[0].tolist(), tokenizer_result["offset_mapping"][0].tolist())):
            # Special tokens
            if token_id in all_special_token_ids or tok_start==tok_end:
                token_pos_tags.append(unk_id)
                continue
            
            match_found = False
            while word_start <= tok_end and word_idx < len(spacy_words):
                word_pos, word_start, word_end = spacy_words[word_idx]
                # The huggingface tokenizer includes spaces at the beginning of
                # the tokens while the spacy tokenizer crops spaces. We therefore
                # only check if the final tokens character is within the bounds 
                # of the spacy word.
                if word_start <= tok_end and tok_end <= word_end:
                    token_pos_tags.append(word_pos)
                    match_found = True
                    break
                else:
                    word_idx += 1
            if not match_found:
                # Set result as 'XX' and reset word_idx.
                token_pos_tags.append(unk_id)
                word_idx = 0


        assert len(token_pos_tags) == input_ids.shape[1]     # Must be true!

        model.config.attention_weight_tuning["pos_tag_ids"] = [token_pos_tags]   # single batch list, therefore wrapped into []

        return input_ids, attention_mask


    assess_outputs = True
    tuning_grid = torch.tensor([[1.0, 8.0], [1.0, 9.0], [1.0, 10.0], [1.0, 11.0], [1.0, 12.0]])
    ids_dot = [13, 382, 497, 570, 627, 662, 948, 1131, 1210, 1462, 1861, 1865, 2029, 2266, 2950, 3238, 3343]
    ids_comma = [11, 345, 498, 518, 705, 756, 761, 1145, 1173, 1174, 1282, 1350, 1909, 2637, 2907, 3638, 3755, 4063, 7026, 10113, 10560]
    ids_punct = ids_dot + ids_comma
    user_assistant_tokens = [882, 78191]
    referer_words = [438, 323, 477, 719, 420, 430, 1884, 1521, 902, 369, 3686, 779, 1418]
    tuning_grid = tuning_grid.tolist()
    tuning_grid = [{"tune_tokens": [
        {"token": 128006, "tuning": vals[0], "target_layers": LayerSubsets.NON_EXIT_LAYERS}, 
        {"token": 128007, "tuning": vals[1], "target_layers": LayerSubsets.NON_EXIT_LAYERS}
    ]
        # + [{"token": id_punct, "tuning": vals, "target_layers": LayerSubsets.NON_EXIT_LAYERS} for id_punct in ids_punct]
        # + [{"token": id_punct, "tuning": vals[0], "tune_offset": 1, "target_layers": LayerSubsets.NON_EXIT_LAYERS} for id_punct in ids_punct]
        # + [{"token": id_punct, "tuning": vals[1], "tune_offset": 2, "target_layers": LayerSubsets.NON_EXIT_LAYERS} for id_punct in ids_punct]
        # + [{"token": id_punct, "tuning": vals[2], "tune_offset": 3} for id_punct in ids_punct]
        # + [{"token": token_id, "tuning": vals} for token_id in user_assistant_tokens]
        # + [{"token": token_id, "tuning": vals} for token_id in referer_words]

    } for vals in tuning_grid]
    # tuning_grid = [{"tune_pos": [
    #     {"pos_id": pos_to_id["VERB"], "tuning": vals},
    #     {"pos_id": pos_to_id["AUX"], "tuning": vals}
    #     # {"pos_id": pos_to_id["ADV"], "tuning": vals},
    #     # {"pos_id": pos_to_id["ADP"], "tuning": vals},
    #     # {"pos_id": pos_to_id["DET"], "tuning": vals},
    #     # {"pos_id": pos_to_id["NOUN"], "tuning": vals},
    #     # {"pos_id": pos_to_id["PRON"], "tuning": vals},
    #     # {"pos_id": pos_to_id["PROPN"], "tuning": vals},
    #     # {"pos_id": pos_to_id["CCONJ"], "tuning": vals},
    #     # {"pos_id": pos_to_id["SCONJ"], "tuning": vals},
    #     # {"pos_id": pos_to_id["PUNCT"], "tuning": vals},

    # ]} for vals in tuning_grid]
    # attn_weight_tuning_grid_search(inference.instantiate_model, dataset, attention_weight_tuning=tuning_grid, assess_outputs=assess_outputs, tokenization_func=pos_tagging_tokenization_func)
    attn_weight_tuning_grid_search(inference.instantiate_model, dataset, attention_weight_tuning=tuning_grid, assess_outputs=assess_outputs)

elif job=="single_generation":
    # Info: Results differ slightly when a model has previously been used 
    # during runtime. Reproducible results can be obtained when executing the same
    # test as first inference of separate Python sessions.
    dataset = dataset.select(range(2,4))
    # ids_dot = [13, 382, 497, 570, 627, 662, 948, 1131, 1210, 1462, 1861, 1865, 2029, 2266, 2950, 3238, 3343]
    # tunings = [
    #     # {"tune_tokens": {128006: {"tuning": 1}, 128007: {"tuning": 4}}},
    #     {"tune_tokens": [
    #         {"token": 128006, "tuning": 1.0}, 
    #         # {"token": 128007, "tuning": vals[1], "tune_offset": 0}
    #         # {"token": id_dot, "tuning": 2, "tune_offset": 1} for id_dot in ids_dot
    #     ]}
    # ]
    from debug_cache import debug_cache
    debug_cache["tokenizer"] = inference.tokenizer
    tunings = [{"tune_pos": [
        {"pos_id": pos_to_id["VERB"], "tuning": 3.0},
        {"pos_id": pos_to_id["AUX"], "tuning": 3.0}
    ]}]
    for tuning in tunings:
        inference.instantiate_model(attention_weight_tuning=tuning)
        print()
        print("################# Tuning: ", tuning)
        analyze_attributions(dataset, inference.model, inference.tokenizer, out_file="plots/softmax_threshold_0p9/saliency_test" + data_file.replace(".json", ""), eval="normal", tokenization_func=pos_tagging_tokenization_func)

elif job=="test":
    print("\n\n\nAll tokens with 'and':")
    vocab_size = inference.model.config.vocab_size
    for id in range(vocab_size):
        if 'and' in inference.tokenizer.decode([id]):
            print(f"{id} - '{inference.tokenizer.decode([id])}'")

    import pdb; pdb.set_trace()
