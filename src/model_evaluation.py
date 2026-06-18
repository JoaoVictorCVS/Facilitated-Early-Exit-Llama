"""
Functions for evaluating an EELlama model with a variety of metrics, including 
perplexity and LLM-as-a-judge assessment of generated summary quality.

The evaluation to perform can be configured by setting the global variables at 
the beginning of the file. Further configuration of the model and evaluation procedure
can be coded in the if-else decisions regarding the EVAL variable at the end of
this file. 
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse

from transformers import AutoConfig, AutoTokenizer
import torch
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm
from itertools import product
import time

import inference_fine_tuned as inference

from thop import profile

PRINT_TO_FILE = True
results_file_path = "./output_files/prompt_adaption_experiments.md"

# system_prompt = None
system_prompt = "Please summarize the article given by the user in 20 words or more."

data_dir = "../datasets/"
data_file = "cnn-dm_validation_very_short-shuffled12"
out_dir = "../datasets/"

# EVAL="grid_search"
# EVAL="perplexities"
# EVAL="judge"
# EVAL="generation_judged"
# EVAL="timing_analysis"
# EVAL = "Estimated_Computation"
EVAL ="MACs/FLOPs"


CREATE_DATASET = False
if CREATE_DATASET:
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation")
    # Reduce to only short instances of the dataset
    print("Filtering dataset")
    inference.instantiate_model()
    # dataset = dataset.filter(lambda row: len(tokenizer.encode(row['article']+row['highlights']))<400)
    # dataset.save_to_disk("../datasets/cnn-dm_short")
    dataset = dataset.filter(lambda row: len(inference.tokenizer.encode(row['article']+row['highlights']))<200)

    keep_only = 12
    dataset = dataset.shuffle(seed=42).select(range(keep_only))

    dataset.save_to_disk(f"../datasets/cnn-dm_validation_very_short-shuffled{keep_only}")

print("Loading dataset from disk")
if data_dir[-1] != "/": data_dir += "/"
if data_file.endswith(".json"):
    dataset = load_dataset("json", data_dir=data_dir, data_files={"validation": data_file}, split="validation")
else:
    dataset = load_from_disk(f"{data_dir}{data_file}")

def model_parameter_grid_search(model_instantiator, dataset, **kwargs):
    """Performs a grid search over given values of model parameters.

    The outputs will be logged in a textfile and the evaluation results will be 
    exported as csv file.

    Parameters
    ----------
    model_instantiator : function
        A function that instantiates a model with the parameters given as keyword
        arguments and returns the model and its tokenizer.
    dataset : datasets.Dataset
        The dataset used to evaluate each model on. Each instance should have 
        a "prompt" and "completion" field to compute the model perplexity.
    **kwargs
        Should contain one or more parameters of the model to instantiate, each 
        with a list of values to use in the grid search.

    """
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='output_files/grid_search.log', level=logging.INFO, filemode="w")

    all_outputs = []

    # Iterate over all combinations of argument values
    for grid_args in tqdm([dict(zip(kwargs.keys(), values)) for values in list(product(*kwargs.values()))]):
        # grid_args is a dict with a single value for each of the given keys
        logger.info(f"Executing model with parameters {grid_args}")

        model, tokenizer = model_instantiator(**grid_args)
        perplexities, all_exit_points = perplexity(dataset, model, tokenizer)

        ppl_mean = round(torch.tensor(perplexities).mean().item(), 4)
        ppl_std = round(torch.tensor(perplexities).std().item(), 4)

        all_exit_points = torch.tensor(all_exit_points, dtype=torch.float)
        exit_frequencies = {exit_layer: round((all_exit_points == exit_layer).sum().item()/all_exit_points.shape[0], 4) for exit_layer in inference.exit_layers}
        exit_points_mean = round(all_exit_points.mean().item(), 4)
        exit_points_std = round(all_exit_points.std().item(), 4)

        logger.info(f"Perplexities: {perplexities}")
        logger.info(f"Perplexities mean: {ppl_mean}, std: {ppl_std}")
        logger.info(f"Exit points mean: {exit_points_mean}, std: {exit_points_std}")
        logger.info(f"Exit frequencies: {str(exit_frequencies)}")
        logger.info("")

        all_outputs.append({"args": str(grid_args), "ppl_mean": ppl_mean, "ppl_std": ppl_std, "exit_points_mean": exit_points_mean, "exit_points_std": exit_points_std})
    
    logger.info("Writing csv file.")
    import csv
    with open('output_files/grid_search_results.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, all_outputs[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(all_outputs)

    logger.info("Finished.")


def perplexity(dataset, model, tokenizer, system_prompt=None):
    """Computes the perplexity of the given model on the given dataset using teacher 
    forcing.

    Parameters
    ----------
    dataset : datasets.Dataset
        The dataset used to compute the perplexity on. Each instance should have 
        a "prompt" and "completion" field.
    model : torch.nn.Module
        The language model to compute the perplexity for.
    tokenizer : transformers.PreTrainedTokenizer
        Will be used to encode the prompt.
    system_prompt : str, optional
        An optional system prompt to prepend to each prompt in the dataset using 
        role "system".

    Returns
    -------
    perplexities : list of float
        A list of perplexity values, one for each instance in the dataset.
    exit_points : list of int
        A list of exit layers for all tokens in the dataset.
    """
    # Reference code: https://huggingface.co/docs/transformers/perplexity

    use_cache = False

    torch.manual_seed(0)

    # Step 1: Get prompt lengths for each instance
    prompt_lengths = []
    for example in dataset:
        messages_prompt_only = [
            {"role": "user", "content": example["prompt"]},
        ]

        if system_prompt:
            messages_prompt_only.insert(0, {"role": "system", "content": system_prompt})

        prompt_tokens = tokenizer.apply_chat_template(
            messages_prompt_only,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False
        ).to(model.device)        # shape (1, sequence_length)
        prompt_lengths.append(prompt_tokens.shape[1])

    # Step 2: Tokenize all instances (prompt + completion)
    all_input_ids = []
    all_attention_masks = []
    
    for example in dataset:
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]

        if system_prompt:
            messages_prompt_only.insert(0, {"role": "system", "content": system_prompt})

        tokenizer_result = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True
        ).to(model.device)
        all_input_ids.append(tokenizer_result["input_ids"].squeeze(0))  # squeezes from shape (1, sequence_length) to (sequence_length,)
        all_attention_masks.append(tokenizer_result["attention_mask"].squeeze(0))


    # Step 3: Pad all sequences to the same length
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = 128105      # set <|reserved_special_token_100|> as padding token to allow for padding sequenecs. 
    input_ids = torch.nn.utils.rnn.pad_sequence(
        all_input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    ).to(model.device)  # shape (batch_size, longest_sequence_length) where batch_size=len(dataset)
    
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        all_attention_masks,
        batch_first=True,
        padding_value=0 # don't attend to padding tokens
    ).to(model.device)


    # Step 4: Create labels tensor with prompt tokens masked
    labels = input_ids.clone()
    for i, prompt_length in enumerate(prompt_lengths):
        # -100 is the index that will be ignored for loss computation
        labels[i, :prompt_length] = -100  # Mask prompt tokens
    
    # Also mask padding tokens
    labels[attention_mask == 0] = -100


    # # Step 5: Forward pass with the entire batch
    # with torch.no_grad():
    #     outputs = model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         labels=labels
    #     )
        
    #     logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

    # Notice: Use above code if adding support for batch processing. Else use the
    # code below for step 5

    # Step 5: Forward pass on each instance separately, then combine logits
    all_logits = []
    
    all_exit_points = []
    with torch.no_grad():
        ctx = torch.autocast(device_type=model.device.type, dtype=model.dtype) \
            if model.dtype in [torch.float16, torch.bfloat16] else nullcontext()
        with ctx:
            for i in range(len(dataset)):
                stats = {}
                outputs = model(
                    input_ids=input_ids[i:i+1],
                    attention_mask=attention_mask[i:i+1],
                    labels=labels[i:i+1],
                    use_cache = use_cache,
                    eval_stats=["exit_layer"],
                    stats=stats
                )
                # Outputs will be computed causually, i.e. each given all previous tokens
                # (due to the causal_mask) computed in the models forward function
                # with teacher forcing.
                all_logits.append(outputs.logits.squeeze(0))  # Remove batch dimension
                response = outputs.logits.squeeze(0)
                # print("Teacher forced output:")
                # print(tokenizer.decode(response.argmax(-1)))
                # print("Reference:")
                # print(tokenizer.decode([l for l in labels[i] if l!=-100]))
                print(f"Exit layers: {torch.masked_select(torch.tensor(stats['exit_layer'][0]), (labels[i]!=-100).cpu())}")
                all_exit_points = all_exit_points + torch.masked_select(torch.tensor(stats['exit_layer'][0]), (labels[i]!=-100).cpu()).tolist()

    # Combine all logits back into a batch tensor
    logits = torch.stack(all_logits, dim=0)  # Shape: (batch_size, seq_len, vocab_size)


    # Step 6: Compute per-instance perplexity
    perplexities = []
    
    # Shift logits and labels for causal language modeling
    # (model predicts next token, so we compare logits[i] with labels[i+1])
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    
    for i in range(len(dataset)):
        # Get token-level losses for this instance
        token_losses = loss_fct(
            shift_logits[i],  # Shape: (seq_len-1, vocab_size)
            shift_labels[i]   # Shape: (seq_len-1,)
        ) # Shape: (seq_len-1,). Computes -log(exp(shift_logits[i][shift_labels[i]]) / sum_t(exp(shift_logits[t]))), so for each token i the negative log of the softmax probability of the true class p(x[i]). For tokens where the label is -100 the output is 0.
        
        # Only consider non-masked tokens (where label != -100)
        valid_token_mask = (shift_labels[i] != -100)
        valid_losses = token_losses[valid_token_mask]
        
        # Compute average loss and perplexity for this instance
        if valid_losses.numel() > 0:
            avg_loss = valid_losses.mean()  # As for perplexity: 1/t * sum_i( log(p(x[i]|x[<i])) ) where t is the number of tokens (!=-100)
            perplexity = torch.exp(avg_loss)
            perplexities.append(perplexity.item())
        else:
            # Handle edge case where no valid tokens exist
            perplexities.append(float('inf'))
    

    return perplexities, all_exit_points
    # Should be as low as possible.

def eval_generation(dataset, model, tokenizer, reference_dataset=None, system_prompt=None, out_dataset_file=None):
    """Generates completions for each prompt in the dataset and computes the exit 
    layers of the generation process and perplexities on the completions given 
    the prompts from the reference_dataset using teacher forcing.

    Parameters
    ----------
    dataset : datasets.Dataset
        The dataset with the prompts to generate completions for. Each instance 
        should have a "prompt" field and an "id" field.
    model : torch.nn.Module
        The language model to use for generation.
    tokenizer : transformers.PreTrainedTokenizer
        Will be used to encode the prompt and decode the model's output.
    reference_dataset : datasets.Dataset, optional
        An optional dataset with the reference prompts to compute perplexity on 
        the generated completion. Each instance should have a "prompt" field.
    system_prompt : str, optional
        An optional system prompt to prepend to each prompt in the dataset using 
        role "system".
    out_dataset_file : str, optional
        An optional file path to save the dataset with generated completions.

    Returns
    -------
    perplexities : list of float
        If a `reference_dataset` is given, a list of perplexity values, one for 
        each completion generated.
    exit_points : list of int
        A list of exit layers for all tokens generated.
    """
    
    temp_dataset = []   # Create a dataset to store the generated completions and the reference prompts for later comparison
    all_exit_points = []
    mean_exit_points = [] # mean over all tokens of an instance
    # Generate completions for each dataset prompt and track exit layers
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            output_ids, output_text, stats = inference.generate(model, tokenizer, True, dataset[i]["prompt"], system_prompt=system_prompt, max_new_tokens=512, do_sample=False, use_cache=False, skip_special_tokens=False, eval_stats=["exit_layer"], print_output=False)
            temp_dataset.append({"id": dataset[i]["id"], "completion": output_text, "exit_layers": stats["exit_layer"], "prompt": reference_dataset[i]["prompt"] if reference_dataset else ""})
            print(f"Exit layers: {stats['exit_layer']}")
            all_exit_points = all_exit_points + stats['exit_layer']
            mean_exit_points.append(round(torch.tensor(stats['exit_layer'], dtype=torch.float).mean().item(), 2))

    temp_dataset = Dataset.from_list(temp_dataset)
    perplexities = None

    if reference_dataset:
        # Compute perplexity of the model for generating the prompts generated above
        # given the prompt from the reference_dataset
        perplexities, _ = perplexity(temp_dataset, model, tokenizer, system_prompt=system_prompt)

    if out_dataset_file:
        output_dataset = temp_dataset.rename_column("completion", "generated")
        output_dataset = output_dataset.remove_columns(['prompt'])
        output_dataset.to_json(out_dataset_file)


    print("Mean exit points per instance: ", mean_exit_points)
    return perplexities, all_exit_points

def LLM_judge_summary(texts, summaries, label_summaries, intermittent=False):
    """Evaluates the given summaries for the given texts using a single-output, 
    reference-based LLM-as-a-judge.

    Parameters
    ----------
    texts : list of str
        The input texts (news articles) that were summarized.
    summaries : list of str
        The summaries to evaluate.
    label_summaries : list of str
        Reference summaries to give to model for comparison.
    intermittent : bool, optional
        Whether to pause for a few seconds between each prompt to the judge LLM.

    Returns
    -------
    all_scores : dict of list of float
        A dictionary with the scores for each metric. Each value is a list of 
        scores, one for each summary. Metrics included: "coherence", "consistency",
        "fluency", "relevance", "overall", where overall is the mean of the other
        four metrics.
    """
    # 
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval.metrics import GEval
    from deepeval import evaluate
    from deepeval.metrics.g_eval import GEvalTemplate
    from deepeval.metrics.g_eval import Rubric
    from deepeval.dataset import EvaluationDataset, Golden
    from typing import List, Optional, Tuple
    import textwrap

    # A custom template to use G-Eval for text summarization
    # We adapt the default template to mainly use the instructions used in the 
    # the G-Eval paper's repository (https://doi.org/10.48550/arXiv.2303.16634)
    # See licenses/G-Eval_LICENSE.txt (MIT License)
    # and licenses/DeepEval_LICENSE.txt (Apache-2.0 License)
    # Original template: https://github.com/confident-ai/deepeval/blob/main/deepeval/metrics/g_eval/template.py
    class CustomGEvalTemplate(GEvalTemplate):
        @staticmethod
        def generate_evaluation_results(
            evaluation_steps: str,
            test_case_content: str,
            parameters: str,
            rubric: Optional[str] = None,
            score_range: Tuple[int, int] = (0, 10),
            _additional_context: Optional[str] = None,
        ):
            rubric_text = ""
            dependencies = "evaluation steps"
            score_explanation = f"with {score_range[1]} indicating strong alignment with the evaluation steps and {score_range[0]} indicating no alignment"
            reasoning_expectation = "Be specific and grounded in the evaluation steps."
            additional_context = (
                f"\n\nAdditional Context:\n{_additional_context}\n"
                if _additional_context
                else ""
            )

            # We use the rubric expected outcome text to plant the criteria text
            # into the template. However, the prefix prepended by deepeval 
            # ("<score_range>: ") should be removed beforehand.
            colon_space_index = rubric.find(": ")
            start_index = colon_space_index+2 if colon_space_index!=-1 and colon_space_index<10 else 0
            criteria_text = rubric[start_index:]

            # The prompts used by the G-Eval paper don't always include evaluation 
            # steps. We pass a single step "<<empty>>" to indicate that no evaluation
            # steps should be included.
            if "<<empty>>" in evaluation_steps:
                evaluation_steps = ""
            else:
                evaluation_steps = f"\nEvaluation Steps:\n{evaluation_steps}\n"


            prompt = textwrap.dedent(
                f"""You will be given one summary written for a news article. \nYour task is to rate the summary on one specific metric. \nPlease make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
{criteria_text}

Assess the extent to which the summary given below as 'Actual Output' for the news article given below as 'Input' meets the Evaluation Criteria using the reference summary given below as 'Expected Output'. Return a JSON object with two fields:

- `"score"`: an integer between {score_range[0]} and {score_range[1]}, {score_explanation}.
- `"reason"`: a brief explanation for why the score was given. This must mention specific strengths or shortcomings, referencing relevant details from the input. Do **not** quote the score itself in the explanation.

Your explanation should:
- {reasoning_expectation}
- Mention key details from the test case parameters.
- Be concise, clear, and focused on the evaluation logic.

Only return valid JSON. Do **not** include any extra commentary or text.

---
{evaluation_steps}

{rubric_text}
Test Case:
{test_case_content}

Parameters:
{parameters}
{additional_context}

---
**Example JSON:**
{{
    "reason": "your concise and informative reason here",
    "score": {score_range[0]}
}}

JSON:
"""
            )

            return prompt

    coherence_text = "Coherence (1-10) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby ”the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic.”. Do not evaluate consistency with the original article (correctness of the content can be ignored), fluency of the text at hand or relevance of the content, but rather the overall understandability of the summary as a whole."
    coherence_metric = GEval(
        name="Coherence",
        rubric=[Rubric(score_range=(1,10), expected_outcome=coherence_text)],# deepeval will extract score_range as follows: (rubric[0].score_range[0], rubric[-1].score_range[1])
        evaluation_steps=[
            "Read the 'Input' carefully and identify the main topic and key points.",
            "Read the 'Actual Output' and compare it to the 'Input' and the 'Expected Output'. Check if the 'Actual Output' covers the main topic and key points of the 'Input', and if it presents them in a clear and logical order.",
            "Assign a score for coherence on a scale of 1 to 10, where 1 is the lowest and 10 is the highest based on the Evaluation Criteria."
        ],
        verbose_mode=False,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        evaluation_template=CustomGEvalTemplate,
        model="gemini-2.0-flash-001"
    )
    consistency_text = "Consistency (1-10) - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts. Do not evaluate the coherence or fluency of the text at hand or the relevance of the content, but rather the factual correctness of the summary with respect to the original article."
    consistency_metric = GEval(
        name="Consistency",
        rubric=[Rubric(score_range=(1,10), expected_outcome=consistency_text)],# deepeval will extract score_range as follows: (rubric[0].score_range[0], rubric[-1].score_range[1])
        evaluation_steps=[
            "Read the 'Input' carefully and identify the main facts and details it presents.",
            "Read the 'Actual Output' and compare it to the 'Input'. Check if the 'Actual Output' contains any factual errors that are not supported by the article.",
            "Assign a score for consistency based on the Evaluation Criteria."
        ],
        verbose_mode=True,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        evaluation_template=CustomGEvalTemplate,
        model="gemini-2.0-flash-001"
    )
    fluency_text = """Fluency (1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure. Do not evaluate the coherence, consistency or relevance of the text (correctness of the content can be ignored), but rather grammatical and stylistic correctness.

- 1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
- 2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
- 3: Good. The summary has few or no errors and is easy to read and follow."""
    fluency_metric = GEval(
        name="Fluency",
        rubric=[Rubric(score_range=(1,3), expected_outcome=fluency_text)],# deepeval will extract score_range as follows: (rubric[0].score_range[0], rubric[-1].score_range[1])
        evaluation_steps=[
            "<<empty>>"
        ],
        verbose_mode=True,
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_template=CustomGEvalTemplate,
        model="gemini-2.0-flash-001"
    )
    relevance_text = "Relevance (1-10) - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information. Relevance is not about the coherence or fluency of the text at hand or about the consistency of the content with the original article (correctness of the content should not be evaluated), but rather about if all the relevant information from the article is included in the summary and if irrelevant information is avoided."
    relevance_metric = GEval(
        name="Relevance",
        rubric=[Rubric(score_range=(1,10), expected_outcome=relevance_text)],# deepeval will extract score_range as follows: (rubric[0].score_range[0], rubric[-1].score_range[1])
        evaluation_steps=[
            "Read the 'Actual Output' and the 'Input' carefully.",
            "Compare the 'Actual Output' to the 'Input' and identify the main points of the article.",
            "Assess how well the 'Actual Output' covers the main points of the 'Input', and how much irrelevant or redundant information it contains.",
            "Assign a relevance score from 1 to 10."
        ],
        verbose_mode=True,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        evaluation_template=CustomGEvalTemplate,
        model="gemini-2.0-flash-001"
    )

    all_scores = {
        "coherence": [], 
        "consistency": [],
        "fluency": [],
        "relevance": []
    }

    # Split dataset into parts to not overwhelm the judge LLM
    inst_per_step = 1 if intermittent else 25
    for t in range(0, len(texts), inst_per_step):
        range_to_take = range(t, min(t+inst_per_step, len(texts)))

        dataset = EvaluationDataset(goldens=[Golden(input=texts[i], actual_output=summaries[i], expected_output=label_summaries[i]) for i in range_to_take])
        for golden in dataset.goldens:
            dataset.add_test_case(
                LLMTestCase(input=golden.input, actual_output=golden.actual_output, expected_output=golden.expected_output)
            )

        # Retry with backoff on 429 / quota errors
        max_retries = 5
        for attempt in range(max_retries):
            try:
                eval_result = evaluate(test_cases=dataset.test_cases, metrics=[coherence_metric, consistency_metric, fluency_metric, relevance_metric])
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** (attempt + 1) * 10  # 20s, 40s, 80s, 160s, 320s
                print(f"LLM judge call failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
        # Returns list of TestResult, each containing a list metrics_data of MetricData, containing score and reason, as well as error.

        assert eval_result.test_results[0].metrics_data[0].name == "Coherence [GEval]" and eval_result.test_results[0].metrics_data[1].name == "Consistency [GEval]" and eval_result.test_results[0].metrics_data[2].name == "Fluency [GEval]" and eval_result.test_results[0].metrics_data[3].name == "Relevance [GEval]"
        assert eval_result.test_results[-1].metrics_data[0].name == "Coherence [GEval]" and eval_result.test_results[-1].metrics_data[1].name == "Consistency [GEval]" and eval_result.test_results[-1].metrics_data[2].name == "Fluency [GEval]" and eval_result.test_results[-1].metrics_data[3].name == "Relevance [GEval]"
        all_scores["coherence"] += [test_result.metrics_data[0].score for test_result in eval_result.test_results] 
        all_scores["consistency"] += [test_result.metrics_data[1].score for test_result in eval_result.test_results]
        all_scores["fluency"] += [test_result.metrics_data[2].score for test_result in eval_result.test_results]
        all_scores["relevance"] += [test_result.metrics_data[3].score for test_result in eval_result.test_results]

        if intermittent:
            print("sleeping")   # Do not overwhelm the judge LLM with too many requests
            # Always wait for a 1/3 minute: 1 min / floor(15/4), because with 1 min / ceil(15/4), there could be minutes in which ceil(15/4)=4 bursts take place -> 16 requests
            sleep_period = 1 / 3 * 60   # 3 = floor(15/4)
            if t==0: sleep_period *= 2
            time.sleep(sleep_period)

    all_scores["overall"] = [ (all_scores["coherence"][i] + all_scores["consistency"][i] + all_scores["fluency"][i] + all_scores["relevance"][i]) / 4.0 for i in range(len(all_scores["coherence"])) ]

    return all_scores

def eval_generation_with_llm_as_a_judge(dataset, model, tokenizer, reference_dataset=None, system_prompt=None, out_dataset_file=None, intermittent=False, tokenization_func=None):
    """Generates completions for each prompt in the dataset and computes the exit 
    layers of the generation process. Then assesses the completions using an 
    llm-as-a-judge given the prompts from the dataset or reference_dataset.

    Parameters
    ----------
    dataset : datasets.Dataset
        The dataset with the prompts to generate completions for. Each instance 
        should have a "prompt" field, a "completion" field (for evaluation) and 
        an "id" field.
    model : torch.nn.Module
        The language model to use for generation.
    tokenizer : transformers.PreTrainedTokenizer
        Will be used to encode the prompt and decode the model's output.
    reference_dataset : datasets.Dataset, optional
        An optional dataset with the reference prompts for which to compute the 
        summary scores. Each instance should have a "prompt" field.
    system_prompt : str, optional
        An optional system prompt to prepend to each prompt in the dataset using 
        role "system".
    out_dataset_file : str, optional
        An optional file path to save the dataset with generated completions.
    tokenization_func : callable, optional
        An optional function to use for tokenizing the input. See inference_fine_tuned.py:generate() 
        for more information.

    Returns
    -------
    summary_scores : dict
        A dict containing a list of scores (coherence, consistency, fluency, relevance, 
        overall) for the summaries generated as compared to the prompts from the 
        dataset or, if given, the prompts from the `reference_dataset` for each 
        instance in the dataset.
    exit_points : list of int
        A list of exit layers for all tokens generated.
    output_lengths : list of int
        A list of the number of tokens generated for each instance in the dataset.
    """
    
    temp_dataset = []   # Create a dataset to store the generated completions
    all_exit_points = []
    mean_exit_points = [] # mean over all tokens of an instance
    output_lengths = []
    summary_scores = { "coherence": [], "consistency": [], "fluency": [], "relevance": [], "overall": [] }
    min_duration = 0   # Ensure at least this delay between successive calls to the LLM judge to prevent exhausting the per-minute limit
    # Generate completions for each dataset prompt and track exit layers
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            start_time = time.time()
            output_ids, output_text, stats = inference.generate(model, tokenizer, True, dataset[i]["prompt"], system_prompt=system_prompt, max_new_tokens=512, do_sample=False, use_cache=False, skip_special_tokens=False, eval_stats=["exit_layer"], print_output=False, tokenization_func=tokenization_func)
            temp_dataset.append({"id": dataset[i]["id"], "completion": output_text, "exit_layers": stats["exit_layer"], "num_output_tokens": len(output_ids)})
            print(f"Exit layers: {stats['exit_layer']}")
            all_exit_points = all_exit_points + stats['exit_layer']
            mean_exit_points.append(round(torch.tensor(stats['exit_layer'], dtype=torch.float).mean().item(), 2))
            output_lengths.append(len(output_ids))

            if intermittent:
                # Compute summary score of the summary generated for the prompts from 
                # the reference_dataset or the dataset
                prompt = dataset[i]["prompt"] if not reference_dataset else reference_dataset[i]["prompt"]
                new_summary_scores = LLM_judge_summary([prompt], [output_text], [dataset[i]["completion"]])
                for key in summary_scores.keys():
                    summary_scores[key] += new_summary_scores[key]

                duration = time.time() - start_time
                if duration < min_duration: time.sleep(min_duration - duration)


    temp_dataset = Dataset.from_list(temp_dataset)

    if not intermittent:
        # Compute summary scores of all summaries generated for the prompts from 
        # the reference_dataset
        prompts = dataset["prompt"] if not reference_dataset else reference_dataset["prompt"]
        try:
            summary_scores = LLM_judge_summary(prompts, temp_dataset["completion"], dataset["completion"])
        except Exception as e:
            # A RetryError occurs when the model is overloaded.
            print(f"Error computing summary scores: {e}")
            summary_scores = None

    if out_dataset_file:
        output_dataset = temp_dataset.rename_column("completion", "generated")
        if summary_scores != None:
            summary_scores_column = [{key: val[i] for key, val in summary_scores.items()} for i in range(len(output_dataset))]
            output_dataset = output_dataset.add_column("summary_scores", summary_scores_column)
        if len(out_dataset_file)>100:
            # If the filename is too long, use a timestamp as filename and save 
            # the mapping in file_descriptions.txt
            path = out_dataset_file[:out_dataset_file.rfind("/")+1]    # Whole filename until (inclusively) the last occurence of "/"
            timestamp_filename = path + str(int(time.time())) + ".json"
            with open(path + "file_descriptions.txt", "a") as f:
                f.write(f"\n{timestamp_filename} - {out_dataset_file[len(path):].replace('.json', '')}")
            out_dataset_file = timestamp_filename
        output_dataset.to_json(out_dataset_file)


    print("Mean exit points per instance: ", mean_exit_points)
    return summary_scores, all_exit_points, mean_exit_points, output_lengths


def load_judged_generations(dataset_dir, dataset_file, original_dataset=None):
    """Loads the scores and exit point from a dataset file generated with eval_generation_with_llm_as_a_judge
    and returns the same values as this function. If the dataset file does not 
    contain summary scores, they will be computed using the original_dataset.
    
    Parameters
    ----------
    dataset_dir : str
        The directory where the dataset file is located.
    dataset_file : str
        The name of the dataset file.
    original_dataset : datasets.Dataset, optional
        A dataset containing fields "prompt" and "completion" to compute summary
        scores on if the dataset file does not contain them.

    Returns
    -------
    summary_scores : dict
        A dict containing a list of scores (coherence, consistency, fluency, relevance, 
        overall) for the summaries as stored in the dataset_file, or, if not present, 
        assessed in comparison to the prompts from the original_dataset for each 
        instance in the dataset.
    exit_points : list of int
        A list of exit layers for all tokens generated.
    output_lengths : list of int
        A list of the number of tokens generated for each instance in the dataset.
    """
    generations_dataset = load_dataset("json", data_dir=dataset_dir, data_files={"validation": dataset_file}, split="validation")
    summary_scores = { "coherence": [], "consistency": [], "fluency": [], "relevance": [], "overall": [] }
    all_exit_points = []
    mean_exit_points = [] # mean over all tokens of an instance
    output_lengths = []

    if not "summary_scores" in generations_dataset.column_names and original_dataset:
        # Compute summary scores of all summaries generated for the prompts from 
        # the original_dataset
        prompts = original_dataset["prompt"]
        try: 
            summary_scores = LLM_judge_summary(prompts, generations_dataset["generated"], original_dataset["completion"])

            # Save updated dataset
            summary_scores_column = [{key: val[i] for key, val in summary_scores.items()} for i in range(len(generations_dataset))]
            generations_dataset = generations_dataset.add_column("summary_scores", summary_scores_column)
            file_path = (dataset_dir if dataset_dir.endswith("/") else dataset_dir + "/") + dataset_file
            generations_dataset.to_json(file_path)
        except Exception as e:
            print(f"Error computing summary scores: {e}")
            print("Continuing with empty summary scores.")
            summary_scores_column = [{key: 0.0 for key, val in summary_scores.items()} for i in range(len(generations_dataset))]
            generations_dataset = generations_dataset.add_column("summary_scores", summary_scores_column)
            # Do not save this updated dataset.

    if not "num_output_tokens" in generations_dataset.column_names:
        print("Computing number of output tokens using the inference.tokenizer")
        num_output_tokens_column = [len(inference.tokenizer(generations_dataset[i]["generated"], return_tensors="pt")["input_ids"][0]) for i in range(len(generations_dataset))]
        generations_dataset = generations_dataset.add_column("num_output_tokens", num_output_tokens_column)


    for i in tqdm(range(len(generations_dataset))):
        instance_summary_scores = generations_dataset[i]["summary_scores"]
        for key in instance_summary_scores.keys():
            summary_scores[key].append(instance_summary_scores[key])

        all_exit_points = all_exit_points + generations_dataset[i]['exit_layers']
        mean_exit_points.append(round(torch.tensor(generations_dataset[i]['exit_layers'], dtype=torch.float).mean().item(), 2))
        output_lengths.append(generations_dataset[i]["num_output_tokens"])

    return summary_scores, all_exit_points, mean_exit_points, output_lengths

def compute_early_exit_compute_stats(
    exit_points,
    full_model_layers,
    macs_per_layer=None,
):
    """
    Estimate relative MAC/FLOP reduction from token-level exit layers.
    """

    exit_points = torch.tensor(exit_points, dtype=torch.float)

    total_tokens = exit_points.numel()
    early_exit_layer_executions = exit_points.sum().item()
    full_layer_executions = total_tokens * full_model_layers

    compute_fraction = early_exit_layer_executions / full_layer_executions
    compute_reduction = 1.0 - compute_fraction

    stats = {
        "total_tokens": total_tokens,
        "early_exit_layer_executions": early_exit_layer_executions,
        "full_layer_executions": full_layer_executions,
        "compute_fraction": compute_fraction,
        "compute_reduction_percent": compute_reduction * 100,
    }

    if macs_per_layer is not None:
        early_exit_macs = early_exit_layer_executions * macs_per_layer
        full_macs = full_layer_executions * macs_per_layer

        stats.update({
            "early_exit_macs": early_exit_macs,
            "full_macs": full_macs,
            "early_exit_flops": 2 * early_exit_macs,
            "full_flops": 2 * full_macs,
        })

    return stats

def clean_thop_buffers(model):
    """
    Remove THOP buffers/hooks from previous profiling attempts.
    """

    for module in model.modules():
        for attr in ("total_ops", "total_params"):
            if attr in module._buffers:
                del module._buffers[attr]
            elif hasattr(module, attr):
                delattr(module, attr)



def estimate_model_macs_per_token(model, tokenizer, seq_len=128):
    """
    Estimate MACs/token using THOP without modifying the original model.
    """

    import copy
    from thop import profile

    model_for_profile = copy.deepcopy(model)
    model_for_profile.eval()
    clean_thop_buffers(model_for_profile)

    dummy_input_ids = torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=(1, seq_len),
        device=model.device
    )

    dummy_attention_mask = torch.ones_like(dummy_input_ids)

    with torch.no_grad():
        macs, params = profile(
            model_for_profile,
            inputs=(dummy_input_ids,),
            verbose=False
        )

    del model_for_profile
    torch.cuda.empty_cache()

    return {
        "total_macs_for_seq": macs,
        "macs_per_token": macs / seq_len,
        "params": params,
    }

def compute_macs_from_exit_points(
    exit_points,
    full_model_layers,
    macs_per_token_full_model,
):
    """
    Estimate actual MACs/FLOPs used by early-exit generation.
    """

    exit_points = torch.tensor(exit_points, dtype=torch.float)

    total_tokens = exit_points.numel()

    full_layer_executions = total_tokens * full_model_layers
    early_exit_layer_executions = exit_points.sum().item()

    compute_fraction = early_exit_layer_executions / full_layer_executions

    full_macs = total_tokens * macs_per_token_full_model
    early_exit_macs = full_macs * compute_fraction

    return {
        "total_tokens": total_tokens,
        "full_layer_executions": full_layer_executions,
        "early_exit_layer_executions": early_exit_layer_executions,

        "compute_fraction": compute_fraction,
        "compute_reduction_percent": 100 * (1 - compute_fraction),

        "full_macs": full_macs,
        "early_exit_macs": early_exit_macs,
        "mac_reduction": full_macs - early_exit_macs,

        "full_flops": 2 * full_macs,
        "early_exit_flops": 2 * early_exit_macs,
        "flop_reduction": 2 * (full_macs - early_exit_macs),
    }

def print_histogram_string(hist, ref_hist=None, indention=""):
    """Print a Unicode bar chart from a dict of {label: fraction}."""
    # block = "▆"
    block = "■"
    reference_sep = "|"
    space = " "
    scale = 40  # number of blocks for a full (1.0) bar

    hist_string = ""
    for key, value in hist.items():
        bar_width = int(value * scale)
        bar = block * bar_width

        if ref_hist != None:
            ref_pos = int(ref_hist[key]*scale)
            bar += space * max(0, ref_pos-bar_width)
            bar = bar[:ref_pos-1] + reference_sep + bar[ref_pos:]

        key_trunc = key if len(str(key))<8 else str(key)[:7] + "."
        percent = f"{value*100:.2f}%"
        hist_string += indention + f"{key_trunc:<8}▕{bar} {percent}\n"
        
    if ref_hist==None:
        hist_string += indention + f"Frequencies for reference: {str(hist)}\n"

    return hist_string[:-1] # cut last \n


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Provides various evaluation functions to evaluate the performance of the EELlama model.")
    parser.add_argument("--eval", type=str, help="The evaluation function to execute.")
    parser.add_argument("--model-timestamp", type=str, help="The timestamp of the model to evaluate.")
    parser.add_argument("--eval-perplexities", action="store_true", help="Set, if perplexities should be evaluated in addition to the eval selected.")
    cli_args = parser.parse_args()
    EVAL = cli_args.eval if cli_args.eval is not None else EVAL
    model_timestamp = "0000000000"
    if cli_args.model_timestamp is not None:
        model_timestamp = cli_args.model_timestamp
    eval_perplexities = cli_args.eval_perplexities if cli_args.eval_perplexities else False


    dataset = dataset.rename_column("article", "prompt")
    dataset = dataset.rename_column("highlights", "completion")

    indention = ""
    output_string = ""

    def print_outputs(reference_exit_frequencies="saved"):
        # set reference_exit_frequencies=None, if no reference histogram should be displayed.
        if reference_exit_frequencies=="saved":
            # reference_exit_frequencies = {7: 0.1140, 15: 0.1793, 23: 0.2349, 31: 0.4717}  # fine-tuned model
            # reference_exit_frequencies = {7: 0.1219, 15: 0.1777, 23: 0.2400, 31: 0.4605}    # fine-tuned model with tuned embedding and lm_head
            # reference_exit_frequencies = {7: 0.1341, 15: 0.1743, 23: 0.2391, 31: 0.4524}    # fine-tuned model with fine-tuned embedding and lm_head on cnn-dm_validation_short-shuffled52-102.json
            reference_exit_frequencies = {1: 0.0214, 2: 0.2076, 3: 0.0581, 4: 0.0285, 5: 0.0963, 6: 0.0214, 7: 0.0255, 8: 0.0217, 9: 0.0169, 10: 0.0371, 11: 0.0184, 12: 0.0105, 13: 0.0086, 14: 0.0079, 15: 0.4202}    # eellama-3p2-1B-layerskip with exp-decaying thresolds on cnn-dm_validation_short-shuffled52-102.json


        global output_string, indention, all_exit_points
        all_exit_points = torch.tensor(all_exit_points, dtype=torch.float)
        exit_frequencies = {exit_layer: round((all_exit_points == exit_layer).sum().item()/all_exit_points.shape[0], 4) for exit_layer in inference.exit_layers}

        ref_hist = reference_exit_frequencies
        output_string += indention + f"Exit points mean: {round(all_exit_points.mean().item(), 4)}, std: {round(all_exit_points.std().item(), 4)}\n"
        # output_string += f"Exit frequencies: {str(exit_frequencies)}"
        output_string += indention + f"Exit frequencies: \n{print_histogram_string(exit_frequencies, ref_hist=ref_hist, indention=indention)}\n"

        output_string += indention + f"\nModel timestamp: {model_timestamp}\n"

        output_string += "\n\n"

        if PRINT_TO_FILE:
            with open(results_file_path, "a") as results_file:
                results_file.writelines(output_string)
        else:
            print(output_string)


    def model_evaluation(model, tokenizer, tokenization_func=None, dataset_files=["cnn-dm_validation_short-shuffled32-52.json"], load_generations=False, generations_files=None, file_appendix="_generated_completions", description="", is_reference=False):
        """Evaluate the given model on the given datasets and print the results.

        Parameters
        ----------
        dataset : datasets.Dataset
            The dataset with the prompts to generate completions for. 
        model : torch.nn.Module
            The language model to use for generation.
        tokenizer : transformers.PreTrainedTokenizer
            Will be used to encode the prompt and decode the `model`'s output.
        tokenization_func : callable
            An optional function to use for tokenizing the input. See inference_fine_tuned.py:generate() 
            for more information.
        dataset_files : list of string
            The file names of the datasets to use as present in the `data_dir` (global).
            Each instance in the dataset should have a "prompt" field, a "completion" 
            field (for evaluation) and an "id" field.
        load_generations : bool
            Whether to load the results from already generated files given in `generations_files`
            instead of generating results with the model and the LLM-as-a-judge.
        generations_files : list of string
            The file names of the datasets containing the already generated results 
            as present in the `out_dir` (global). The filenames passed in `dataset_files`
            should correspond to the generated files given in `generations_files`.
        file_appendix : string
            An appendix to add at the end of the dataset filename. The string will be 
            used as output file name for the generated results dataset json file.
        description : string
            An optional string to print in the output before printing the results, which
            can be used to describe the experiment. Indentions are automatically inserted
            before each line of the string.
        is_reference : bool
            If True, no reference histogram is included in the output diagram.
        
        """

        global output_string, data_dir, out_dir, all_exit_points

        assert not load_generations or (len(dataset_files)==len(generations_files)), "When loading previously generated files, the corresponding datasets must be given in dataset_files."

        for i, data_file in enumerate(dataset_files):
            # In case load_generations is True, the dataset would actually only 
            # be needed, if summary scores have to be recomputed.
            if data_file.endswith(".json"):
                dataset = load_dataset("json", data_dir=data_dir, data_files={"validation": data_file}, split="validation")
            else:
                dataset = load_from_disk(f"{data_dir}/{data_file}")
            dataset = dataset.rename_column("article", "prompt")
            dataset = dataset.rename_column("highlights", "completion")

            output_string = ""
            if description != "": output_string += indention + ("\n"+indention).join(description.split("\n"))
            output_string += indention + f"Dataset: {data_file}\n"

            if eval_perplexities:
                perplexities, all_exit_points = perplexity(dataset, model, tokenizer)
                output_string += indention + "\nPerplexities: "
                output_string += str(perplexities) + "\n"
                output_string += indention + f"ppl mean: {round(torch.tensor(perplexities).mean().item(), 4)}, "
                output_string += f"ppl std: {round(torch.tensor(perplexities).std().item(), 4)}\n"
                all_exit_points = torch.tensor(all_exit_points, dtype=torch.float)
                output_string += indention + f"Exit points mean (Teacher Forcing): {round(all_exit_points.mean().item(), 4)}, std: {round(all_exit_points.std().item(), 4)}\n\n"

            
            seed = 0
            print(f"Setting seed {seed} for judged evaluation")
            torch.manual_seed(seed) # (line added at time 1774469415 in addition to the line already present in inference_fine_tuned.py to counteract the line added earlier in the perplexity() function)
        
            if out_dir[-1] != "/": out_dir += "/"
            if not load_generations:
                evaluation_start_time = time.time()
                out_dataset_file=f"{out_dir}{data_file.replace('.json', file_appendix + '.json')}" if data_file.endswith(".json") else f"{out_dir}{data_file}{file_appendix + '.json'}"
                summary_scores, all_exit_points, mean_exit_points, output_lengths = eval_generation_with_llm_as_a_judge(dataset, model, tokenizer, reference_dataset=None, system_prompt=None, out_dataset_file=out_dataset_file, tokenization_func=tokenization_func)
                evaluation_duration = time.time() - evaluation_start_time
            else:
                summary_scores, all_exit_points, mean_exit_points, output_lengths = load_judged_generations(out_dir, generations_files[i], original_dataset=dataset)
                evaluation_duration = 0
            
            print(output_lengths)
            print(all_exit_points)
            print(mean_exit_points)

            output_string += indention + f"Evaluation duration: { evaluation_duration//60:.0f}:{evaluation_duration%60:.0f} minutes\n"
            output_string += indention + f"Mean output length: {round(torch.tensor(output_lengths, dtype=torch.float).mean().item(), 4)}, total tokens generated: {sum(output_lengths)}\n"

            if summary_scores is None:
                output_string += indention + "Summary scores: unavailable (LLM judge failed)\n\n"
            else:
                output_string += indention + f"Std of overall summary score: {round(torch.tensor(summary_scores['overall']).std().item(), 4)}\n"

                # reference_sum_scores = None if is_reference else {'coherence': 0.45555558800697327, 'consistency': 0.7555555701255798, 'fluency': 1.0, 'relevance': 0.4500000476837158, 'overall': 0.6652777194976807}  # fine-tuned model
                # reference_sum_scores = None if is_reference else {'coherence': 0.5000, 'consistency': 0.7556, 'fluency': 1.0, 'relevance': 0.4944, 'overall': 0.6875}  # fine-tuned model with fine-tuned embedding and lm_head
                # reference_sum_scores = None if is_reference else {'coherence': 0.5267, 'consistency': 0.7267, 'fluency': 1.0, 'relevance': 0.4956, 'overall': 0.6872}  # fine-tuned model with fine-tuned embedding and lm_head on cnn-dm_validation_short-shuffled52-102.json
                reference_sum_scores = None if is_reference else {'coherence': 0.47333335876464844, 'consistency': 0.7244445085525513, 'fluency': 0.9900000095367432, 'relevance': 0.47111114859580994, 'overall': 0.664722204208374}  # eellama-3p2-1B-layerskip with exp-decaying thresolds on cnn-dm_validation_short-shuffled52-102.json
                mean_summary_scores = {key: torch.tensor(val).mean().item() for key, val in summary_scores.items()}
                sum_scores_indention = "\t\t\t\t\t\t"
                output_string += indention + sum_scores_indention + "Summary scores:\n"
                output_string += print_histogram_string(mean_summary_scores, ref_hist=reference_sum_scores, indention=indention+sum_scores_indention)
                output_string += "\n\n"
            output_string += indention + f"Mean exit points per instance: {mean_exit_points}\n"

            if is_reference:
                print_outputs(reference_exit_frequencies=None)
            else:
                print_outputs()

    print()

    if EVAL=="perplexities":
        inference.instantiate_model()
        perplexities, all_exit_points = perplexity(dataset, inference.model, inference.tokenizer)
        output_string += indention + "\nPerplexities: "
        output_string += str(perplexities) + "\n"
        output_string += indention + f"mean: {round(torch.tensor(perplexities).mean().item(), 4)}, "
        output_string += f"std: {round(torch.tensor(perplexities).std().item(), 4)}\n"

        print_outputs(reference_exit_frequencies=None)

    elif EVAL=="generation":
        inference.instantiate_model()
        print("Loading reference dataset from disk")
        reference_dataset = load_dataset("json", data_dir="../datasets/", data_files={"validation": "cnn-dm_validation_short-shuffled32-52.json"}, split="validation")
        reference_dataset = reference_dataset.rename_column("article", "prompt")
        reference_dataset = reference_dataset.rename_column("highlights", "completion")

        perplexities, all_exit_points = eval_generation(dataset, inference.model, inference.tokenizer, reference_dataset=reference_dataset, system_prompt=None, out_dataset_file=f"{data_dir}{data_file.replace('.json', '_generated_completions.json')}")
        indention = "\t\t\t\t"
        output_string += indention + "Free generation\n"
        output_string += indention + "Perplexities "
        output_string += indention + "\nPerplexities: "
        output_string += str(perplexities) + "\n"
        output_string += indention + f"mean: {round(torch.tensor(perplexities).mean().item(), 4)}, "
        output_string += f"std: {round(torch.tensor(perplexities).std().item(), 4)}\n"

    elif EVAL=="grid_search":
        # threshold_vals = torch.tensor([1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.8, 0.7], dtype=torch.float64)
        # threshold_grid = threshold_vals.tolist()    # fixed threshold for all exit layers
        # threshold_grid = torch.cat([torch.linspace(1.0, lowest_threshold, 16)[1:].unsqueeze(0) for lowest_threshold in torch.arange(1.0, -0.05, -0.05)], dim=0).tolist()     # The sixteen because of the 16 layers in total (15 exit layers, we remove the first value.)
        threshold_grid = torch.cat([torch.tensor([(1-lowest_threshold)*torch.exp(-((1-lowest_threshold)/15*4)/(1-lowest_threshold+1e-6)*l) + lowest_threshold for l in range(16)])[1:].unsqueeze(0) for lowest_threshold in torch.arange(1.0, 0.49, -0.01)], dim=0).tolist()

        model_parameter_grid_search(inference.instantiate_model, dataset, ee_softmax_threshold=threshold_grid)

    elif EVAL=="judge":
        reference_dataset = load_dataset("json", data_dir="../datasets/", data_files={"validation": "cnn-dm_validation_short-shuffled32-52.json"}, split="validation")
        reference_dataset = reference_dataset.rename_column("article", "prompt")
        reference_dataset = reference_dataset.rename_column("highlights", "completion")
        generated_dataset = load_dataset("json", data_dir="../datasets/softmax_thresholds_(0p68, 0p62, 0p56, 0p5)/", data_files={"validation": "cnn-dm_validation_short-shuffled32-52_generated_completions_no_system_prompt.json"}, split="validation")
        all_scores = LLM_judge_summary(reference_dataset["prompt"][:3], generated_dataset["generated"][:3], reference_dataset["completion"][:3])
        
        print("All scores: ", all_scores)

    elif EVAL=="generation_judged":

        inference.instantiate_model()
        print("Device: ", inference.model.device)

        # # Load all the parameters updated in state_dict
        # state_dict_path = "../models/eellama-3-8B-instruct-lora-tuned-embedding/tuned_embedding_and_lm_head.pth"
        # state_dict = torch.load(state_dict_path)
        # assert type(inference.model.base_model.model.model).__name__=="EELlamaModel"  # The structure might be different if not using a peft model or so.
        # missing_keys, unexpected_keys = inference.model.load_state_dict(state_dict, strict=False)
        # assert len(unexpected_keys) == 0, "The following parameters could not be loaded: " + str(unexpected_keys)

        # # Load an early-exit-tuned layer 0
        # state_dict_path = f"../models/llama-3-eellama-3p2-1B-layerskip/CEDA-l-02_state_dict_update_layer0.pth"
        # state_dict = torch.load(state_dict_path)
        # assert type(inference.model.base_model.model.model).__name__=="EELlamaModel"  # The structure might be different if not using a peft model or so.
        # missing_keys, unexpected_keys = inference.model.load_state_dict(state_dict, strict=False)
        # assert len(unexpected_keys) == 0, "The following parameters could not be loaded: " + str(unexpected_keys)

        load_generations = False
        files = [
            # "cnn-dm_validation_short-shuffled52-102.json",
            data_file
        ]
        generations_files = [
        ]
        is_reference = True

        file_appendix="_generated_completions"
        model_evaluation(inference.model, inference.tokenizer, dataset_files=files, load_generations=load_generations, generations_files=generations_files, file_appendix=file_appendix, description="", is_reference=is_reference)

    elif EVAL=="MACs/FLOPs":

        inference.instantiate_model()
        print("Device: ", inference.model.device)

        print("\nEstimating MACs with THOP...")
        mac_profile = estimate_model_macs_per_token(
            inference.model,
            inference.tokenizer,
            seq_len=128
        )

        print(f"Parameters: {mac_profile['params']:,}")
        print(f"MACs for sequence length 128: {mac_profile['total_macs_for_seq']:,}")
        print(f"Estimated MACs/token: {mac_profile['macs_per_token']:,.2f}")

        files = [
            data_file
        ]

        for current_data_file in files:
            if current_data_file.endswith(".json"):
                dataset = load_dataset(
                    "json",
                    data_dir=data_dir,
                    data_files={"validation": current_data_file},
                    split="validation"
                )
            else:
                dataset = load_from_disk(f"{data_dir}/{current_data_file}")

            dataset = dataset.rename_column("article", "prompt")
            dataset = dataset.rename_column("highlights", "completion")

            print(f"\nDataset: {current_data_file}")

            _, all_exit_points, mean_exit_points, output_lengths = \
                eval_generation_with_llm_as_a_judge(
                    dataset,
                    inference.model,
                    inference.tokenizer,
                    reference_dataset=None,
                    system_prompt=None,
                    out_dataset_file=None,
                    intermittent=False
                )

            compute_stats = compute_macs_from_exit_points(
                exit_points=all_exit_points,
                full_model_layers=max(inference.exit_layers),
                macs_per_token_full_model=mac_profile["macs_per_token"]
            )

            print("\nCompute estimate:")
            print(f"Total generated tokens: {compute_stats['total_tokens']}")
            print(f"Full layer executions: {compute_stats['full_layer_executions']:.2f}")
            print(f"Early-exit layer executions: {compute_stats['early_exit_layer_executions']:.2f}")
            print(f"Compute fraction: {compute_stats['compute_fraction']:.4f}")
            print(f"Compute reduction: {compute_stats['compute_reduction_percent']:.2f}%")

            print("\nMAC/FLOP estimate:")
            print(f"Full MACs: {compute_stats['full_macs']:,.2f}")
            print(f"Early-exit MACs: {compute_stats['early_exit_macs']:,.2f}")
            print(f"MAC reduction: {compute_stats['mac_reduction']:,.2f}")

            print(f"Full FLOPs: {compute_stats['full_flops']:,.2f}")
            print(f"Early-exit FLOPs: {compute_stats['early_exit_flops']:,.2f}")
            print(f"FLOP reduction: {compute_stats['flop_reduction']:,.2f}")


    elif EVAL=="Estimated_Computation":

        inference.instantiate_model()
        print("Device: ", inference.model.device)

        files = [
            data_file
        ]

        for data_file in files:
            if data_file.endswith(".json"):
                dataset = load_dataset(
                    "json",
                    data_dir=data_dir,
                    data_files={"validation": data_file},
                    split="validation"
                )
            else:
                dataset = load_from_disk(f"{data_dir}/{data_file}")

            dataset = dataset.rename_column("article", "prompt")
            dataset = dataset.rename_column("highlights", "completion")

            print(f"\nDataset: {data_file}")

            _, all_exit_points, mean_exit_points, output_lengths = \
                eval_generation_with_llm_as_a_judge(
                    dataset,
                    inference.model,
                    inference.tokenizer,
                    reference_dataset=None,
                    system_prompt=None,
                    out_dataset_file=None,
                    intermittent=False
                )

            compute_stats = compute_early_exit_compute_stats(
                exit_points=all_exit_points,
                full_model_layers=max(inference.exit_layers),
                macs_per_layer=None
            )

            print("\nCompute estimate:")
            print(f"Total generated tokens: {compute_stats['total_tokens']}")
            print(f"Full layer executions: {compute_stats['full_layer_executions']:.2f}")
            print(f"Early-exit layer executions: {compute_stats['early_exit_layer_executions']:.2f}")
            print(f"Compute fraction: {compute_stats['compute_fraction']:.4f}")
            print(f"Estimated compute reduction: {compute_stats['compute_reduction_percent']:.2f}%")


    elif EVAL=="timing_analysis":
        start_time = time.time()
        inference.instantiate_model()
        time_instantiated = time.time()

        perplexities, all_exit_points = perplexity(dataset, inference.model, inference.tokenizer)
        time_perplexity = time.time()

        perplexities, all_exit_points = eval_generation(dataset, inference.model, inference.tokenizer, reference_dataset=None, out_dataset_file=None)
        time_generation = time.time()

        def format_time(seconds):
            return f"{seconds:.2f} s, ({int(seconds//60)}:{int(seconds%60)} min)"

        print("Dataset: ", data_file)
        print("Time to instantiate model: ", format_time(time_instantiated - start_time))
        print("Time to compute perplexity (no generation): ", format_time(time_perplexity - time_instantiated))
        print("Time for generation: ", format_time(time_generation - time_perplexity))

    if EVAL in ["generation"]:
        print_outputs()
