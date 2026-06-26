"""
Evaluation script comparing three model variants on SST-2 sentiment classification:

  1. Base Llama          — full-precision inference with AutoModelForCausalLM
  2. EELlamma            — confidence-based early exit (softmax threshold)
  3. SkipDecode EELlamma — position-based layer budget (SkipDecode)

Each variant is scored on accuracy, wall-clock latency, average active layers per
token, and estimated compute reduction relative to the full model.

Usage
-----
    cd src/
    python evaluate_sst2.py
    python evaluate_sst2.py --config configs/evaluation/evaluate_sst2.yaml
    python evaluate_sst2.py --num_examples 100
"""

import argparse
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers_override.models.llama.modeling_eellama import EeLlamaForCausalLM
from utils.configuration_utils import GenericConfig
import utils.logging_utils  # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template — must match fine_tune_sst2.py exactly
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a sentiment classifier. "
    "Reply with exactly one word: 'positive' or 'negative'."
)

LABEL_WORDS = {0: "negative", 1: "positive"}


def make_prompt(sentence: str) -> str:
    return (
        f"Classify the sentiment of the following movie review.\n\n"
        f"Review: {sentence}\n"
        f"Sentiment:"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class EvalConfig(GenericConfig):
    """Configuration for the SST-2 evaluation script."""

    def __init__(self):
        config: Dict[str, Any] = {
            # Number of validation examples to evaluate (null = all ~872)
            "num_examples": None,

            # Base Llama model (no early exit)
            "base_model": {
                "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                "adapter_path": None,
                "enabled": True,
            },

            # EELlamma with confidence-based early exit
            "eellama_model": {
                "model_id": "meta-llama/Llama-3.2-1B",
                "adapter_path":"./models/llama-3-eellama-3p2-1B-layerskip/layer0",
                "enabled": True,
                "exit_layers": [3, 7, 11, 15],
                "ee_softmax_threshold": 0.9,
                "ee_entropy_threshold": None,
                "untied_heads": False,
            },

            # SkipDecode EELlamma
            "skipdecode_model": {
                "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                "adapter_path": "./models/eellama-skipdecode-sst2-final",
                "enabled": True,
                "exit_layers": [3, 7, 11, 15],
                "untied_heads": False,
                "skipdecode_min_exit_layer": 5,
                "skipdecode_max_exit_layer": 11,
                "skipdecode_num_warmup_layers": 1,
                "skipdecode_max_sequence_length": 128,
                "skipdecode_prompt_size": 40,
            },
        }
        super().__init__(config)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_tokenizer(model_id: str, adapter_path: Optional[str] = None) -> AutoTokenizer:
    source = adapter_path if adapter_path else model_id
    if source and os.path.exists(source):
        source = os.path.abspath(source)
    tok = AutoTokenizer.from_pretrained(source)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def _resolve_path(path: Optional[str]) -> Optional[str]:
    if path and os.path.exists(path):
        return os.path.abspath(path)
    return path


def load_base_model(cfg: Dict[str, Any]):
    logger.info("Loading base Llama model: %s", cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.bfloat16,
        device_map=0,
    )
    adapter_path = _resolve_path(cfg.get("adapter_path"))
    tokenizer = _load_tokenizer(cfg["model_id"], adapter_path)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def load_eellama_model(cfg: Dict[str, Any]):
    logger.info("Loading EELlamma model: %s", cfg["model_id"])
    kwargs: Dict[str, Any] = {
        "exit_layers": cfg["exit_layers"],
        "untied_heads": cfg.get("untied_heads", False),
        "output_full_model": False,
    }
    if cfg.get("ee_softmax_threshold") is not None:
        kwargs["ee_softmax_threshold"] = cfg["ee_softmax_threshold"]
    if cfg.get("ee_entropy_threshold") is not None:
        kwargs["ee_entropy_threshold"] = cfg["ee_entropy_threshold"]

    model = EeLlamaForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.bfloat16,
        device_map=0,
        **kwargs,
    )
    adapter_path = _resolve_path(cfg.get("adapter_path"))
    tokenizer = _load_tokenizer(cfg["model_id"], adapter_path)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def load_skipdecode_model(cfg: Dict[str, Any]):
    logger.info("Loading SkipDecode EELlamma model: %s", cfg["model_id"])
    model = EeLlamaForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.bfloat16,
        device_map=0,
        exit_layers=cfg["exit_layers"],
        untied_heads=cfg.get("untied_heads", False),
        output_full_model=False,
        skipdecode_enabled=True,
        skipdecode_min_exit_layer=cfg.get("skipdecode_min_exit_layer", 0),
        skipdecode_max_exit_layer=cfg.get("skipdecode_max_exit_layer", None),
        skipdecode_num_warmup_layers=cfg.get("skipdecode_num_warmup_layers", 1),
        skipdecode_max_sequence_length=cfg.get("skipdecode_max_sequence_length", 128),
        skipdecode_prompt_size=cfg.get("skipdecode_prompt_size", 0),
    )
    adapter_path = _resolve_path(cfg.get("adapter_path"))
    tokenizer = _load_tokenizer(cfg["model_id"], adapter_path)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def get_label_token_ids(tokenizer) -> Dict[str, int]:
    ids = {}
    for word in LABEL_WORDS.values():
        toks = tokenizer.encode(" " + word, add_special_tokens=False)
        ids[word] = toks[0]
    return ids


def evaluate_model(
    model,
    tokenizer,
    dataset,
    label_token_ids: Dict[str, int],
    model_name: str,
    num_layers: int,
    collect_exit_layers: bool = False,
) -> Dict[str, Any]:
    """Score a single model on the SST-2 validation set.

    Returns a dict with accuracy, latency, and (optionally) exit-layer stats.
    """
    pos_id = label_token_ids["positive"]
    neg_id = label_token_ids["negative"]

    correct = 0
    total = 0
    latencies: List[float] = []
    exit_layers_all: List[int] = []

    logger.info("Evaluating %s on %d examples ...", model_name, len(dataset))

    with torch.inference_mode():
        for example in dataset:
            sentence = example["sentence"]
            gt_label = LABEL_WORDS[example["label"]]
            gt_id = label_token_ids[gt_label]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_prompt(sentence)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            stats: Dict[str, Any] = {}
            t0 = time.perf_counter()

            if collect_exit_layers:
                outputs = model(
                    **inputs,
                    logits_to_keep=1,
                    eval_stats=["exit_layer"],
                    stats=stats,
                )
            else:
                outputs = model(**inputs, logits_to_keep=1)

            t1 = time.perf_counter()

            last_logits = outputs.logits[0, -1]
            pred_id = pos_id if last_logits[pos_id] > last_logits[neg_id] else neg_id

            correct += int(pred_id == gt_id)
            total += 1
            latencies.append(t1 - t0)

            if collect_exit_layers and "exit_layer" in stats:
                raw = stats["exit_layer"]
                # stats["exit_layer"] may be a list-of-lists (one per seq position)
                # or a flat list; flatten to get all recorded layers
                if raw and isinstance(raw[0], list):
                    for sub in raw:
                        exit_layers_all.extend(sub)
                else:
                    exit_layers_all.extend(raw)

    accuracy = correct / total if total > 0 else 0.0
    lat_tensor = torch.tensor(latencies)

    result: Dict[str, Any] = {
        "model": model_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "latency_mean_ms": lat_tensor.mean().item() * 1000,
        "latency_std_ms": lat_tensor.std().item() * 1000,
        "latency_total_s": lat_tensor.sum().item(),
    }

    if exit_layers_all:
        el = torch.tensor(exit_layers_all, dtype=torch.float)
        result["exit_layer_mean"] = el.mean().item()
        result["exit_layer_std"] = el.std().item()
        result["compute_fraction"] = el.mean().item() / num_layers
        result["compute_reduction_pct"] = (1.0 - el.mean().item() / num_layers) * 100
    else:
        result["exit_layer_mean"] = float(num_layers)
        result["exit_layer_std"] = 0.0
        result["compute_fraction"] = 1.0
        result["compute_reduction_pct"] = 0.0

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results_table(results: List[Dict[str, Any]]) -> None:
    header = (
        f"{'Model':<28} {'Accuracy':>10} {'Lat (ms)':>12} {'Lat std':>10} "
        f"{'Avg exit layer':>16} {'Compute red.':>14}"
    )
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("SST-2 Evaluation Results")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['model']:<28} "
            f"{r['accuracy']:>9.4f}  "
            f"{r['latency_mean_ms']:>10.1f}  "
            f"{r['latency_std_ms']:>8.1f}  "
            f"{r['exit_layer_mean']:>14.2f}  "
            f"{r['compute_reduction_pct']:>12.1f}%"
        )
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Llama / EELlamma / SkipDecode on SST-2")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file (optional; defaults are used otherwise).",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Number of validation examples to evaluate (default: all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.config:
        config = EvalConfig.load(args.config)
    else:
        config = EvalConfig()

    if args.num_examples is not None:
        config["num_examples"] = args.num_examples

    logger.info("Configuration:\n%s", config.to_string())

    # Load SST-2 validation split
    logger.info("Loading SST-2 validation split ...")
    raw = load_dataset("glue", "sst2")
    val_split = raw["validation"]
    if config["num_examples"] is not None:
        val_split = val_split.select(range(min(config["num_examples"], len(val_split))))
    logger.info("Evaluation set size: %d", len(val_split))

    results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1. Base Llama
    # ------------------------------------------------------------------
    base_cfg = config["base_model"]
    if base_cfg["enabled"]:
        model, tokenizer = load_base_model(base_cfg)
        label_ids = get_label_token_ids(tokenizer)

        # Determine number of hidden layers from config
        num_layers = model.config.num_hidden_layers

        res = evaluate_model(
            model, tokenizer, val_split, label_ids,
            model_name="Base Llama",
            num_layers=num_layers,
            collect_exit_layers=False,
        )
        results.append(res)
        logger.info("Base Llama accuracy: %.4f", res["accuracy"])

        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 2. EELlamma (confidence-based early exit)
    # ------------------------------------------------------------------
    ee_cfg = config["eellama_model"]
    if ee_cfg["enabled"]:
        model, tokenizer = load_eellama_model(ee_cfg)
        label_ids = get_label_token_ids(tokenizer)
        num_layers = model.config.num_hidden_layers

        res = evaluate_model(
            model, tokenizer, val_split, label_ids,
            model_name="EELlamma (conf.)",
            num_layers=num_layers,
            collect_exit_layers=True,
        )
        results.append(res)
        logger.info("EELlamma accuracy: %.4f", res["accuracy"])

        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3. SkipDecode EELlamma
    # ------------------------------------------------------------------
    sd_cfg = config["skipdecode_model"]
    if sd_cfg["enabled"]:
        model, tokenizer = load_skipdecode_model(sd_cfg)
        label_ids = get_label_token_ids(tokenizer)
        num_layers = model.config.num_hidden_layers

        res = evaluate_model(
            model, tokenizer, val_split, label_ids,
            model_name="SkipDecode EELlamma",
            num_layers=num_layers,
            collect_exit_layers=True,
        )
        results.append(res)
        logger.info("SkipDecode accuracy: %.4f", res["accuracy"])

        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print_results_table(results)

    # Detailed per-model breakdown
    for r in results:
        print(f"[{r['model']}]")
        print(f"  Accuracy          : {r['accuracy']:.4f}  ({r['correct']} / {r['total']})")
        print(f"  Mean latency      : {r['latency_mean_ms']:.1f} ms  ± {r['latency_std_ms']:.1f} ms")
        print(f"  Total eval time   : {r['latency_total_s']:.1f} s")
        print(f"  Avg active layers : {r['exit_layer_mean']:.2f}  ± {r['exit_layer_std']:.2f}")
        print(f"  Compute fraction  : {r['compute_fraction']:.4f}")
        print(f"  Compute reduction : {r['compute_reduction_pct']:.1f}%")
        print()
