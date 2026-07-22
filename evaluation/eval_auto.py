"""
Automatic evaluation: BLEU, ROUGE-L, BERTScore.

Compares base Qwen3-8B vs fine-tuned model on held-out test set.

Usage:
    python evaluation/eval_auto.py --model checkpoints/qwen3-8b-bangla-support/final_adapter
    python evaluation/eval_auto.py --model Qwen/Qwen3-8B   # baseline
"""

import argparse
import json
import sys
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import nltk
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rich.console import Console
from rich.table import Table
from rouge_score import rouge_scorer
from tqdm import tqdm

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL dataset."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def generate_responses(model_path: str, test_data: list[dict], max_examples: int = 500) -> list[str]:
    """Generate model responses for test examples."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    console.print(f"[yellow]Loading model: {model_path}[/yellow]")
    base_model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    system_prompt = (
        "তুমি একজন সহায়ক বাংলা ই-কমার্স গ্রাহক সেবা সহকারী। "
        "গ্রাহকদের প্রশ্নের উত্তর পেশাদার, বিনয়ী এবং সংক্ষিপ্তভাবে দাও।"
    )

    predictions = []
    for example in tqdm(test_data[:max_examples], desc="Generating"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["instruction"]},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        predictions.append(response.strip())

    return predictions


def compute_metrics(references: list[str], predictions: list[str]) -> dict:
    """Compute BLEU, ROUGE-L, and BERTScore."""
    # Ensure NLTK data is available
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    results = {}

    # ── BLEU ────────────────────────────────────────────────────
    console.print("[yellow]Computing BLEU...[/yellow]")
    smoother = SmoothingFunction().method1
    bleu_scores = []
    for ref, pred in zip(references, predictions):
        ref_tokens = list(ref)  # Character-level for Bangla (no word boundaries)
        pred_tokens = list(pred)
        if len(pred_tokens) == 0:
            bleu_scores.append(0.0)
            continue
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother)
        bleu_scores.append(score)

    results["bleu"] = sum(bleu_scores) / len(bleu_scores)

    # ── ROUGE-L ─────────────────────────────────────────────────
    console.print("[yellow]Computing ROUGE-L...[/yellow]")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_scores = []
    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        rouge_scores.append(score["rougeL"].fmeasure)

    results["rouge_l"] = sum(rouge_scores) / len(rouge_scores)

    # ── BERTScore ───────────────────────────────────────────────
    console.print("[yellow]Computing BERTScore (using bangla-bert-base)...[/yellow]")
    P, R, F1 = bert_score(
        predictions,
        references,
        model_type="sagorsarker/bangla-bert-base",
        lang="bn",
        verbose=False,
    )
    results["bertscore_precision"] = P.mean().item()
    results["bertscore_recall"] = R.mean().item()
    results["bertscore_f1"] = F1.mean().item()

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Bangla support model")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF ID")
    parser.add_argument("--test_data", type=str, default="dataset/splits/test.jsonl")
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--output", type=str, default="evaluation/results/eval_auto.json")
    args = parser.parse_args()

    console.print("\n[bold cyan]═══ Automatic Evaluation ═══[/bold cyan]\n")

    # Load test data
    test_data = load_jsonl(args.test_data)
    console.print(f"[green]✓ Loaded {len(test_data):,} test examples[/green]")

    # Generate predictions
    predictions = generate_responses(args.model, test_data, args.max_examples)
    references = [ex["output"] for ex in test_data[: args.max_examples]]

    # Compute metrics
    metrics = compute_metrics(references, predictions)

    # Display results
    table = Table(title=f"Results: {args.model}")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right", style="green")

    for metric, score in metrics.items():
        table.add_row(metric, f"{score:.4f}")

    console.print(table)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model": args.model, "metrics": metrics, "num_examples": len(predictions)},
            f,
            indent=2,
            ensure_ascii=False,
        )
    console.print(f"\n[green]✓ Results saved to {output_path}[/green]\n")


if __name__ == "__main__":
    main()
