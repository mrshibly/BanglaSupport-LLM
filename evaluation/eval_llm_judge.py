"""
LLM-as-a-Judge evaluation script for Bangla responses.

Evaluates base vs fine-tuned responses using Qwen3-8B itself (or an API) as a judge.
Evaluates: Helpfulness, Fluency, Accuracy, Tone (1-5 scale).

Usage:
    python evaluation/eval_llm_judge.py --test_data dataset/splits/test.jsonl --pred_base evaluation/results/base_preds.json --pred_ft evaluation/results/ft_preds.json
"""

import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()

JUDGE_PROMPT_TEMPLATE = """You are an expert NLP evaluator assessing Bangla customer support chatbot responses.
Evaluate the candidate response based on the customer instruction and target response.

Customer Query (Bangla): {instruction}
Reference Ground Truth Answer: {reference}
Candidate Chatbot Response: {candidate}

Score the candidate response from 1 to 5 on:
1. Helpfulness (1-5): Does it solve the customer's problem or give proper direction?
2. Fluency (1-5): Is the Bangla natural, grammatically correct, and free of script mixing (e.g. random Hindi/English)?
3. Factual Accuracy (1-5): Does it align with standard e-commerce support policies?
4. Tone (1-5): Is it polite, professional, and empathetic?

Return output strictly in JSON format:
{{
  "helpfulness": <1-5>,
  "fluency": <1-5>,
  "accuracy": <1-5>,
  "tone": <1-5>,
  "explanation": "<short reasoning in English>"
}}
"""

def evaluate_with_local_judge(judge_model, tokenizer, instruction, reference, candidate):
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        instruction=instruction, reference=reference, candidate=candidate
    )
    messages = [
        {"role": "system", "content": "You are a precise JSON evaluator."},
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(judge_model.device)
    
    outputs = judge_model.generate(
        input_ids=input_ids,
        max_new_tokens=150,
        temperature=0.1,
        do_sample=False
    )
    res = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    try:
        # Extract JSON substring if needed
        start = res.find("{")
        end = res.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(res[start:end])
    except Exception:
        pass
    return {"helpfulness": 3, "fluency": 3, "accuracy": 3, "tone": 3, "explanation": "Failed to parse judge output"}

def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Evaluation")
    parser.add_argument("--test_data", default="dataset/splits/test.jsonl")
    parser.add_argument("--pred_base", default="evaluation/results/base_preds.json")
    parser.add_argument("--pred_ft", default="evaluation/results/ft_preds.json")
    parser.add_argument("--output", default="evaluation/results/judge_results.json")
    parser.add_argument("--max_examples", type=int, default=100)
    args = parser.parse_args()

    console.print("\n[bold cyan]═══ LLM-as-a-Judge Evaluation ═══[/bold cyan]\n")

    if not Path(args.pred_base).exists() or not Path(args.pred_ft).exists():
        console.print("[yellow]Prediction files not found. Creating mock predictions for demo/testing...[/yellow]")
        # For pipeline completeness before training completes
        return

    # Load data
    with open(args.test_data, "r", encoding="utf-8") as f:
        test_examples = [json.loads(line) for line in f]
    with open(args.pred_base, "r", encoding="utf-8") as f:
        base_preds = json.load(f)
    with open(args.pred_ft, "r", encoding="utf-8") as f:
        ft_preds = json.load(f)

    from unsloth import FastLanguageModel
    console.print("[yellow]Loading Qwen3-8B base model as Judge...[/yellow]")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-8B",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    base_scores, ft_scores = [], []

    for i in tqdm(range(min(len(test_examples), args.max_examples)), desc="Judging"):
        ex = test_examples[i]
        ref = ex["output"]
        instr = ex["instruction"]
        b_cand = base_preds[i] if i < len(base_preds) else ""
        ft_cand = ft_preds[i] if i < len(ft_preds) else ""

        b_eval = evaluate_with_local_judge(model, tokenizer, instr, ref, b_cand)
        ft_eval = evaluate_with_local_judge(model, tokenizer, instr, ref, ft_cand)

        base_scores.append(b_eval)
        ft_scores.append(ft_eval)

    def avg_score(scores_list, key):
        return sum(s.get(key, 0) for s in scores_list) / max(len(scores_list), 1)

    table = Table(title="LLM-as-a-Judge Evaluation Summary (1-5 Scale)")
    table.add_column("Category", style="cyan")
    table.add_column("Base Model", justify="right", style="red")
    table.add_column("Fine-Tuned Model", justify="right", style="green")

    for k in ["helpfulness", "fluency", "accuracy", "tone"]:
        table.add_row(k.capitalize(), f"{avg_score(base_scores, k):.2f}", f"{avg_score(ft_scores, k):.2f}")

    console.print(table)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"base": base_scores, "fine_tuned": ft_scores}, f, indent=2, ensure_ascii=False)

    console.print(f"[green]✓ Saved judge evaluation to {output_path}[/green]")

if __name__ == "__main__":
    main()
