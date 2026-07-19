"""
Download Aya Dataset — Bengali subset.

Extracts human-curated Bengali instruction pairs from CohereForAI/aya_dataset.

Usage:
    python dataset/scripts/download_aya.py
"""

import json
from pathlib import Path

from datasets import load_dataset
from rich.console import Console

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def main():
    console.print("\n[bold cyan]═══ Aya Dataset — Bengali Subset Download ═══[/bold cyan]\n")

    # ── Step 1: Download ────────────────────────────────────────
    console.print("[yellow]Downloading CohereForAI/aya_dataset...[/yellow]")
    ds = load_dataset("CohereForAI/aya_dataset", split="train")
    console.print(f"[green]✓ Downloaded {len(ds):,} total examples[/green]")

    # ── Step 2: Filter for Bengali ──────────────────────────────
    console.print("[yellow]Filtering for Bengali...[/yellow]")

    bengali_examples = []
    for example in ds:
        lang = example.get("language", "")
        if lang.lower() in ("bengali", "bangla", "bn"):
            bengali_examples.append({
                "instruction": example.get("inputs", ""),
                "input": "",
                "output": example.get("targets", ""),
                "source": "aya_dataset",
                "categories": ["general"],
                "primary_category": "general",
            })

    console.print(f"[green]✓ Found {len(bengali_examples):,} Bengali examples[/green]")

    # ── Step 3: Save ────────────────────────────────────────────
    output_path = RAW_DIR / "aya_bengali.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in bengali_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    console.print(f"[green]✓ Saved to {output_path}[/green]\n")


if __name__ == "__main__":
    main()
