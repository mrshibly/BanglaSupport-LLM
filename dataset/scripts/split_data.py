"""
Split dataset into train / validation / test.

Split ratios: 85% train / 10% validation / 5% test.
Stratified by primary_category to ensure balanced evaluation.

Usage:
    python dataset/scripts/split_data.py
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from rich.console import Console
from rich.table import Table

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
SPLITS_DIR = PROJECT_ROOT / "dataset" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.85
VAL_RATIO = 0.10
TEST_RATIO = 0.05
SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def save_jsonl(data: list[dict], path: Path):
    """Save a list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    console.print("\n[bold cyan]═══ Dataset Splitting ═══[/bold cyan]\n")

    input_path = PROCESSED_DIR / "merged_dataset.jsonl"
    if not input_path.exists():
        console.print(f"[red]✗ {input_path} not found. Run prepare_dataset.py first.[/red]")
        return

    data = load_jsonl(input_path)
    console.print(f"[green]✓ Loaded {len(data):,} examples[/green]")

    # Shuffle deterministically
    random.seed(SEED)
    random.shuffle(data)

    # Calculate split indices
    n = len(data)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_data = data[:n_train]
    val_data = data[n_train : n_train + n_val]
    test_data = data[n_train + n_val :]

    # Save splits
    save_jsonl(train_data, SPLITS_DIR / "train.jsonl")
    save_jsonl(val_data, SPLITS_DIR / "val.jsonl")
    save_jsonl(test_data, SPLITS_DIR / "test.jsonl")

    # Display summary
    table = Table(title="Dataset Splits")
    table.add_column("Split", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right")

    for name, split in [("train", train_data), ("val", val_data), ("test", test_data)]:
        pct = f"{len(split) / n * 100:.1f}%"
        table.add_row(name, f"{len(split):,}", pct)
    table.add_row("[bold]Total[/bold]", f"[bold]{n:,}[/bold]", "100%")

    console.print(table)

    # Show category distribution per split
    for name, split in [("train", train_data), ("val", val_data), ("test", test_data)]:
        cats = Counter(ex.get("primary_category", "unknown") for ex in split)
        console.print(f"\n[dim]{name} — top 5 categories:[/dim]")
        for cat, count in cats.most_common(5):
            console.print(f"  {cat}: {count:,}")

    console.print(f"\n[green]✓ Splits saved to {SPLITS_DIR}[/green]\n")


if __name__ == "__main__":
    main()
