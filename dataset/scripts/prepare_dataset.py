"""
Prepare dataset: normalize, merge sources, and deduplicate.

Combines filtered Bangla-Instruct + Aya Bengali into a single clean dataset.

Usage:
    python dataset/scripts/prepare_dataset.py
"""

import hashlib
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    """Basic text normalization for Bangla."""
    import unicodedata

    # NFC normalization (standard for Bangla Unicode)
    text = unicodedata.normalize("NFC", text)
    # Strip whitespace
    text = text.strip()
    # Collapse multiple spaces
    text = " ".join(text.split())
    return text


def compute_hash(instruction: str, output: str) -> str:
    """Compute a hash for deduplication."""
    combined = f"{instruction.strip()}|{output.strip()}"
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def main():
    console.print("\n[bold cyan]═══ Dataset Preparation ═══[/bold cyan]\n")

    all_examples = []

    # ── Step 1: Load all sources ────────────────────────────────
    sources = {
        "bangla_instruct": RAW_DIR / "bangla_instruct_filtered.jsonl",
        "aya_dataset": RAW_DIR / "aya_bengali.jsonl",
    }

    for source_name, path in sources.items():
        if path.exists():
            examples = load_jsonl(path)
            console.print(f"[green]✓ Loaded {len(examples):,} from {source_name}[/green]")
            all_examples.extend(examples)
        else:
            console.print(f"[red]✗ {path} not found — skipping {source_name}[/red]")

    console.print(f"\n[yellow]Total before dedup: {len(all_examples):,}[/yellow]")

    # ── Step 2: Normalize ───────────────────────────────────────
    console.print("[yellow]Normalizing text...[/yellow]")
    for ex in all_examples:
        ex["instruction"] = normalize_text(ex.get("instruction") or "")
        ex["input"] = normalize_text(ex.get("input") or "")
        ex["output"] = normalize_text(ex.get("output") or "")

    # ── Step 3: Filter short/empty ──────────────────────────────
    valid = []
    for ex in all_examples:
        # Skip if instruction or output is too short (< 5 chars)
        if len(ex["instruction"]) < 5 or len(ex["output"]) < 5:
            continue
        # Skip if combined length > 4096 chars (likely noise)
        if len(ex["instruction"]) + len(ex["output"]) > 4096:
            continue
        valid.append(ex)

    console.print(f"[green]✓ After length filter: {len(valid):,}[/green]")

    # ── Step 4: Deduplicate ─────────────────────────────────────
    console.print("[yellow]Deduplicating...[/yellow]")
    seen_hashes: set[str] = set()
    deduped = []
    for ex in tqdm(valid, desc="Dedup"):
        h = compute_hash(ex["instruction"], ex["output"])
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(ex)

    console.print(f"[green]✓ After dedup: {len(deduped):,} (removed {len(valid) - len(deduped):,} duplicates)[/green]")

    # ── Step 5: Show source distribution ────────────────────────
    source_counts = {}
    for ex in deduped:
        src = ex.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    table = Table(title="Final Dataset by Source")
    table.add_column("Source", style="cyan")
    table.add_column("Count", justify="right", style="green")

    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        table.add_row(src, f"{count:,}")
    table.add_row("[bold]Total[/bold]", f"[bold]{len(deduped):,}[/bold]")

    console.print(table)

    # ── Step 6: Save ────────────────────────────────────────────
    output_path = PROCESSED_DIR / "merged_dataset.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in deduped:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    console.print(f"\n[green]✓ Saved merged dataset to {output_path}[/green]\n")


if __name__ == "__main__":
    main()
