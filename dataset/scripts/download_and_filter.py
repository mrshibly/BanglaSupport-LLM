"""
Download and filter Bangla-Instruct dataset for customer support.

Filters 342K Bangla instruction pairs down to ~20K customer-support-relevant
examples using keyword matching across 12 intent categories.

Usage:
    python dataset/scripts/download_and_filter.py
"""

import json
import os
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Intent categories with Bangla keywords
# ============================================================
SUPPORT_KEYWORDS: dict[str, list[str]] = {
    "order_tracking": [
        "অর্ডার", "ট্র্যাক", "ডেলিভারি", "শিপমেন্ট", "কুরিয়ার",
        "পৌঁছায়নি", "পাইনি", "পৌঁছেনি", "কোথায়", "স্ট্যাটাস",
        "ট্র্যাকিং", "চালান", "প্রেরণ",
    ],
    "refund": [
        "রিফান্ড", "ফেরত", "টাকা ফেরত", "রিটার্ন", "মানি ব্যাক",
        "ফিরিয়ে", "প্রত্যর্পণ",
    ],
    "cancellation": [
        "ক্যানসেল", "বাতিল", "অর্ডার বাতিল", "বাতিলকরণ",
    ],
    "payment": [
        "পেমেন্ট", "টাকা", "বিকাশ", "নগদ", "কার্ড", "পরিশোধ",
        "লেনদেন", "পেমেন্ট ব্যর্থ", "চার্জ", "ব্যালেন্স",
        "ক্রেডিট", "ডেবিট", "ট্রানজেকশন",
    ],
    "account": [
        "অ্যাকাউন্ট", "পাসওয়ার্ড", "লগইন", "নিবন্ধন", "সাইন আপ",
        "প্রোফাইল", "ইমেইল", "ফোন নম্বর", "ভেরিফিকেশন",
    ],
    "product": [
        "প্রোডাক্ট", "পণ্য", "ভুল পণ্য", "ত্রুটিপূর্ণ", "মান",
        "নকল", "ভাঙা", "ক্ষতিগ্রস্ত", "সাইজ", "রঙ",
    ],
    "complaint": [
        "অভিযোগ", "সমস্যা", "সাহায্য", "সমাধান", "খারাপ",
        "বিরক্ত", "হতাশ", "অসন্তুষ্ট", "ঠকানো",
    ],
    "general_service": [
        "সেবা", "গ্রাহক", "সাপোর্ট", "জানাতে", "জানান",
        "কাস্টমার", "হেল্পলাইন", "যোগাযোগ",
    ],
    "address": [
        "ঠিকানা", "পরিবর্তন", "আপডেট", "শিপিং ঠিকানা",
        "ডেলিভারি ঠিকানা",
    ],
    "voucher": [
        "ভাউচার", "কুপন", "ডিসকাউন্ট", "অফার", "প্রমো",
        "প্রমোশন", "কোড", "ছাড়",
    ],
    "how_to": [
        "কিভাবে", "কীভাবে", "উপায়", "পদ্ধতি", "নিয়ম",
        "প্রক্রিয়া", "ধাপ",
    ],
    "query": [
        "কেন", "কোথায়", "কখন", "কতদিন", "কত টাকা",
        "কত সময়", "কবে", "কী", "কি",
    ],
}


def classify_example(text: str) -> list[str]:
    """Return list of matching intent categories for a text."""
    text_lower = text.lower()
    matched = []
    for category, keywords in SUPPORT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(category)
    return matched


def main():
    console.print("\n[bold cyan]═══ Bangla-Instruct Dataset Download & Filter ═══[/bold cyan]\n")

    # ── Step 1: Download ────────────────────────────────────────
    console.print("[yellow]Downloading md-nishat-008/Bangla-Instruct...[/yellow]")
    ds = load_dataset("md-nishat-008/Bangla-Instruct", split="train")
    console.print(f"[green]✓ Downloaded {len(ds):,} examples[/green]\n")

    # ── Step 2: Filter ──────────────────────────────────────────
    console.print("[yellow]Filtering for customer support relevance...[/yellow]")

    filtered_examples = []
    category_counts: Counter = Counter()

    for example in tqdm(ds, desc="Filtering"):
        instruction = example.get("instruction", "")
        response = example.get("response", "")
        combined_text = f"{instruction} {response}"

        categories = classify_example(combined_text)
        if categories:
            filtered_examples.append({
                "instruction": instruction,
                "input": "",
                "output": response,
                "source": "bangla_instruct",
                "categories": categories,
                "primary_category": categories[0],
            })
            for cat in categories:
                category_counts[cat] += 1

    console.print(f"\n[green]✓ Filtered: {len(filtered_examples):,} / {len(ds):,} examples[/green]\n")

    # ── Step 3: Show distribution ───────────────────────────────
    table = Table(title="Intent Category Distribution")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("% of Filtered", justify="right")

    for cat, count in category_counts.most_common():
        pct = f"{count / len(filtered_examples) * 100:.1f}%"
        table.add_row(cat, f"{count:,}", pct)

    console.print(table)

    # ── Step 4: Save ────────────────────────────────────────────
    output_path = RAW_DIR / "bangla_instruct_filtered.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in filtered_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    console.print(f"\n[green]✓ Saved to {output_path}[/green]")
    console.print(f"  Total examples: {len(filtered_examples):,}")

    # ── Step 5: Save sample for manual review ───────────────────
    sample_path = RAW_DIR / "sample_for_review.jsonl"
    import random
    random.seed(42)
    sample = random.sample(filtered_examples, min(100, len(filtered_examples)))
    with open(sample_path, "w", encoding="utf-8") as f:
        for ex in sample:
            f.write(json.dumps(ex, ensure_ascii=False, indent=2) + "\n")

    console.print(f"[green]✓ Saved 100 samples for manual review to {sample_path}[/green]\n")


if __name__ == "__main__":
    main()
