"""
Fine-tune Qwen3-8B on Bangla customer support data using QLoRA + Unsloth.

Usage:
    python training/train.py
    python training/train.py --max_steps 50   # smoke test
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

import yaml


def load_config(config_path: str = "training/configs/qlora_qwen3_8b.yaml") -> dict:
    """Load training configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL dataset."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def format_chat_template(example: dict, tokenizer) -> str:
    """Format an example using the Qwen3 chat template.

    Converts our instruction/input/output format to the Qwen3 chat format:
        <|im_start|>system\nYou are a helpful Bangla e-commerce customer support assistant.<|im_end|>
        <|im_start|>user\n{instruction + input}<|im_end|>
        <|im_start|>assistant\n{output}<|im_end|>
    """
    system_prompt = (
        "তুমি একজন সহায়ক বাংলা ই-কমার্স গ্রাহক সেবা সহকারী। "
        "গ্রাহকদের প্রশ্নের উত্তর পেশাদার, বিনয়ী এবং সংক্ষিপ্তভাবে দাও।"
    )

    user_msg = example["instruction"]
    if example.get("input"):
        user_msg = f"{user_msg}\n\n{example['input']}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": example["output"]},
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-8B for Bangla support")
    parser.add_argument("--config", default="training/configs/qlora_qwen3_8b.yaml")
    parser.add_argument("--max_steps", type=int, default=-1, help="Override max steps (for smoke test)")
    args = parser.parse_args()

    config = load_config(args.config)

    # ── Step 1: Load model with Unsloth FastLanguageModel ──
    print("Loading 4-bit model via Unsloth...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"]["name"],
        max_seq_length=config["model"]["max_seq_length"],
        load_in_4bit=True,
    )
    print("✓ Step 1: Model loaded successfully.")

    # ── Step 2: Apply LoRA with Unsloth ────────────────────────────
    print("Applying LoRA adapters...")
    lora_cfg = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config["training"]["seed"],
    )
    print("✓ Step 2: Unsloth PEFT LoRA applied successfully.")

    # ── Step 3: Load and format dataset ─────────────────────────
    print("Loading dataset...")
    from datasets import Dataset

    train_raw = load_jsonl(config["data"]["train_path"])[:25000]
    val_raw = load_jsonl(config["data"]["val_path"])[:500]

    if args.max_steps > 0:
        print(f"⚡ Smoke test mode active (--max_steps {args.max_steps}): slicing dataset for fast execution.")
        train_raw = train_raw[: max(args.max_steps * 10, 500)]
        val_raw = val_raw[:100]

    print(f"  Train: {len(train_raw):,} examples")
    print(f"  Val:   {len(val_raw):,} examples")

    # Format with chat template
    train_texts = [format_chat_template(ex, tokenizer) for ex in train_raw]
    val_texts = [format_chat_template(ex, tokenizer) for ex in val_raw]

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})
    print("✓ Step 3: Dataset loaded and formatted successfully.")

    # ── Step 4: Configure trainer ───────────────────────────────
    print("Configuring SFTTrainer...")
    from trl import SFTTrainer, SFTConfig

    t_cfg = config["training"]

    training_args = SFTConfig(
        output_dir=config["output"]["dir"],
        num_train_epochs=t_cfg["epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        weight_decay=t_cfg["weight_decay"],
        fp16=t_cfg["fp16"],
        bf16=t_cfg["bf16"],
        logging_steps=t_cfg["logging_steps"],
        save_strategy="no" if args.max_steps > 0 else t_cfg["save_strategy"],
        eval_strategy="no" if args.max_steps > 0 else t_cfg["eval_strategy"],
        eval_steps=t_cfg["eval_steps"],
        seed=t_cfg["seed"],
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        report_to="none",
        run_name="bangla-support-qwen3-8b",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    # ── Step 5: Train ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    trainer.train()

    # ── Step 6: Save adapter ────────────────────────────────────
    adapter_path = Path(config["output"]["dir"]) / "final_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n✓ Adapter saved to {adapter_path}")

    # ── Step 7: Log final metrics ───────────────────────────────
    if args.max_steps <= 0:
        metrics = trainer.evaluate()
        print("\nFinal evaluation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        # ── Step 8: Push to HuggingFace Hub & GitHub ─────────────────
        hub_id = config["output"].get("hub_model_id")
        if hub_id:
            print(f"\n🚀 Pushing merged model to HuggingFace Hub: {hub_id}...")
            try:
                model.push_to_hub_merged(hub_id, tokenizer, save_method="merged_16bit")
                print(f"✓ Successfully pushed merged model to Hugging Face Hub ({hub_id})!")
            except Exception as e:
                print(f"⚠️ HuggingFace Merged Push Warning: {e}")
                try:
                    model.push_to_hub(hub_id)
                    tokenizer.push_to_hub(hub_id)
                    print(f"✓ Pushed adapter to Hugging Face Hub ({hub_id})!")
                except Exception as e2:
                    print(f"⚠️ Failed to push adapter to HF Hub: {e2}")

        print("\n📦 Auto-pushing repository updates to GitHub...")
        import subprocess
        try:
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Auto-commit: QLoRA training completed & metrics logged"], check=True)
            subprocess.run(["git", "push", "origin", "main"], check=True)
            print("✓ Successfully pushed latest changes to GitHub (origin/main)!")
        except Exception as ge:
            print(f"⚠️ Git auto-push warning: {ge}")
    else:
        print("\n✓ Smoke test training completed successfully!")


if __name__ == "__main__":
    main()
