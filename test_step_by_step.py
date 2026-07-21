import sys, traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("1. Loading yaml...", flush=True)
import yaml
with open("training/configs/qlora_qwen3_8b.yaml") as f:
    config = yaml.safe_load(f)
print("✓ Yaml loaded.", flush=True)

print("2. Importing torch & transformers...", flush=True)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
print("✓ Imports done.", flush=True)

print("3. Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
print("✓ Tokenizer loaded.", flush=True)

print("4. Loading pre-downloaded 4-bit model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
print("✓ Model loaded on CPU.", flush=True)

print("5. Applying PEFT...", flush=True)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["lora_alpha"],
    lora_dropout=config["lora"]["lora_dropout"],
    target_modules=config["target_modules"] if "target_modules" in config else config["lora"]["target_modules"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
print("✓ PEFT applied.", flush=True)

print("6. Loading dataset...", flush=True)
from datasets import Dataset, load_dataset
import json
with open("dataset/splits/train.jsonl", "r", encoding="utf-8") as f:
    train_raw = [json.loads(line) for idx, line in enumerate(f) if idx < 100]
print(f"✓ {len(train_raw)} dataset rows loaded.", flush=True)

print("7. Formatting template...", flush=True)
def format_chat_template(example, tokenizer):
    system_prompt = "তুমি একজন সহায়ক বাংলা ই-কমার্স গ্রাহক সেবা সহকারী।"
    user_msg = example["instruction"]
    if example.get("input"):
        user_msg = f"{user_msg}\n\n{example['input']}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": example["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

train_texts = [format_chat_template(ex, tokenizer) for ex in train_raw]
train_dataset = Dataset.from_dict({"text": train_texts})
print("✓ Dataset formatted.", flush=True)

print("8. Importing SFTTrainer & SFTConfig...", flush=True)
from trl import SFTTrainer, SFTConfig
training_args = SFTConfig(
    output_dir=config["output"]["dir"],
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    fp16=False,
    use_cpu=True,
    logging_steps=1,
    max_steps=5,
    report_to="none"
)
print("✓ TrainingArguments created.", flush=True)

print("9. Creating SFTTrainer...", flush=True)
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    args=training_args,
)
print("✓ SFTTrainer created.", flush=True)

print("10. Starting trainer.train()...", flush=True)
trainer.train()
print("✓ ALL DONE SUCCESSFULLY!", flush=True)
