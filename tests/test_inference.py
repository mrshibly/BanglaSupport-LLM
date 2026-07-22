"""
Inference test script for BanglaSupport-LLM using Unsloth + monkeypatched allocator.

Usage:
    python tests/test_inference.py
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Monkeypatch transformers caching_allocator_warmup BEFORE unsloth import
import transformers.modeling_utils
transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

import torch
from unsloth import FastLanguageModel

BASE_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
LOCAL_ADAPTER_PATH = "checkpoints/qwen3-8b-bangla-support/final_adapter"
HUB_ADAPTER_ID = "mrshibly/bangla-support-qwen3-8b"

SYSTEM_PROMPT = (
    "তুমি একজন সহায়ক বাংলা ই-কমার্স গ্রাহক সেবা সহকারী। "
    "গ্রাহকদের প্রশ্নের উত্তর পেশাদার, বিনয়ী এবং সংক্ষিপ্তভাবে দাও।"
)


def main():
    import os
    adapter_target = LOCAL_ADAPTER_PATH if os.path.exists(LOCAL_ADAPTER_PATH) else HUB_ADAPTER_ID

    print("==================================================")
    print(f"Loading Base Model: {BASE_MODEL_ID}")
    print(f"Attaching LoRA Adapter: {adapter_target}")
    print("==================================================")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    model.load_adapter(adapter_target)
    FastLanguageModel.for_inference(model)
    print("✓ Fine-tuned Model Loaded Successfully onto GPU!\n")

    test_questions = [
        "আমার অর্ডারটি ৩ দিন ধরে পেন্ডিং আছে, ডেলিভারি কখন পাব?",
        "পণ্য ডেলিভারির পর পছন্দ না হলে কিভাবে রিফান্ড বা ফেরত পাওয়া যাবে?",
        "আপনাদের কি ক্যাশ অন ডেলিভারি (COD) সুবিধা আছে?",
    ]

    for idx, q in enumerate(test_questions, 1):
        print(f"--------------------------------------------------")
        print(f"📌 Question {idx}: {q}")
        print(f"--------------------------------------------------")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"🤖 Response:\n{response.strip()}\n")

    print("==================================================")
    print("✓ Local Inference Test Complete!")
    print("==================================================")


if __name__ == "__main__":
    main()
