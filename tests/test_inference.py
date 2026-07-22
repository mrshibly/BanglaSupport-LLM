"""
Inference test script for BanglaSupport-LLM.

Usage:
    python tests/test_inference.py
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
ADAPTER_ID = "mrshibly/bangla-support-qwen3-8b"

SYSTEM_PROMPT = (
    "তুমি একজন সহায়ক বাংলা ই-কমার্স গ্রাহক সেবা সহকারী। "
    "গ্রাহকদের প্রশ্নের উত্তর পেশাদার, বিনয়ী এবং সংক্ষিপ্তভাবে দাও।"
)


def main():
    print("==================================================")
    print(f"Loading Model Adapter: {ADAPTER_ID}")
    print("==================================================")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    model.eval()
    print("✓ Model Loaded Successfully!\n")

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
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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
    print("✓ Test Complete!")
    print("==================================================")


if __name__ == "__main__":
    main()
