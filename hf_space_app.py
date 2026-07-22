"""
Hugging Face Space App for BanglaSupport-LLM.

Deploy directly to Hugging Face Spaces (Gradio SDK - Free CPU Basic).
Uses the fine-tuned adapter uploaded to: mrshibly/bangla-support-qwen3-8b
"""

import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
ADAPTER_HUB_ID = "mrshibly/bangla-support-qwen3-8b"

SYSTEM_PROMPT = (
    "তুমি একজন সহায়ক বাংলা ই-কমার্স গ্রাহক সেবা সহকারী। "
    "গ্রাহকদের প্রশ্নের উত্তর পেশাদার, বিনয়ী এবং সংক্ষিপ্তভাবে দাও।"
)

# Load model and tokenizer for CPU Basic
print("Loading base model and Hugging Face adapter...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_HUB_ID)
model.eval()
print("✓ Model and adapter loaded successfully.")


def respond(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": message})
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


demo = gr.ChatInterface(
    fn=respond,
    title="🇧🇩 BanglaSupport-LLM Customer Assistant",
    description="Fine-tuned Qwen2.5-7B on 25k Bangla e-commerce instruction pairs using Unsloth QLoRA.",
    examples=[
        "আমার অর্ডারটির ডেলিভারি স্ট্যাটাস কীভাবে চেক করব?",
        "পণ্য ফেরত দেওয়ার নিয়ম ও শর্তাবলী কি কি?",
        "আপনার পেমেন্ট পদ্ধতি কি কি রয়েছে?",
    ],
    theme="soft",
)

if __name__ == "__main__":
    demo.launch()
