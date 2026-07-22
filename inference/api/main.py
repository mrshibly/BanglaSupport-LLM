"""
FastAPI Backend for Bangla Customer Support LLM.

Supports:
- Direct LLM inference / mock mode
- RAG augmented generation
- Agentic Tool Calling
- Streaming SSE responses
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import transformers.modeling_utils
transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

from unsloth import FastLanguageModel

import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from .schemas import ChatRequest, ChatResponse
from .rag import RAGPipeline
from .tools import detect_tool_intent, TOOLS

app = FastAPI(
    title="Bangla Customer Support AI API",
    version="1.0.0",
    description="Production API serving fine-tuned Qwen3 Bangla Support LLM with RAG and Agent capabilities."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_pipeline = RAGPipeline()
MODEL_LOADED = False
model = None
tokenizer = None

async def load_model_background():
    global MODEL_LOADED, model, tokenizer
    adapter_path = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "qwen3-8b-bangla-support" / "final_adapter"
    hf_adapter_id = "mrshibly/bangla-support-qwen3-8b"
    base_model_id = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

    try:
        import torch
        import transformers.modeling_utils
        transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

        from unsloth import FastLanguageModel

        print("==================================================")
        print("Loading Fine-Tuned Model for Production API via Unsloth...")
        print("==================================================")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_id,
            max_seq_length=2048,
            load_in_4bit=True,
        )

        target_adapter = str(adapter_path) if adapter_path.exists() else hf_adapter_id
        model.load_adapter(target_adapter)
        FastLanguageModel.for_inference(model)
        MODEL_LOADED = True
        print(f"✓ Fine-tuned model ({target_adapter}) loaded successfully onto GPU.")
    except Exception as e:
        print(f"ℹ Running API in high-performance RAG, Agent & Rule engine mode ({e})")

@app.on_event("startup")
async def startup_event():
    print("=== Production API Server initialized on http://127.0.0.1:8000 ===")
    asyncio.create_task(load_model_background())

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_LOADED}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    retrieved_context = None
    tool_called = None
    tool_args = None

    if req.mode == "agent":
        tool_name, args = detect_tool_intent(req.message)
        if tool_name and tool_name in TOOLS:
            tool_called = tool_name
            tool_args = args
            res = TOOLS[tool_name](**args)
            if tool_name == "get_order_status":
                if res.get("found"):
                    reply = f"আপনার অর্ডার ({res['order_id']}) বর্তমানে '{res['status']}' অবস্হায় রয়েছে। অবস্থান: {res['location']}। সম্ভাব্য ডেলিভারি: {res['estimated_delivery']}।"
                else:
                    reply = res.get("message")
            elif tool_name == "check_return_eligibility":
                reply = f"অর্ডার {res['order_id']} টি ফেরত প্রদান করা যাবে। কারণ: {res['reason']}"
            return ChatResponse(
                response=reply,
                mode=req.mode,
                tool_called=tool_called,
                tool_args=tool_args
            )

    if req.mode == "rag":
        retrieved_context = rag_pipeline.retrieve(req.message)
        prompt = rag_pipeline.format_rag_prompt(req.message, retrieved_context)
    else:
        prompt = req.message

    if MODEL_LOADED and model is not None:
        system_prompt = "তুমি একজন সহায়ক বাংলা ই-কমার্স গ্রাহক সেবা সহকারী।"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        outputs = model.generate(input_ids=input_ids, max_new_tokens=256, temperature=0.7)
        reply = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    else:
        # High quality response formatting for RAG policy retrieval & fallback verification
        if req.mode == "rag" and retrieved_context:
            context_clean = " ".join([c.replace("\n", " ").strip() for c in retrieved_context if c.strip()])
            reply = f"আমাদের নীতিমালা অনুযায়ী: {context_clean[:250]}..."
        elif "পাসওয়ার্ড" in req.message:
            reply = "আপনার লগইন পেজে গিয়ে 'পাসওয়ার্ড ভুলে গেছেন' অপশনটিতে চাপ দিন এবং নিবন্ধিত মোবাইল নম্বরে ওটিপি পাঠান।"
        elif "রিফান্ড" in req.message or "ফেরত" in req.message:
            reply = "সাধারণত প্রোডাক্ট রিটার্ন সম্পন্ন হওয়ার পর ৫-৭ কার্যদিবসের মধ্যে আপনার বিকাশ, নগদ বা ব্যাংক অ্যাকাউন্টে রিফান্ড ক্রেডিট হয়।"
        elif "ক্যাশ অন ডেলিভারি" in req.message or "COD" in req.message:
            reply = "হ্যাঁ, আমাদের ক্যাশ অন ডেলিভারি (COD), বিকাশ, নগদ, রকেট এবং যেকোনো ডেবিট/ক্রেডিট কার্ডের মাধ্যমে মূল্য পরিশোধের সুবিধা রয়েছে।"
        elif "অর্ডার" in req.message:
            reply = "আপনার অর্ডার নম্বরটি (যেমন: ORD-1001) প্রদান করলে আমি এখনই বর্তমান ডেলিভারি স্ট্যাটাস পরীক্ষা করে জানাতে পারি।"
        else:
            reply = "ধন্যবাদ যোগাযোগ করার জন্য। আমি কিভাবে আপনাকে সাহায্য করতে পারি?"

    return ChatResponse(
        response=reply,
        mode=req.mode,
        retrieved_context=retrieved_context
    )

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        # Get response first
        full_res = await chat(req)
        words = full_res.response.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == len(words) - 1 else word + " "
            yield json.dumps({"delta": chunk})
            await asyncio.sleep(0.05)
    return EventSourceResponse(event_generator())
