"""
FastAPI Backend for Bangla Customer Support LLM.

Supports:
- Direct LLM inference / mock mode
- RAG augmented generation
- Agentic Tool Calling
- Streaming SSE responses
"""

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

@app.on_event("startup")
async def startup_event():
    global MODEL_LOADED, model, tokenizer
    merged_path = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "qwen3-8b-bangla-support" / "merged"
    if merged_path.exists():
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(merged_path),
                max_seq_length=2048,
                load_in_4bit=True
            )
            FastLanguageModel.for_inference(model)
            MODEL_LOADED = True
            print("✓ Fine-tuned model loaded successfully.")
        except Exception as e:
            print(f"⚠ Could not load fine-tuned model: {e}. Falling back to rule/mock mode.")
    else:
        print("ℹ No fine-tuned checkpoint found yet. Running in API fallback/mock mode.")

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
        # High quality fallback responses for immediate UI verification
        if "পাসওয়ার্ড" in req.message:
            reply = "আপনার লগইন পেজে গিয়ে 'পাসওয়ার্ড ভুলে গেছেন' অপশনটিতে চাপ দিন এবং নিবন্ধিত মোবাইল নম্বরে ওটিপি পাঠান।"
        elif "রিফান্ড" in req.message:
            reply = "সাধারণত রিটার্ন সম্পন্ন হওয়ার পর ৫-৭ কার্যদিবসের মধ্যে আপনার বিকাশ বা ব্যাংক অ্যাকাউন্টে রিফান্ড ক্রেডিট হয়।"
        elif "অর্ডার" in req.message:
            reply = "আপনার অর্ডার নম্বরটি প্রদান করলে আমি এখনই বর্তমান স্ট্যাটাস পরীক্ষা করে জানাতে পারি।"
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
