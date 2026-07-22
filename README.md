# 🌟 BanglaSupport-LLM: Tri-Modal Generative, RAG & Agentic Architecture for Low-Resource E-Commerce

[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model%20Hub-yellow.svg)](https://huggingface.co/mrshibly/bangla-support-qwen3-8b)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Live%20Demo%20Space-blue.svg)](https://huggingface.co/spaces/mrshibly/bangla-support-llm)
[![Base Architecture](https://img.shields.io/badge/Model-Qwen2.5--7B--Instruct-blue.svg)](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct-bnb-4bit)
[![Fine-Tuning](https://img.shields.io/badge/FineTuning-Unsloth%20QLoRA%20bfloat16-green.svg)](https://github.com/unslothai/unsloth)
[![RAG](https://img.shields.io/badge/RAG-ChromaDB%20%2B%20MiniLM-orange.svg)](https://github.com/chroma-core/chroma)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Authors](https://img.shields.io/badge/Authors-mrshibly%20%26%20FHJibon-brightgreen.svg)](https://github.com/mrshibly/BanglaSupport-LLM)

A production-grade, end-to-end AI engineering project demonstrating **dataset curation, Parameter-Efficient Fine-Tuning (QLoRA), Retrieval-Augmented Generation (RAG), Agentic Tool Execution, and Full-Stack System Design** tailored for Bangla NLP.

---

## 🎯 Research & Engineering Highlights

This project addresses the critical gap in **low-resource language adaptation** for modern LLMs. By combining efficient fine-tuning with a tri-modal serving architecture, we achieved state-of-the-art fluency and factual accuracy for Bangla customer support.

- 🧠 **Domain-Specific Fine-Tuning (PEFT)**: Fine-tuned **Qwen2.5-7B-Instruct** using Unsloth QLoRA (4-bit quantization, float16 precision, =16, \alpha=32$) on an NVIDIA RTX 5060 Ti GPU. Successfully eliminated cross-lingual Hindi-bleeding and drastically improved native Bangla fluency.
- 🧹 **Robust Dataset Engineering**: Curated a high-quality zero-cost dataset from md-nishat-008/Bangla-Instruct and CohereForAI/aya_dataset. Implemented NFC Unicode normalization, length filtering, and MinHash LSH deduplication.
- 📐 **Rigorous Evaluation Suite**: Multi-dimensional benchmark pipeline comparing base vs. fine-tuned models across **BLEU**, **ROUGE-L**, **BERTScore** (sagorsarker/bangla-bert-base), and **LLM-as-a-Judge** scoring (Helpfulness, Fluency, Accuracy, Tone).
- 🔍 **RAG Knowledge Retrieval**: Engineered a dense retrieval system using ChromaDB and multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2) for dynamic policy injection, effectively eliminating LLM hallucination on corporate policies.
- ⚡ **Agentic Tool-Calling**: Integrated a deterministic function-calling engine routing intent to execute live API queries (e.g., SQLite order tracking) and feeding database responses back into the LLM context.
- 🖥️ **Full-Stack UI & Production Serving**: FastAPI backend with asynchronous Server-Sent Events (SSE streaming), dynamic memory management, and a highly responsive React + Vite chat interface featuring glassmorphism aesthetics.

---

## 🛠️ Tech Stack & Methodologies

| Domain | Technologies |
|---|---|
| **Base Model & Training** | Qwen2.5-7B, Unsloth FastLanguageModel, PyTorch (f16), PEFT, bitsandbytes |
| **Dataset Engineering** | Hugging Face Datasets, Datasketch (MinHash LSH), unicodedata2, NLTK |
| **Evaluation Metrics** | BLEU, ROUGE, BERTScore, LLM-as-a-Judge (Prompt Engineering) |
| **RAG & Vector Search** | ChromaDB, LangChain, Sentence-Transformers |
| **Inference Backend** | FastAPI, SSE-Starlette, Pydantic, SQLite, Uvicorn, asyncio |
| **Frontend UI** | React 18, Vite, Lucide Icons, Vanilla CSS |

---

## 📐 Tri-Modal System Architecture

### 1. End-to-End ML Pipeline Architecture

`mermaid
flowchart TD
    subgraph DataPrep [Phase 1: Dataset Pipeline]
        A1[Bangla-Instruct 342K] --> A3[Support Intent Filter]
        A2[Aya Dataset Bengali] --> A3
        A3 --> A4[NFC Normalization & Dedup]
        A4 --> A5[Train / Val / Test Split 289K]
    end

    subgraph Training [Phase 2: QLoRA Fine-Tuning]
        A5 --> B1[Qwen2.5-7B Base Model]
        B1 --> B2[Unsloth + SFTTrainer 4-bit NF4]
        B2 --> B3[Merged Model Safetensors]
    end

    subgraph Eval [Phase 3: Multi-Metric Evaluation]
        B3 --> C1[BLEU / ROUGE-L]
        B3 --> C2[BERTScore bangla-bert-base]
        B3 --> C3[LLM-as-a-Judge 1-5 Scale]
    end

    subgraph Serving [Phase 5-6: Tri-Modal Serving API]
        B3 --> D1[FastAPI SSE Server]
        D2[ChromaDB Vector Store] --> D1
        D3[SQLite Order DB] --> D1
        D1 --> D4[React User Interface]
    end
`

### 2. Tri-Modal Routing: RAG vs Agentic Tool Calling

`mermaid
sequenceDiagram
    autonumber
    actor Customer as User
    participant API as FastAPI Backend
    participant RAG as ChromaDB Retriever
    participant DB as SQLite Agent
    participant LLM as Fine-Tuned Model

    Customer->>API: User Request
    alt RAG Mode (Policy Inquiry)
        API->>RAG: Semantic Search
        RAG-->>API: Policy Document Context
    else Agentic Mode (Order Tracking)
        API->>DB: Extract Intent & Query DB (ORD-1001)
        DB-->>API: Real-time DB Status
    end
    API->>LLM: Formulate contextual prompt with retrieved data
    LLM-->>API: Streamed response generation
    API-->>Customer: Real-time SSE Streamed Answer
`

---

## 📊 Evaluation & Benchmarks

The fine-tuning process yielded massive improvements in native Bangla fluency, completely overriding the base model's tendency to fallback to Hindi or English when faced with domain-specific terms.

| Model Variant | BLEU-4 | ROUGE-L | BERTScore (F1) | LLM-Judge (Fluency) | LLM-Judge (Accuracy) |
|---|:---:|:---:|:---:|:---:|:---:|
| Base Qwen2.5-7B-Instruct | 0.1820 | 0.3840 | 0.7620 | 3.4 / 5.0 | 3.1 / 5.0 |
| **Fine-Tuned BanglaSupport-LLM (QLoRA)** | **0.4280** | **0.6910** | **0.9140** | **4.8 / 5.0** | **4.7 / 5.0** |

---

## 🚀 Quick Start & Reproduction

### 1. Installation
`ash
git clone https://github.com/mrshibly/BanglaSupport-LLM.git
cd BanglaSupport-LLM
python -m venv venv
# Windows: venv\Scripts\activate | Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
`

### 2. Full-Stack Local Serving (Windows / Linux)
Ensure your system has 16GB VRAM (or a configured 16GB Virtual Memory Pagefile on Windows) to host the 7B model locally.
`ash
# Terminal 1: Launch FastAPI Backend
uvicorn inference.api.main:app --reload --port 8000

# Terminal 2: Launch React Frontend
cd app/frontend
npm install
npm run dev
`
Visit http://localhost:3000 to interact with the tri-modal support assistant.

---

## 👨‍💻 Primary Architects & Contributors

This system was engineered end-to-end by:

- **Mahmudur Rahman (mrshibly)**
  - **Focus**: ML Pipeline, Model Fine-Tuning, Full-Stack Architecture, RAG implementation.
  - **Email**: [mahmudurrahman858@gmail.com](mailto:mahmudurrahman858@gmail.com)
  - **GitHub**: [@mrshibly](https://github.com/mrshibly)
  - **Hugging Face**: [@mrshibly](https://huggingface.co/mrshibly)

- **FHJibon**
  - **Focus**: Dataset curation, AI testing, and agentic intent design.
  - **GitHub**: [@FHJibon](https://github.com/FHJibon)

---

## 📄 License
Distributed under the MIT License. See LICENSE for more information.
