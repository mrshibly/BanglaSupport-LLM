# 🇧🇩 Bangla Customer Support LLM

A fine-tuned **Qwen3-8B** model for Bangla e-commerce customer support, demonstrating the full ML lifecycle: dataset curation → QLoRA training → evaluation → RAG comparison → agentic tool-calling → deployment.

## ✨ Highlights

- **Model**: Qwen3-8B fine-tuned with QLoRA (4-bit, rank 16) via Unsloth
- **Dataset**: ~23K native Bangla examples filtered from Bangla-Instruct (342K, ACL 2025) + Aya Bengali
- **Evaluation**: BLEU, ROUGE-L, BERTScore, LLM-as-a-Judge, human evaluation
- **RAG Comparison**: 3-system comparison (Base+RAG vs Fine-tuned vs Fine-tuned+RAG)
- **Deployment**: FastAPI + vLLM + React chat interface + Docker
- **Stretch**: Agentic tool-calling with mock e-commerce APIs

## 📋 Example

```
User:   আমার অর্ডার এখনো আসেনি।
AI:     আপনার অর্ডার নম্বরটি দিলে আমি স্ট্যাটাস দেখে বলতে পারি।
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
python dataset/scripts/download_and_filter.py   # Download & filter Bangla-Instruct
python dataset/scripts/download_aya.py           # Download Aya Bengali subset
python dataset/scripts/prepare_dataset.py        # Merge & deduplicate
python dataset/scripts/split_data.py             # Train/val/test split
```

### 3. Fine-Tune

```bash
# Smoke test (50 steps)
python training/train.py --max_steps 50

# Full training (~8-10 hours on RTX 5060 Ti)
python training/train.py
```

### 4. Evaluate

```bash
# Baseline
python evaluation/eval_auto.py --model Qwen/Qwen3-8B

# Fine-tuned
python evaluation/eval_auto.py --model checkpoints/qwen3-8b-bangla-support/final_adapter
```

### 5. Merge & Deploy

```bash
# Merge adapter
python training/merge_adapter.py --adapter checkpoints/qwen3-8b-bangla-support/final_adapter

# Start API
cd inference && uvicorn api.main:app --reload

# Start frontend
cd app/frontend && npm install && npm run dev
```

## 📁 Project Structure

```
bangla-support-llm/
├── dataset/
│   ├── scripts/          # Download, filter, prepare, split
│   └── splits/           # train.jsonl, val.jsonl, test.jsonl
├── training/
│   ├── configs/          # QLoRA hyperparameter configs
│   ├── train.py          # Unsloth + SFTTrainer
│   └── merge_adapter.py  # LoRA → merged model
├── evaluation/
│   ├── eval_auto.py      # BLEU, ROUGE, BERTScore
│   ├── eval_llm_judge.py # LLM-as-a-Judge
│   └── results/
├── inference/
│   └── api/              # FastAPI + RAG + tools
├── app/
│   └── frontend/         # React chat UI
├── docker/
│   └── docker-compose.yml
├── knowledge_base/       # RAG documents
├── notebooks/            # Exploration notebooks
├── README.md
├── benchmarks.md
└── model_card.md
```

## 📊 Results

See [benchmarks.md](benchmarks.md) for full evaluation results.

| Metric | Base Qwen3-8B | Fine-tuned | Δ |
|--------|--------------|------------|---|
| BLEU | — | — | — |
| ROUGE-L | — | — | — |
| BERTScore (F1) | — | — | — |

## 🛠 Hardware

- **GPU**: NVIDIA RTX 5060 Ti (16 GB VRAM)
- **Training**: QLoRA 4-bit → ~10 GB VRAM
- **Training time**: ~8-10 hours

## 📄 License

MIT

## 🤝 Acknowledgments

- [Bangla-Instruct](https://huggingface.co/datasets/md-nishat-008/Bangla-Instruct) — TigerLLM (ACL 2025)
- [Aya Dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset) — Cohere For AI
- [Unsloth](https://github.com/unslothai/unsloth) — Fast LLM fine-tuning
- [Qwen3](https://huggingface.co/Qwen/Qwen3-8B) — Alibaba Cloud
