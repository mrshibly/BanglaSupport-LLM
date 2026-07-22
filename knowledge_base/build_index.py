"""
Build ChromaDB index from Bangla customer support FAQ policy documents.

Usage:
    python knowledge_base/build_index.py
"""

import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rich.console import Console
console = Console(force_terminal=False)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "knowledge_base" / "documents"
DB_DIR = PROJECT_ROOT / "knowledge_base" / "embeddings"

# Sample Bangla e-commerce FAQs/policies to populate knowledge base
SAMPLE_POLICIES = """# ই-কমার্স পলিসি গাইডলাইন

## ১. রিটার্ন পলিসি
পণ্য গ্রহণের পর ৭ দিনের মধ্যে রিটার্ন রিকোয়েস্ট করতে হবে। পণ্যটি অব্যবহৃত এবং মূল প্যাকেজিং সহ থাকতে হবে। ভুল বা ক্ষতিগ্রস্ত পণ্য পেলে বিনামূল্যে রিটার্ন প্রযোজ্য।

## ২. রিফান্ড নিয়মাবলি
রিটার্নকৃত পণ্য হাবে পৌঁছানোর পর যাচাই-বাছাই করা হয়। যাচাই সফল হলে ৫-৭ কার্যদিবসের মধ্যে আপনার বিকাশ, নগদ বা ব্যাংক অ্যাকাউন্টে টাকা রিফান্ড করা হবে।

## ৩. ডেলিভারি সময়সীমা
ঢাকার ভেতরে সাধারণত ২-৩ কার্যদিবস এবং ঢাকার বাইরে ৪-৭ কার্যদিবসের মধ্যে ডেলিভারি সম্পন্ন হয়। প্রিমিয়াম বা এক্সপ্রেস ডেলিভারিতে ২৪ ঘণ্টার মধ্যে ডেলিভারি দেয়া হয়।

## ৪. পেমেন্ট পদ্ধতি
ক্যাশ অন ডেলিভারি (COD), বিকাশ, নগদ, রকেট এবং যেকোনো ডেবিট/ক্রেডিট কার্ডের মাধ্যমে পেমেন্ট গ্রহণ করা হয়।

## ৫. অর্ডার বাতিলকরণ
অর্ডারটি 'Shipped' বা কুরিয়ারে হস্তান্তরের পূর্বে কাস্টমার ড্যাশবোর্ড থেকে সহজেই বাতিল করা সম্ভব। কুরিয়ারে চলে গেলে ডেলিভারি ম্যানের কাছে গ্রহণের সময় বাতিল করতে পারেন।
"""

def main():
    console.print("\n[bold cyan]═══ Building RAG Knowledge Base ═══[/bold cyan]\n")
    
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    policy_file = DOCS_DIR / "ecommerce_policies_bn.md"
    
    if not policy_file.exists():
        with open(policy_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_POLICIES)
        console.print(f"[green]✓ Created sample Bangla policy document at {policy_file}[/green]")

    loader = TextLoader(str(policy_file), encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "।", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    console.print(f"[green]✓ Split documents into {len(chunks)} chunks[/green]")

    console.print("[yellow]Embedding chunks with paraphrase-multilingual-MiniLM-L12-v2...[/yellow]")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR)
    )
    console.print(f"[green]✓ Saved ChromaDB index to {DB_DIR}[/green]\n")

if __name__ == "__main__":
    main()
