from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_DIR = PROJECT_ROOT / "knowledge_base" / "embeddings"

class RAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        if DB_DIR.exists() and any(DB_DIR.iterdir()):
            self.vector_db = Chroma(
                persist_directory=str(DB_DIR),
                embedding_function=self.embeddings
            )
        else:
            self.vector_db = None

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        if not self.vector_db:
            return ["পলিসি সম্বলিত কোনো তথ্য ভাণ্ডার পাওয়া যায়নি।"]
        results = self.vector_db.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]

    def format_rag_prompt(self, query: str, contexts: list[str]) -> str:
        context_str = "\n---\n".join(contexts)
        return f"প্রসঙ্গ তথ্য:\n{context_str}\n\nগ্রাহকের প্রশ্ন: {query}\n\nউপরের প্রাসঙ্গিক তথ্যের ওপর ভিত্তি করে সংক্ষিপ্ত ও নির্ভুল উত্তর দাও:"
