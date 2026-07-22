from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_DIR = PROJECT_ROOT / "knowledge_base" / "embeddings"

class RAGPipeline:
    def __init__(self):
        self._embeddings = None
        self._vector_db = None

    def _get_vector_db(self):
        if self._vector_db is None:
            try:
                self._embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    model_kwargs={"device": "cpu"}
                )
                if DB_DIR.exists() and any(DB_DIR.iterdir()):
                    self._vector_db = Chroma(
                        persist_directory=str(DB_DIR),
                        embedding_function=self._embeddings
                    )
            except Exception as e:
                print(f"⚠ RAG init note: {e}")
        return self._vector_db

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        vdb = self._get_vector_db()
        if not vdb:
            return ["পলিসি সম্বলিত কোনো তথ্য ভাণ্ডার পাওয়া যায়নি।"]
        results = vdb.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]

    def format_rag_prompt(self, query: str, contexts: list[str]) -> str:
        context_str = "\n---\n".join(contexts)
        return f"প্রসঙ্গ তথ্য:\n{context_str}\n\nগ্রাহকের প্রশ্ন: {query}\n\nউপরের প্রাসঙ্গিক তথ্যের ওপর ভিত্তি করে সংক্ষিপ্ত ও নির্ভুল উত্তর দাও:"
