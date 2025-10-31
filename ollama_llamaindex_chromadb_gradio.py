import os
from typing import Tuple, List, Dict, Any

import gradio as gr
from dotenv import load_dotenv

# RAG components
import chromadb
from chromadb.config import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(BASE_DIR, "chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "quickstart2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

_embed = None
_llm = None
_index = None

def get_embed():
    global _embed
    if _embed is None:
        _embed = HuggingFaceEmbedding(model_name=EMBED_MODEL, trust_remote_code=True)
    return _embed

def get_llm():
    global _llm
    if _llm is None:
        _llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            request_timeout=120.0,
            additional_kwargs={"options": {"num_gpus": 1}}
        )
    return _llm

def ensure_index(rebuild: bool = False):
    global _index
    if rebuild or _index is None:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
        try:
            chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
        except Exception:
            chroma_collection = chroma_client.create_collection(COLLECTION_NAME)

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        docs = SimpleDirectoryReader(DATA_DIR).load_data()
        if not docs:
            sample_path = os.path.join(DATA_DIR, "sample.txt")
            if not os.path.exists(sample_path):
                with open(sample_path, "w", encoding="utf-8") as f:
                    f.write("สวัสดี! ใส่ไฟล์ลง data/ แล้วค่อย Rebuild Index\n")
            docs = SimpleDirectoryReader(DATA_DIR).load_data()

        _index = VectorStoreIndex.from_documents(
            docs, embed_model=get_embed(), storage_context=storage_context
        )
    return _index

def upload_files(files: list) -> str:
    if not files:
        return "ไม่ได้เลือกไฟล์"
    for f in files:
        dest = os.path.join(DATA_DIR, os.path.basename(f.name))
        with open(dest, "wb") as out:
            out.write(open(f.name, "rb").read())
    return f"อัปโหลด {len(files)} ไฟล์แล้ว — กด Rebuild Index เพื่ออัปเดต"

def do_rebuild() -> str:
    ensure_index(rebuild=True)
    return "Rebuild สำเร็จ ✅"

def ask(question: str) -> Tuple[str, str]:
    if not question or not question.strip():
        return "กรุณาพิมพ์คำถาม", ""
    qe = ensure_index().as_query_engine(llm=get_llm())
    resp = qe.query(question)
    srcs = []
    for n in getattr(resp, "source_nodes", []) or []:
        srcs.append({
            "score": getattr(n, "score", None),
            "text_preview": (n.get_text() or "")[:300] if hasattr(n, "get_text") else "",
            "metadata": getattr(n.node, "metadata", {}) if hasattr(n, "node") else {},
        })
    import json
    return str(resp), json.dumps(srcs, ensure_ascii=False, indent=2)

with gr.Blocks(title="RAG Web UI (Gradio)") as demo:
    gr.Markdown("# RAG Web UI (Gradio)")
    gr.Markdown("ใช้ LlamaIndex + Chroma + Ollama")

    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(label="คำถาม", lines=5, placeholder="พิมพ์คำถามที่นี่...")
            ask_btn = gr.Button("ถาม", variant="primary")
            answer = gr.Textbox(label="คำตอบ", lines=8)
            sources = gr.Code(label="แหล่งที่มา (JSON)")
        with gr.Column(scale=1):
            files = gr.File(label="อัปโหลดไฟล์ (เก็บใน data/)", file_count="multiple")
            upload_btn = gr.Button("อัปโหลด")
            rebuild_btn = gr.Button("Rebuild Index")
            status = gr.Markdown()

    upload_btn.click(fn=upload_files, inputs=[files], outputs=[status])
    rebuild_btn.click(fn=do_rebuild, outputs=[status])
    ask_btn.click(fn=ask, inputs=[question], outputs=[answer, sources])

if __name__ == "__main__":
    demo.launch()
