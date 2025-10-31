# test_chroma_retrieval_and_inspect.py
import os
from dotenv import load_dotenv

from retrieve_direct_chroma import pretty_print

# --------- ENV / defaults ----------
load_dotenv()
CHROMA_DIR = "./chroma_db"
DATA_DIR = "data"
COLLECTION_NAME = "quickstart2"
QUERY_TEXT = "วันเรียนภาคฤดูร้อนเริ่มเมื่อไหร่"
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# --------- LlamaIndex imports ----------
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter                 
from llama_index.core.postprocessor import SimilarityPostprocessor             
from llama_index.core.postprocessor import SentenceTransformerRerank           

# --------- Embedding model ----------
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, trust_remote_code=True)
Settings.embed_model = embed_model

# --------- Node parsing (chunking) ----------  ทำก่อน ingest เสมอ
Settings.node_parser = SentenceSplitter(chunk_size=600, chunk_overlap=120)

# --------- Chroma Vector Store (persistent) ----------
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb.config import Settings as ChromaSettings

# --- Wrapper สำหรับ Chroma embedding_function ---
class LlamaIndexEmbeddingFunction:
    """Wrapper สำหรับ Chroma >= 0.4.16"""
    def __init__(self, li_embed):
        self.li_embed = li_embed

    def __call__(self, input):
        if isinstance(input, str):
            texts = [input]
        else:
            texts = list(input)
        return self.li_embed.get_text_embedding_batch(texts)

    def embed_query(self, input, **kwargs):
        if isinstance(input, list) and len(input) == 1:
            input = input[0]
        return self.li_embed.get_text_embedding(input)

    def embed_documents(self, input=None, **kwargs):
        texts = input if input is not None else kwargs.get("texts") or kwargs.get("documents") or []
        return self.li_embed.get_text_embedding_batch(list(texts))

    def name(self) -> str:
        model_name = getattr(self.li_embed, "model_name", None) or "unknown"
        return f"llamaindex_hf::{model_name}"

# --- helper: บังคับให้เป็น 2D list ---
def ensure_2d_embedding(vec):
    try:
        import numpy as np
        if isinstance(vec, np.ndarray):
            vec = vec.tolist()
    except Exception:
        pass
    if isinstance(vec, (int, float)):
        raise TypeError("Expected 1D embedding vector, but got a scalar float")
    if isinstance(vec, list) and (not vec or isinstance(vec[0], (int, float))):
        return [vec]
    return vec

embed_fn = LlamaIndexEmbeddingFunction(embed_model)

# ใช้ PersistentClient ตัวเดียว
client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=ChromaSettings(anonymized_telemetry=False),
)

# รวม get_or_create / get / create และบังคับ space=cosine
# NOTE: ถ้าคอลเลกชันถูกสร้างมาก่อนโดยไม่ได้กำหนด space=cosine ให้ลบโฟลเดอร์ฐานแล้วรันใหม่
try:
    chroma_collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},                 # NEW: บังคับ cosine
    )
except AttributeError:
    try:
        chroma_collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
        )
    except Exception:
        chroma_collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},             # NEW
        )

# --------- Inspect ก่อน: นับ / peek ----------
print("\n[C] Inspect Chroma")
count = None
try:
    count = chroma_collection.count()
    print(f"  count: {count}")
except Exception:
    print("  (รุ่น Chroma นี้ไม่มี .count())")

peek = {}
try:
    peek = chroma_collection.peek(limit=5) or {}
except Exception as e:
    print(f"  peek error: {e}")

print("  peek ids:", peek.get("ids", []))

# --------- ถ้าคลังยังว่าง: ingest จาก data/ ---------
if (count is not None and count == 0) or (not peek.get("ids")):
    print("  คลังว่าง → ingest จากโฟลเดอร์ data/")
    docs = SimpleDirectoryReader(DATA_DIR).load_data()
    if not docs:
        raise RuntimeError(
            "ไม่พบเอกสารใน data/ — ใส่ไฟล์อย่างน้อย 1 ไฟล์ เช่น data/sample.txt"
        )
    # ผูก ChromaVectorStore กับ collection เดิม
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # สร้างดัชนีและเขียน embeddings ลง Chroma
    _ = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

    # อัปเดตสถานะหลัง ingest
    try:
        count = chroma_collection.count()
        print(f"  count หลัง ingest: {count}")
    except Exception:
        pass
    try:
        peek = chroma_collection.peek(limit=5) or {}
        print("  peek ids หลัง ingest:", peek.get("ids", []))
    except Exception:
        pass

# --------- สร้าง Index จาก vector store ที่มีอยู่ (ไม่ re-ingest) ----------
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store)

def retrieve_with_llamaindex(query: str, top_k: int = 3):
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    results = []
    for rank, n in enumerate(nodes, start=1):
        results.append({
            "rank": rank,
            "score": n.score,  # ยิ่งสูงยิ่งใกล้
            "node_id": getattr(n, "node_id", getattr(n, "id_", None)),
            "source": n.metadata.get("file_path") or n.metadata.get("filename") or n.metadata,
            "text": n.get_content().strip()[:500]  # ตัดแสดง 500 ตัวอักษรพอให้เห็นบริบท
        })
    return results

# --------- A) Retrieval test (ไม่ใช้ LLM) ----------
print("\n[A] Retrieval test (top 3 nodes):")
# ดึง candidates มาก่อนเยอะหน่อย แล้วค่อยตัด/จัดอันดับ
retriever = index.as_retriever(similarity_top_k=10)
nodes = retriever.retrieve(QUERY_TEXT)

# ตัดทิ้งผลที่ความคล้ายต่ำเกินไป (ถ้าใช้ cosine ~ 0.35 กำลังดี)
cutoff = 0.35
nodes = SimilarityPostprocessor(similarity_cutoff=cutoff).postprocess_nodes(
    nodes, query_str=QUERY_TEXT
)

# Re-rank ด้วย cross-encoder (มี fallback ถ้าโหลดโมเดลใหญ่ไม่ได้)
rerank_models_try = [
    "BAAI/bge-reranker-large",                     # คุณภาพดี (หนัก)
    "cross-encoder/ms-marco-MiniLM-L-6-v2",        # เบากว่า (เร็ว)
]
reranker = None
for m in rerank_models_try:
    try:
        reranker = SentenceTransformerRerank(model=m, top_n=3)
        # โปรบหนึ่งทีแบบแห้ง ๆ เพื่อจับ error ให้เร็ว
        _ = reranker.model  # เข้าถึง property เพื่อให้แน่ใจว่าโหลดสำเร็จ
        print(f"  ใช้ reranker: {m}")
        print("---------------------------------")
        li = retrieve_with_llamaindex(QUERY_TEXT, 3)
        pretty_print("LlamaIndex Retriever (no LLM)", li)
        break
    except Exception as e:
        print(f"  โหลด reranker ไม่สำเร็จ {m}: {e}")
        reranker = None

if reranker is not None and nodes:
    nodes = reranker.postprocess_nodes(nodes, query_str=QUERY_TEXT)

# แสดงผล 3 อันดับแรก
for i, n in enumerate(nodes[:3], 1):
    score = getattr(n, "score", None)
    text = n.get_content()
    snippet = text[:200].replace("\n", " ")
    print(f"  #{i} score={score}")
    print("     ", snippet, "..." if len(text) > 200 else "")

# --------- (เสริม) low-level query ที่ Chroma (ไม่ผ่าน LlamaIndex) ----------
print("\n[C] Low-level query ที่ Chroma (ไม่ผ่าน LlamaIndex):")
try:
    q_emb = embed_model.get_text_embedding(QUERY_TEXT)   # -> list[float]
    q_emb_2d = ensure_2d_embedding(q_emb)                # -> list[list[float]]
    low_q = chroma_collection.query(
        query_embeddings=q_emb_2d,
        n_results=5,
        include=["distances", "documents", "metadatas"],   # NEW
    )
    print("  ids:", low_q.get("ids"))
    print("  distances:", low_q.get("distances"))
    docs_ = low_q.get("documents", [[]])[0]
    metas_ = low_q.get("metadatas", [[]])[0]
    for i, (doc, meta) in enumerate(zip(docs_[:3], metas_[:3]), 1):
        doc_snip = (doc or "")[:180].replace("\n", " ")
        print(f"   -> [{i}] {doc_snip}...")
        if meta:
            print(f"      meta: {meta}")
except Exception as e:
    print("  low-level query error:", e)

print("\nเสร็จสิ้น ✅")
