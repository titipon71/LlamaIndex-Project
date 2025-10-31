# retrieve_direct_faiss.py
import os
import json
import numpy as np
from dotenv import load_dotenv

# ---------- ตั้งค่าเบื้องต้น ----------
DATA_DIR = "data"
FAISS_DIR = "./faiss_db"  # โฟลเดอร์สำหรับเก็บไฟล์ดัชนี (ถ้าจะ persist)
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TOP_K = int(os.getenv("TOP_K", 3))  # จำนวนผลลัพธ์ที่อยากดู

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)
load_dotenv()

# ---------- โหลดเอกสาร + ฝังลงเวกเตอร์สโตร์ (ถ้ายังไม่มี) ----------
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

# โหลดเอกสารจากโฟลเดอร์ data/
documents = SimpleDirectoryReader(DATA_DIR).load_data()
if not documents:
    raise RuntimeError("ไม่พบไฟล์ในโฟลเดอร์ data/ ใส่เอกสารก่อนหรือสร้าง data/sample.txt เพื่อทดสอบ")

# สร้าง embedding model (ไม่เกี่ยวกับ LLM)
embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL,
    trust_remote_code=True
)

# ---------------------------------------------------------
# วิธีที่ 1: ใช้ LlamaIndex + FAISSVectorStore
# ---------------------------------------------------------
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore

# หา dimension ของเวกเตอร์จากโมเดล
_dummy_dim_vec = embed_model.get_text_embedding("dim_probe")
EMB_DIM = len(_dummy_dim_vec)

# เลือก metric: ใช้ Inner Product (IP) สำหรับ cosine similarity (ควร normalize เวกเตอร์)
faiss_index_for_li = faiss.IndexFlatIP(EMB_DIM)

# สร้าง vector store + storage context
faiss_store = FaissVectorStore(faiss_index=faiss_index_for_li)
storage_context = StorageContext.from_defaults(vector_store=faiss_store)

# สร้าง/อัปเดตดัชนี (ฝังเอกสาร -> เก็บใน FAISS)
# LlamaIndex จะดูแลการ chunk ให้เองตามดีฟอลต์
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    storage_context=storage_context,
)

# ---------------------------------------------------------
# วิธีที่ 2: คิวรี FAISS ตรง ๆ (ทำดัชนีอีกก้อนเพื่อสาธิต)
# ---------------------------------------------------------
# เราจะ chunk เอง + ฝังเอง + build FAISS index เอง
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
nodes = splitter.get_nodes_from_documents(documents)

texts = [n.get_content() for n in nodes]
metas = [n.metadata for n in nodes]

# ฝังทีละชุด (จะใช้ batch ก็ได้ แต่ให้เขียนง่ายๆแบบวน loop)
emb_list = [embed_model.get_text_embedding(t) for t in texts]
emb = np.array(emb_list, dtype="float32")
# normalize L2 เพื่อใช้ IP ให้เทียบเท่า cosine similarity
emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

# FAISS index (IP)
faiss_index_direct = faiss.IndexFlatIP(EMB_DIM)
faiss_index_direct.add(emb_norm)

# mapping id -> (text, meta) สำหรับอ้างอิงผลลัพธ์
id2payload = [
    {
        "text": texts[i],
        "meta": metas[i]
    }
    for i in range(len(texts))
]

# บันทึกเพื่อใช้งานครั้งถัดไป (ทำ persistence)
# faiss.write_index(faiss_index_direct, os.path.join(FAISS_DIR, "index_ip.faiss"))
# with open(os.path.join(FAISS_DIR, "payload.json"), "w", encoding="utf-8") as f:
#     json.dump(id2payload, f, ensure_ascii=False)

# ---------- ฟังก์ชันดึงผล ----------
def retrieve_with_llamaindex(query: str, top_k: int = TOP_K):
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    results = []
    for rank, n in enumerate(nodes, start=1):
        results.append({
            "rank": rank,
            "score": float(n.score),  # ยิ่งสูงยิ่งใกล้ (เพราะใช้ IP/cosine sim)
            "node_id": getattr(n, "node_id", getattr(n, "id_", None)),
            "source": n.metadata.get("file_path") or n.metadata.get("filename") or n.metadata,
            "text": n.get_content().strip()[:500]
        })
    return results

def retrieve_with_faiss(query: str, top_k: int = TOP_K):
    # เข้ารหัสคำถาม → normalize → คิวรี FAISS index
    q = np.array(embed_model.get_text_embedding(query), dtype="float32")[None, :]
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    sims, idxs = faiss_index_direct.search(q, top_k)  # sims = cosine similarity (เพราะเรา normalize แล้ว + ใช้ IP)
    sims = sims[0]
    idxs = idxs[0]

    results = []
    for i, (score, idx) in enumerate(zip(sims, idxs), start=1):
        if idx == -1:
            continue
        payload = id2payload[idx]
        meta = payload["meta"]
        text = payload["text"]
        results.append({
            "rank": i,
            "score": float(score),  # ยิ่งสูงยิ่งคล้าย (cosine sim)
            "source": (meta.get("file_path") or meta.get("filename") or meta),
            "text": (text or "")[:500],
        })
    return results

# ---------- CLI เลือกโหมด ----------
def pretty_print(title: str, rows: list[dict]):
    print("\n" + "="*8 + f" {title} " + "="*8)
    if not rows:
        print("(no results)")
        return
    for r in rows:
        head = f"[{r['rank']}] score={r['score']:.4f} ความคล้าย (ยิ่งสูงยิ่งใกล้)"
        print(head)
        print(f"source: {r['source']}")
        print(f"text  : {r['text']}\n")

if __name__ == "__main__":
    print("โหมดทดสอบดึงจากเวกเตอร์สโตร์โดย 'ไม่ใช้ LLM' (FAISS)")
    print("พิมพ์คำถาม แล้วดูผลจาก 2 วิธี (LlamaIndex Retriever / FAISS Direct)")
    print("กด Ctrl+C เพื่อออก\n")
    while True:
        try:
            q = input("ถามอะไรดี: ").strip()
            if not q:
                continue

            li = retrieve_with_llamaindex(q, TOP_K)
            pretty_print("LlamaIndex Retriever (FAISS, no LLM)", li)

            fd = retrieve_with_faiss(q, TOP_K)
            pretty_print("FAISS Direct Query (cosine sim via IP)", fd)
        except KeyboardInterrupt:
            print("\nบายครับ 👋")
            break
