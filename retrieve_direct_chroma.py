# retrieve_direct.py
import os
from dotenv import load_dotenv

# ---------- ตั้งค่าเบื้องต้น ----------
CHROMA_DIR = "./chroma_db"
DATA_DIR = "data"
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TOP_K = int(os.getenv("TOP_K", 3))  # ปรับจำนวนผลลัพธ์ที่อยากดู

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
load_dotenv()

# ---------- โหลดเอกสาร + ฝังลงเวกเตอร์สโตร์ (ถ้ายังไม่มี) ----------
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# โหลดเอกสารจากโฟลเดอร์ data/
documents = SimpleDirectoryReader(DATA_DIR).load_data()
if not documents:
    raise RuntimeError("ไม่พบไฟล์ในโฟลเดอร์ data/ ใส่เอกสารก่อนหรือสร้าง data/sample.txt เพื่อทดสอบ")

# สร้าง embedding model (ไม่เกี่ยวกับ LLM)
embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL,
    trust_remote_code=True
)

# ---------- เตรียม Chroma (Persistent) ----------
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb.config import Settings

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False)) #เปิด/สร้างฐาน Chroma แบบ persistent
collection_name = "quickstart2"  # ชื่อคอลเลคชัน

try:
    chroma_collection = chroma_client.get_collection(collection_name)
    chroma_collection.delete(collection_name)  # ลบข้อมูลเก่า (ถ้ามี) เพื่อทดสอบใหม่

except:
    chroma_collection = chroma_client.create_collection(collection_name)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# สร้าง/อัปเดตดัชนี (ฝังเอกสาร -> เก็บใน Chroma)
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    storage_context=storage_context,
)
# (ถ้าอยาก pure-Chroma จริง ๆ ต้อง แตกชิ้นเอง + ฝังเอง แล้ว collection.add(ids=..., documents=..., metadatas=..., embeddings=...))

# ---------- วิธีที่ 1: ใช้ LlamaIndex Retriever โดยไม่ใช้ LLM ----------
def retrieve_with_llamaindex(query: str, top_k: int = TOP_K):
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

# ---------- วิธีที่ 2: คิวรี Chroma ตรง ๆ ----------
# (เข้ารหัสคำถามด้วย embed_model แล้ว query ที่คอลเลคชัน)
def retrieve_with_chroma(query: str, top_k: int = TOP_K):
    qvec = embed_model.get_text_embedding(query)  # เรา ฝังคำถาม เป็นเวกเตอร์ด้วยโมเดลเดียวกับที่ใช้ตอน ingest
    out = chroma_collection.query(
        query_embeddings=[qvec],
        n_results=top_k,
        include=["documents", "metadatas", "distances", "embeddings"]  # embeddings ไม่จำเป็นก็ได้  include คือพารามิเตอร์ที่บอกว่า “อยากให้ผลลัพธ์ คืนฟิลด์อะไรบ้าง”
    )
# Chroma คำนวณระยะ/ความคล้าย ด้วย metric ของคอลเลกชัน (ปกติ “cosine”) แล้วส่งกลับ:
# documents: เนื้อหาแต่ละชิ้น
# metadatas: เมทาดาทาของชิ้น
# ids: ไอดีของชิ้น
# distances: ระยะ (ถ้า cosine → ยิ่งต่ำยิ่งคล้าย)
    
    # หมายเหตุ: Chroma ให้ค่า "distances" (ยิ่งน้อยยิ่งคล้าย ถ้าใช้ cosine distance)
    results = []
    docs = out.get("documents", [[]])[0]
    metas = out.get("metadatas", [[]])[0]
    dists = out.get("distances", [[]])[0]
    # emb = out.get("embeddings", [[]])[0] # อันนี้ไม่ต้องแสดงก็ได้
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        results.append({
            "rank": i,
            "distance": dist,  # ยิ่งต่ำยิ่งคล้าย (ปกติเป็น cosine distance)
            "source": (meta.get("file_path") or meta.get("filename") or meta),
            "text": (doc or "")[:500],
            # "embedding": emb[i-1],  # อันนี้ไม่ต้องแสดงก็ได้
        })
    return results

# ---------- CLI เลือกโหมด ----------
def pretty_print(title: str, rows: list[dict]):
    print("\n" + "="*8 + f" {title} " + "="*8)
    if not rows:
        print("(no results)")
        return
    for r in rows:
        if "score" in r:
            head = f"[{r['rank']}] score={r['score']:.4f} ความคล้าย (ยิ่งสูงยิ่งใกล้)"
        else:
            head = f"[{r['rank']}] distance={r['distance']:.4f} ระยะ (ส่วนใหญ่เป็น cosine) (ยิ่งต่ำยิ่งคล้าย)"
        print(head)
        print(f"source: {r['source']}")
        print(f"text  : {r['text']}\n")

if __name__ == "__main__":
    print("โหมดทดสอบดึงจากเวกเตอร์สโตร์โดย 'ไม่ใช้ LLM'")
    print("พิมพ์คำถาม แล้วดูผลจาก 2 วิธี (LlamaIndex Retriever / Chroma Query)")
    print("กด Ctrl+C เพื่อออก\n")
    while True:
        try:
            q = input("ถามอะไรดี: ").strip()
            if not q:
                continue

            li = retrieve_with_llamaindex(q, TOP_K)
            pretty_print("LlamaIndex Retriever (no LLM)", li)

            ch = retrieve_with_chroma(q, TOP_K)
            pretty_print("Chroma Direct Query", ch)
        except KeyboardInterrupt:
            print("\nบายครับ 👋")
            break
