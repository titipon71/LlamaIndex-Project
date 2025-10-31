import os
from dotenv import load_dotenv

# ----- ค่าตั้งต้น -----
FAISS_DIR = "./faiss_store"             # โฟลเดอร์เก็บไฟล์ FAISS + mapping
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
FAISS_DOCS_PATH  = os.path.join(FAISS_DIR, "docstore.json")  # mapping ระหว่าง vector กับ doc_id

# เดิมของคุณ
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")

os.makedirs("data", exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# ----- โหลดเอกสาร -----
documents = SimpleDirectoryReader("data").load_data()
if not documents:
    raise RuntimeError("ไม่พบไฟล์ในโฟลเดอร์ data/ ใส่เอกสารก่อนหรือสร้าง sample.txt เพื่อทดสอบ")

# ----- ตั้งค่า Embedding -----
embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL,
    trust_remote_code=True
)

# ----- ตั้งค่า LLM (ผ่าน Ollama) -----
llm = Ollama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    request_timeout=60.0,
    additional_kwargs={"options": {"num_gpus": 1}},
)

# ===== ใช้ FAISS =====
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore

# หา embedding dimension อัตโนมัติ (กันพลาดเรื่องมิติ)
try:
    test_vec = embed_model.get_text_embedding("dim probe")
    embed_dim = len(test_vec)
except Exception:
    # fallback (ส่วนใหญ่ MiniLM-L12-v2 = 384) แต่ลองใช้วิธีข้างบนก่อนดีที่สุด
    embed_dim = 384

# ถ้ามีไฟล์เก่าแล้ว -> โหลดกลับมา, ถ้าไม่มีก็สร้าง index ใหม่
if os.path.exists(FAISS_INDEX_PATH):
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
else:
    # สร้าง index ใหม่
    faiss_index = faiss.IndexFlatL2(embed_dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ----- สร้าง/อัปเดต Index -----
# หมายเหตุ: ถ้าเอกสารถูกเพิ่ม/เปลี่ยนใหม่ เรียกบรรทัดนี้เพื่ออัปเดตเวกเตอร์
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    storage_context=storage_context,
)

# persist ลงดิสก์ (FAISS index + mapping)
# สำหรับ FaissVectorStore: จะเขียนทั้งตัว index และไฟล์ docstore mapping ไปยัง paths ที่กำหนดไว้
# ถ้าใช้ LlamaIndex รุ่นใหม่ vector_store.persist() ก็ทำงานได้เช่นกัน แต่บรรทัดด้านล่างนี้ชัดเจนกว่า
# faiss.write_index(vector_store.faiss_index, FAISS_INDEX_PATH)
# mapping id <-> doc จะถูกจัดการผ่าน docs_path โดย FaissVectorStore
# หากต้องการบังคับ persist mapping:
vector_store.persist(persist_path=FAISS_INDEX_PATH)

# ----- ทดสอบถามคำถาม -----
query_engine = index.as_query_engine(llm=llm)

while True:
    question = input("ถามอะไรดี: ")
    response = query_engine.query(question)
    print(f"Q: {question}")
    print("A:", str(response))
