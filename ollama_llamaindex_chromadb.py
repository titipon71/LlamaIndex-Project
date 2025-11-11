# ollama_llamaindex_chromadb.py
import os
from dotenv import load_dotenv

# กำหนดค่า default
CHROMA_DIR = "./chroma_db"
# โหลดค่า env
load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")

# สร้างโฟลเดอร์ที่จำเป็น
os.makedirs("data", exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

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
    device="cuda",
    trust_remote_code=True
)

# ----- ตั้งค่า LLM (ผ่าน Ollama) -----
# ถ้า Ollama ไม่ได้รัน หรือไม่มีโมเดล จะ error
llm = Ollama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,   # สำคัญถ้าไม่ได้ใช้ค่า default
    request_timeout=60.0,
    additional_kwargs={
            "options": {
                "num_gpu": 1,
                "temperature": 0.4,
                "top_p": 0.9,
            }
        },
)

# ----- ตั้งค่า Vector Store (Chroma Persistent) -----
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb.config import Settings

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
collection_name = "quickstart2"

# ถ้ามีอยู่แล้วจะเปิดใช้ต่อ
try:
    chroma_collection = chroma_client.get_collection(collection_name)
except:
    chroma_collection = chroma_client.create_collection(collection_name)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ----- สร้าง/อัปเดต Index -----
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    storage_context=storage_context
)


# ----- ทดสอบถามคำถาม -----
query_engine = index.as_query_engine(llm=llm)

while True:
  question = input("ถามอะไรดี: ")
  response = query_engine.query(question)
  print(f"Q: {question}")
  print("A:", str(response))
