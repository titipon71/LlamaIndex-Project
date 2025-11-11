# milvusTest.py
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader

# === โหลดเอกสาร ===
docs = SimpleDirectoryReader("data").load_data()
if not docs:
    raise RuntimeError("ไม่พบไฟล์ในโฟลเดอร์ data ใส่เอกสารก่อน")

embed = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu"
)

vs = MilvusVectorStore(
    dim=384,
    collection_name="quickstart_milvus001",
    uri="http://localhost:19530",
    overwrite=True,
)
ctx = StorageContext.from_defaults(vector_store=vs)

index = VectorStoreIndex.from_documents(docs, embed_model=embed, storage_context=ctx)

# === Query ===
res = index.as_retriever(similarity_top_k=5).retrieve("พลเมืองคืออะไร")
print(res)