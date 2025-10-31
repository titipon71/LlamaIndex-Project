from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# โหลดเอกสาร
documents = SimpleDirectoryReader("data").load_data()


embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", trust_remote_code=True)

# ใช้ LLM จาก Ollama เช่น llama3, mistral ฯลฯ
llm = Ollama(model="qwen3:0.6b")

# สร้าง index
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
)

# สร้าง query engine
query_engine = index.as_query_engine(llm=llm)

# ถามคำถาม
question = input("ถามอะไรดี: ")
print(f"Q: {question}")
response = query_engine.query(question)
print(response)