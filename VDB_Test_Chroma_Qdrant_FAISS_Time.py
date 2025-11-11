# VDB_Test_StoreOnly_Simple.py
from dotenv import load_dotenv
import os, time

load_dotenv()

# ===== LlamaIndex core =====
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter

# ===== Config =====
DOC_DIR = "data"
TOP_K = int(os.getenv("TOP_K", "5"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
DEVICE = os.getenv("EMBED_DEVICE", "cuda")
QUERY = os.getenv("QUERY", "พลเมืองคืออะไร")

# ===== Embedding =====
embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL,
    device=DEVICE,
    normalize=True
)

# ===== Utils =====
def print_results(title, nodes, k):
    print(f"\n===== {title} (top {k}) =====")
    print("คำถาม:", QUERY)
    for i, n in enumerate(nodes[:k], start=1):
        node = getattr(n, "node", n)
        meta = getattr(node, "metadata", {}) or {}
        raw_score = getattr(n, "score", None)
        shown_score = f"{raw_score:.4f}" if raw_score is not None else "N/A"

        text = node.get_content()
        preview = (text[:1000] + "…") if len(text) > 1000 else text
        file_src = meta.get("file_name") or meta.get("source") or "unknown"

        print(f"[{i}] score={shown_score}  source={file_src}")
        print(preview)
        print("-" * 80)


def load_and_chunk(doc_dir):
    documents = SimpleDirectoryReader(doc_dir).load_data()
    if not documents:
        raise RuntimeError(f"ไม่พบไฟล์ในโฟลเดอร์ {doc_dir}")

    try:
        semantic = SemanticSplitterNodeParser(
            embed_model=embed_model,
            breakpoint_percentile_threshold=95,
            buffer_size=1,
        )
        nodes = semantic.get_nodes_from_documents(documents)
        if len(nodes) == 0 or (len(nodes) == 1 and len(nodes[0].get_content()) > 4000):
            raise ValueError("semantic failed; fallback")
        return nodes
    except Exception:
        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=150)
        return splitter.get_nodes_from_documents(documents)


# ===== เตรียม nodes =====
nodes = load_and_chunk(DOC_DIR)
print(f"[Prep] nodes = {len(nodes)}")

# ===== Pre-embedding (สำคัญ – เพื่อวัด Store-Only) =====
print("\n[Prep] Pre-embedding ...")
texts = [n.get_content() for n in nodes]
t0 = time.time()
embs = embed_model.get_text_embedding_batch(texts)
for n, e in zip(nodes, embs):
    n.embedding = e
t1 = time.time()
print(f"[Prep] เสร็จใน {t1 - t0:.4f} sec (จะไม่นับในผล Store-Only)")


# ===== ฟังก์ชันวัดเวลาแบบง่ายที่สุด =====
def simple_bench(title, vector_store):
    print(f"\n===== {title} =====")

    # --- Build Store-Only ---
    t0 = time.time()
    ctx = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(nodes, embed_model=embed_model, storage_context=ctx)
    t1 = time.time()
    print(f"Build (Store-Only): {t1 - t0:.4f} sec")

    # --- Query ---
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    t0 = time.time()
    results = retriever.retrieve(QUERY)
    t1 = time.time()
    print(f"Query latency: {t1 - t0:.4f} sec")

    print_results(title, results, TOP_K)


# -------------------------------------------------------------
# 1) CHROMA
# -------------------------------------------------------------
try:
    import chromadb
    from chromadb.config import Settings
    from llama_index.vector_stores.chroma import ChromaVectorStore

    os.makedirs("chroma_data", exist_ok=True)
    client = chromadb.PersistentClient(path="chroma_data", settings=Settings(anonymized_telemetry=False))

    CHROMA_COLL = os.getenv("CHROMA_COLLECTION", "quick_chroma")
    try:
        client.delete_collection(CHROMA_COLL)
    except:
        pass

    coll = client.create_collection(CHROMA_COLL)
    chroma_vs = ChromaVectorStore(chroma_collection=coll)

    simple_bench("Chroma", chroma_vs)

except Exception as e:
    print(f"[Chroma] ERROR: {e}")


# -------------------------------------------------------------
# 2) QDRANT (embedded)
# -------------------------------------------------------------
try:
    import qdrant_client
    from qdrant_client.http import models as qm
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    os.makedirs("qdrant_local", exist_ok=True)
    qc = qdrant_client.QdrantClient(path="qdrant_local")

    QDRANT_COLL = os.getenv("QDRANT_COLLECTION", "quick_qdrant")

    if qc.collection_exists(QDRANT_COLL):
        qc.delete_collection(QDRANT_COLL)

    qc.create_collection(
        collection_name=QDRANT_COLL,
        vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
    )

    qdrant_vs = QdrantVectorStore(client=qc, collection_name=QDRANT_COLL)

    simple_bench("Qdrant (embedded)", qdrant_vs)

except Exception as e:
    print(f"[Qdrant] ERROR: {e}")


# -------------------------------------------------------------
# 3) FAISS (in-memory)
# -------------------------------------------------------------
try:
    import faiss
    from llama_index.vector_stores.faiss import FaissVectorStore

    faiss_index = faiss.IndexFlatIP(EMBED_DIM)  # ใช้ cosine (เพราะ normalize=True)
    faiss_vs = FaissVectorStore(faiss_index=faiss_index)

    simple_bench("FAISS (local)", faiss_vs)

except Exception as e:
    print(f"[FAISS] ERROR: {e}")
