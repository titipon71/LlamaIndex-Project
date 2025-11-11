# test_VectorDatabase.py
from dotenv import load_dotenv
import os
load_dotenv()

# ===== LlamaIndex core =====
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser , SentenceSplitter

# ===== Config =====
DOC_DIR = "data"
TOP_K = int(os.getenv("TOP_K", "5"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  # dim=384
DEVICE = os.getenv("EMBED_DEVICE", "cuda")
QUERY = "พลเมืองคืออะไร"

# ===== Embedding (ตัวเดียวทุกสโตร์) =====
embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL,
    device=DEVICE,
    normalize=True
)

# ===== Utils =====
def print_results(title, nodes, k):
    print(f"\n===== {title} (top {k}) =====")
    print("คำถาม: ", QUERY)
    if not nodes:
        print("(ไม่มีผลลัพธ์)")
        return

    for i, n in enumerate(nodes[:k], start=1):
        node = getattr(n, "node", n)
        meta = getattr(node, "metadata", {}) or {}

        src_backend = meta.get("_node_source", "").lower()  # ← บอก backend ที่มา
        raw_score = getattr(n, "score", None)

        if src_backend == "weaviate" and raw_score is not None:
            sim = max(0.0, min(1.0, float(raw_score)))
            shown_score = f"{sim:.4f} (norm-weaviate)"
        else:
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
            buffer_size=1
        )
        nodes = semantic.get_nodes_from_documents(documents)
        if len(nodes) == 0 or (len(nodes) == 1 and len(nodes[0].get_content()) > 4000):
            raise ValueError("semantic failed; fallback")
        return nodes
    
    except Exception:
        splitter = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=150,
        )
        return splitter.get_nodes_from_documents(documents)

def build_index_and_query(title, vector_store, nodes, embed_model, k, query):
    ctx = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(nodes, embed_model=embed_model, storage_context=ctx)
    results = index.as_retriever(similarity_top_k=k).retrieve(query)
    print_results(title, results, k)


# ===== เตรียม nodes =====
nodes = load_and_chunk(DOC_DIR)

# -------------------------------------------------------------
# 1) CHROMA (Persistent)
# -------------------------------------------------------------
try:
    import chromadb
    from chromadb.config import Settings
    from llama_index.vector_stores.chroma import ChromaVectorStore

    os.makedirs("chroma_data", exist_ok=True)
    chroma_client = chromadb.PersistentClient(
        path="chroma_data",
        settings=Settings(anonymized_telemetry=False)
    )
    CHROMA_COLL = os.getenv("CHROMA_COLLECTION", "quickstart_chroma")

    try:
        chroma_client.delete_collection(CHROMA_COLL)
        print(f"[Chroma] ลบคอลเล็กชันเก่า: {CHROMA_COLL}")
    except Exception:
        pass

    chroma_collection = chroma_client.create_collection(CHROMA_COLL)
    chroma_vs = ChromaVectorStore(chroma_collection=chroma_collection)

    build_index_and_query("Chroma", chroma_vs, nodes, embed_model, TOP_K, QUERY)

except Exception as e:
    print(f"[Chroma] ข้าม: {e}")

# -------------------------------------------------------------
# 2) QDRANT (embedded local)
# -------------------------------------------------------------
try:
    import qdrant_client
    from qdrant_client.http import models as qm
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    QDRANT_COLL = os.getenv("QDRANT_COLLECTION", "quickstart_qdrant")
    os.makedirs("qdrant_local", exist_ok=True)

    q_client = qdrant_client.QdrantClient(path="qdrant_local")

    if q_client.collection_exists(QDRANT_COLL):
        q_client.delete_collection(QDRANT_COLL)
        print(f"[Qdrant] ลบคอลเล็กชันเก่า: {QDRANT_COLL}")

    q_client.create_collection(
        collection_name=QDRANT_COLL,
        vectors_config=qm.VectorParams(size=384, distance=qm.Distance.COSINE),
    )

    qdrant_vs = QdrantVectorStore(client=q_client, collection_name=QDRANT_COLL)

    build_index_and_query("Qdrant (embedded local)", qdrant_vs, nodes, embed_model, TOP_K, QUERY)

except Exception as e:
    print(f"[Qdrant] ข้าม: {e}")

# -------------------------------------------------------------
# 3) PINECONE (SDK v3) — ล้างด้วย namespace
# -------------------------------------------------------------
try:
    from pinecone import Pinecone, ServerlessSpec
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    import time

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = os.getenv("PINECONE_INDEX", "quickstart-pinecone").lower()
    if not PINECONE_API_KEY:
        print("[Pinecone] ข้าม: ไม่พบ PINECONE_API_KEY")
    else:
        pc = Pinecone(api_key=PINECONE_API_KEY)

        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_REGION", "us-east-1"))
            )
            print(f"[Pinecone] สร้าง index ใหม่: {INDEX_NAME}")

        pc_index = pc.Index(INDEX_NAME)

        ns = os.getenv("PINECONE_NAMESPACE") or f"run_{int(time.time())}"
        try:
            pc_index.delete(delete_all=True, namespace=ns)
            print(f"[Pinecone] ล้าง namespace: {ns}")
        except Exception:
            pass

        pinecone_vs = PineconeVectorStore(pinecone_index=pc_index, namespace=ns)

        build_index_and_query("Pinecone", pinecone_vs, nodes, embed_model, TOP_K, QUERY)

except Exception as e:
    print(f"[Pinecone] ข้าม: {e}")

# -------------------------------------------------------------
# 4) FAISS (LOCAL in-memory)
# -------------------------------------------------------------
try:
    import faiss
    from llama_index.vector_stores.faiss import FaissVectorStore

    dim = 384
    faiss_index = faiss.IndexFlatIP(dim)  # in-memory ใหม่ทุกครั้ง
    faiss_vs = FaissVectorStore(faiss_index=faiss_index)

    build_index_and_query("FAISS (local)", faiss_vs, nodes, embed_model, TOP_K, QUERY)

except Exception as e:
    print(f"[FAISS] ข้าม: {e}")

# -------------------------------------------------------------
# 5) WEAVIATE (Docker + client v4)
# -------------------------------------------------------------
try:
    from llama_index.vector_stores.weaviate import WeaviateVectorStore
    import weaviate

    # เชื่อมต่อ Weaviate ที่รันด้วย Docker
    client = weaviate.connect_to_local(host="127.0.0.1", port=8080, grpc_port=50051)

    WEAVIATE_CLASS = os.getenv("WEAVIATE_CLASS", "QuickstartWeaviate")

    # ลบคอลเลกชันเก่า
    try:
        names = client.collections.list_all()
        if WEAVIATE_CLASS in names:
            client.collections.delete(WEAVIATE_CLASS)
            print(f"[Weaviate] ลบคอลเลกชันเก่า: {WEAVIATE_CLASS}")
    except Exception:
        pass

    weaviate_vs = WeaviateVectorStore(
        weaviate_client=client,
    )

    build_index_and_query("Weaviate (docker)", weaviate_vs, nodes, embed_model, TOP_K, QUERY)
    client.close()

except Exception as e:
    print(f"[Weaviate] ข้าม: {e}")



# # -------------------------------------------------------------
# # 6) MILVUS ขึ้น ไม่มีผลลัพธ์ ( [] ) ตลอด ไม่รู้ทำไม ทำตาม Docs เป๊ะๆ ก็ยังไม่ได้
# documentation 1 : https://llama-index.readthedocs.io/en/latest/vector_stores/milvus.html
# documentation 2 : https://developers.llamaindex.ai/python/examples/vector_stores/milvusindexdemo/
# documentation 3 : https://developers.llamaindex.ai/python/framework/community/integrations/vector_stores/
# # -------------------------------------------------------------
# try:
#     from llama_index.vector_stores.milvus import MilvusVectorStore

#     milvus_vs = MilvusVectorStore(
#     dim=384,
#     collection_name="quickstart_milvus2",
#     uri="http://localhost:19530",
#     overwrite=True,
#     )

#     build_index_and_query("Milvus (docker)", milvus_vs, nodes, embed_model, TOP_K, QUERY)

# except Exception as e:
#     print(f"[Milvus] ข้าม: {e}")
