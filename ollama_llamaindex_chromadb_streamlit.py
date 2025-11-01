import os
from typing import List

import streamlit as st
from dotenv import load_dotenv

# RAG components: LlamaIndex + Chroma + Ollama
import chromadb
from chromadb.config import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# --------- ENV & Paths ---------
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(BASE_DIR, "chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "quickstart2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "scb10x/llama3.2-typhoon2-1b-instruct:latest")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

st.set_page_config(page_title="RAG (Streamlit)", page_icon="🤖", layout="wide")
st.title("RAG Web UI (Streamlit)")
st.caption("LlamaIndex + Chroma + Ollama")

# ---------- cached resources ----------
@st.cache_resource(show_spinner=False)
def get_embed_model():
    return HuggingFaceEmbedding(model_name=EMBED_MODEL, device="cuda", trust_remote_code=True)

@st.cache_resource(show_spinner=False)
def get_llm():
    return Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
        additional_kwargs={"options": {"num_gpus": 1}}
    )

def _list_data_files() -> list[str]:
    files = []
    for name in sorted(os.listdir(DATA_DIR)):
        full = os.path.join(DATA_DIR, name)
        if os.path.isfile(full):
            files.append(name)
    return files

def _ensure_sample_if_empty() -> List[Document]:
    docs = SimpleDirectoryReader(DATA_DIR).load_data()
    if not docs:
        sample_path = os.path.join(DATA_DIR, "sample.txt")
        if not os.path.exists(sample_path):
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write("สวัสดี! ใส่ไฟล์ลง data/ แล้วกด Rebuild Index\n")
        docs = SimpleDirectoryReader(DATA_DIR).load_data()
    return docs

def _load_docs_from_selected(selected_files: list[str]) -> List[Document]:
    if not selected_files:
        # ถ้าไม่เลือกเลย ให้ fallback
        return _ensure_sample_if_empty()
    paths = [os.path.join(DATA_DIR, f) for f in selected_files]
    return SimpleDirectoryReader(input_files=paths).load_data()

@st.cache_resource(show_spinner=False)
def _get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))

def _get_or_create_collection(chroma_client):
    try:
        return chroma_client.get_collection(COLLECTION_NAME)
    except Exception:
        return chroma_client.create_collection(COLLECTION_NAME)

@st.cache_resource(show_spinner=False)
def build_index(docs: List[Document] | None = None) -> VectorStoreIndex:
    chroma_client = _get_chroma_client()
    chroma_collection = _get_or_create_collection(chroma_client)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if docs is None:
        docs = _ensure_sample_if_empty()

    index = VectorStoreIndex.from_documents(
        docs,
        embed_model=get_embed_model(),
        storage_context=storage_context,
        show_progress=True,
        batch_size=16,
    )
    return index

def clear_collection():
    chroma_client = _get_chroma_client()
    # ลบ collection เดิม
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    # สร้างใหม่ทันที เพื่อให้ build_index ครั้งถัดไปเจอแน่ ๆ
    chroma_client.create_collection(COLLECTION_NAME)

# --------- Sidebar: Files + Upload + Actions ---------
with st.sidebar:
    st.header("ไฟล์เอกสาร")

    up = st.file_uploader("อัปโหลดไฟล์ (จะถูกเก็บใน data/)", accept_multiple_files=True)
    if up:
        for f in up:
            dest = os.path.join(DATA_DIR, f.name)
            with open(dest, "wb") as out:
                out.write(f.read())
        st.success(f"อัปโหลด {len(up)} ไฟล์แล้ว — อย่าลืม Rebuild Index")

    st.subheader("เลือกไฟล์ที่จะใช้สร้างดัชนี")

    files_in_data = _list_data_files()

    if "selected_files" not in st.session_state:
        st.session_state.selected_files = set(files_in_data)

    new_selected = set(st.session_state.selected_files)

    for fname in files_in_data:
        checked = fname in st.session_state.selected_files
        c = st.checkbox(fname, value=checked, key=f"file_{fname}")
        if c:
            new_selected.add(fname)
        else:
            new_selected.discard(fname)

    st.session_state.selected_files = new_selected

    st.caption(f"เลือกแล้ว: {len(st.session_state.selected_files)} / {len(files_in_data)}")

    # Rebuild จากไฟล์ที่เลือก
    if st.button("Rebuild Index จากไฟล์ที่เลือก", use_container_width=True):
        build_index.clear()
        st.cache_resource.clear()

        docs = _load_docs_from_selected(list(st.session_state.selected_files))
        _ = build_index(docs=docs)
        st.success("Rebuild จากไฟล์ที่เลือก สำเร็จ ✅")

    # ล้าง collection
    if st.button("ล้าง Collection (Chroma)", use_container_width=True):
        clear_collection()
        build_index.clear()
        st.cache_resource.clear()
        st.success("ล้าง Collection เรียบร้อย 🧹")

# --------- Main Q&A ---------
question = st.text_area("คำถาม", placeholder="พิมพ์คำถามที่นี่...", height=100)
if st.button("ถาม", type="primary"):
    if not question.strip():
        st.error("กรุณาพิมพ์คำถาม")
    else:
        # 🟣 ตรงนี้คือจุดแก้สำคัญ — ใช้ไฟล์ที่เลือกมาสร้าง index
        selected_files = list(st.session_state.get("selected_files", []))
        docs = _load_docs_from_selected(selected_files)

        # เคลียร์ build_index ถ้าอยากบังคับให้สร้างจาก docs ชุดนี้ทุกครั้งก็ได้
        # build_index.clear()

        with st.spinner("กำลังค้นและสรุปคำตอบ..."):
            qe = build_index(docs=docs).as_query_engine(llm=get_llm())
            resp = qe.query(question)

        st.subheader("คำตอบ")
        st.write(str(resp))

        src_nodes = getattr(resp, "source_nodes", []) or []
        st.subheader("แหล่งที่มา")
        if not src_nodes:
            st.caption("ไม่มีรายละเอียดแหล่งที่มาพร้อมใช้งาน")
        else:
            for i, n in enumerate(src_nodes, 1):
                with st.expander(f"แหล่งที่มา #{i} • score={getattr(n, 'score', None)}"):
                    text_preview = (n.get_text() or "")[:1000] if hasattr(n, "get_text") else ""
                    st.code(text_preview)
                    meta = getattr(n.node, "metadata", {}) if hasattr(n, "node") else {}
                    st.json(meta)
