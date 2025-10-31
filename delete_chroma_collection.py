# delete_chroma_collection.py
import os
import chromadb
from chromadb.config import Settings as ChromaSettings

# ---------- CONFIG ----------
CHROMA_DIR = "./chroma_db"           # path ของฐานข้อมูล Chroma ที่เก็บไว้
COLLECTION_NAME = "quickstart2"      # ชื่อคอลเลกชันที่ต้องการลบ

# ---------- สร้าง client ----------
client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=ChromaSettings(anonymized_telemetry=False),
)

# ---------- แสดงคอลเลกชันทั้งหมดก่อนลบ ----------
print("📚 รายชื่อคอลเลกชันทั้งหมดก่อนลบ:")
collections = client.list_collections()
if not collections:
    print("  (ไม่มีคอลเลกชันในฐานข้อมูลนี้)")
else:
    for c in collections:
        print(f"  - {c.name}")

# ---------- ลบคอลเลกชันที่ต้องการ ----------
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"\n✅ ลบคอลเลกชัน '{COLLECTION_NAME}' เรียบร้อยแล้ว")
except Exception as e:
    print(f"\n⚠️ ไม่สามารถลบคอลเลกชัน '{COLLECTION_NAME}' ได้: {e}")

# ---------- ตรวจสอบอีกครั้ง ----------
print("\n📚 รายชื่อคอลเลกชันหลังลบ:")
collections_after = client.list_collections()
if not collections_after:
    print("  (ไม่มีคอลเลกชันหลงเหลืออยู่ในฐานข้อมูล)")
else:
    for c in collections_after:
        print(f"  - {c.name}")

print("\nเสร็จสิ้น ✅")
