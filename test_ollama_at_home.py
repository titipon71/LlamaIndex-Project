import requests, os

base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

r = requests.post(
    f"{base}/api/chat",
    json={
        "model": "scb10x/llama3.2-typhoon2-1b-instruct:latest",
        "messages": [{"role": "user", "content": "ทดสอบจากเครื่องโลคัล"}],
        "stream": False,
    },
)
print(r.status_code)
print(r.text[:400])
