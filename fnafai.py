# fnaf_local_rag_pyautogui.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from llama_cpp import Llama
from langchain_core.documents import Document
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
import random
import time
import pyautogui

# ------------------------------
# Ayarlar
MODEL_PATH = "models/mistral-7b-v0.1.Q4_0.gguf"
EMB_MODEL_NAME = "all-MiniLM-L6-v2"  # hafif ve hızlı
TOP_K = 4

# ------------------------------
# 1) Yerel LLM başlat
llm = Llama(model_path=MODEL_PATH)

# 2) Embedding modeli
embedder = SentenceTransformer(EMB_MODEL_NAME)

# ------------------------------
# 3) Basit scraper
def scrape_text(url, max_chars=15000):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")
    texts = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    txt = "\n".join(texts)
    return txt[:max_chars]

# 4) YouTube transkript alma
def youtube_transcript(video_id):
    try:
        items = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return " ".join([it['text'] for it in items])
    except:
        return ""

# ------------------------------
# 5) Örnek doküman koleksiyonu
docs = []
docs.append(Document(page_content="Topluluk teorisi: Golden Freddy ile Fredbear arasında bağ olabilir."))
docs.append(Document(page_content="Bazı timeline'lar Michael Afton olayını farklı yorumluyor."))
docs.append(Document(page_content="Reddit gönderisi: animatroniklerin geçmişi ve gölgeler."))

# 6) Embedding + FAISS index oluştur
texts = [d.page_content for d in docs]
embs = embedder.encode(texts, convert_to_numpy=True)
d = embs.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embs)

def retrieve(query, k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    results = [texts[i] for i in I[0] if i < len(texts)]
    return results

# 7) Sorgu + persona ile LLM'e sorma
def ask_fnaf_style(user_question):
    retrieved = retrieve(user_question)
    context = "\n\n".join(retrieved)
    persona = (
        "Sen gizemli, mekanik bir rehbersin. Kısa cümleler kullan, "
        "ara sıra duraksama (...) ve '— topluluk teorisi' gibi etiketler ekle."
    )
    prompt = f"{persona}\n\nBağlam:\n{context}\n\nKullanıcı: {user_question}\nCevap:"
    
    resp = llm(
        prompt=prompt,
        max_tokens=256,
        temperature=0.7,
        stop=["User:", "Kullanıcı:"]
    )
    
    return resp["choices"][0]["text"]

# ------------------------------
# 8) PyAutoGUI: pencere sallama
def shake_window():
    x, y = pyautogui.position()
    for _ in range(10):
        dx = random.randint(-10, 10)
        dy = random.randint(-10, 10)
        pyautogui.moveTo(x + dx, y + dy)
        time.sleep(0.03)
    pyautogui.moveTo(x, y)

# ------------------------------
# 9) AI loop
if __name__ == "__main__":
    print("FNaF AI hazır! Çıkmak için 'quit' yaz.\n")
    
    while True:
        q = input("Sen: ")

        if q.lower() in ["quit", "exit", "çık", "çıkış"]:
            break

        try:
            answer = ask_fnaf_style(q)
            
            # Robotik yazdırma efekti
            for c in answer:
                print(c, end="", flush=True)
                time.sleep(random.uniform(0.01, 0.05))
            print("\n")
            
            # Arasıra kendi kendine sorular
            if random.random() < 0.2:
                print("AI: Hmm... sen neden bunu sordun?...\n")
            
            # Arasıra pencere sallama
            if random.random() < 0.1:
                print("\n!!! Pencere sallanıyor !!!\n")
                shake_window()
            
            # Arasıra glitch efekti
            if random.random() < 0.1:
                glitch = "".join(random.choices("!@#$%^&*?<>/|", k=random.randint(5, 15)))
                print("AI glitch:", glitch, "\n")
            
        except Exception as e:
            print("Hata oluştu:", e)
