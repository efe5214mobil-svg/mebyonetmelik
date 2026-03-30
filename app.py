import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import re
import pandas as pd
from PIL import Image

# 🎨 Sayfa Konfigürasyonu (En üstte olmalı)
st.set_page_config(page_title="MEB Yönetmelik Asistanı", page_icon="🎓", layout="wide")

# 💅 Özel CSS ile Arayüzü Güzelleştirme
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .stTable { background-color: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

load_dotenv()

# 🔑 API KEY Kontrolü
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

# 🧠 VECTOR DB (Hata payını azaltmak için cache süresini artırdık)
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="okul_asistani_gpt_db", embedding_function=embeddings)

vector_db = load_vector_db()

# 🗂️ Session State
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# --- SIDEBAR: DERS PROGRAMI ---
with st.sidebar:
    st.header("🏫 Okul Panosu")
    dersprogram_klasor = "dersprogram_dosyasi"
    
    if os.path.exists(dersprogram_klasor):
        dosyalar = [f for f in os.listdir(dersprogram_klasor) if f.lower().endswith(".png")]
        if dosyalar:
            # Sınıf listesini temizle ve sırala
            sinif_listesi = sorted(dosyalar, key=lambda x: x.split('.')[0])
            secilen_dosya = st.selectbox("Sınıf Seçiniz:", sinif_listesi)
            
            img = Image.open(os.path.join(dersprogram_klasor, secilen_dosya))
            st.image(img, caption=f"{secilen_dosya} Programı", use_container_width=True)
        else:
            st.warning("Klasörde PNG bulunamadı.")
    
    if st.button("Sohbeti Temizle"):
        st.session_state.conversation = []
        st.rerun()

# --- ANA PANEL ---
st.title("🎓 MEB Yönetmelik Asistanı")
st.caption("Resmi yönetmeliklere dayalı akıllı sorgulama sistemi")

def uygunsuz_mu(soru):
    yasakli = ["küfür", "argo", "siyaset", "hakaret"] # Basit filtre
    return any(kelime in soru.lower() for kelime in yasakli)

def okul_asistani_sorgula(soru):
    if uygunsuz_mu(soru):
        return "⚠️ Bu soru topluluk kurallarımıza uygun değil.", None, None

    # 1. Arama Geliştirme (k=5 yaparak şansı artıralım)
    docs = vector_db.similarity_search(soru, k=4)
    if not docs:
        return "Üzgünüm, bu konuyla ilgili yönetmelikte bir veri bulamadım.", None, None

    baglam = "\n\n".join([doc.page_content for doc in docs])

    # 2. Mesaj Yapısı ve MODEL DÜZELTMESİ
    messages = [
        {"role": "system", "content": """Sen MEB yönetmelik uzmanısın. 
        Sana verilen dökümanlara göre soruyu yanıtla. 
        Kural 1: Cevabın başında mutlaka 'EVET' veya 'HAYIR' (veya 'BİLGİ BULUNAMADI') ifadesini büyük harflerle kullan.
        Kural 2: Ardından çok kısa bir açıklama ekle (Maksimum 2 cümle).
        Kural 3: Sadece dökümana sadık kal."""}
    ]
    
    # Geçmişi ekle (Son 3 mesajı alarak hafıza yönetimi yapalım)
    messages.extend(st.session_state.conversation[-3:])
    messages.append({"role": "user", "content": f"BAĞLAM:\n{baglam}\n\nSORU: {soru}"})

    try:
        completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile", # DOĞRU MODEL İSMİ
            temperature=0,
            max_tokens=150
        )
        cevap = completion.choices[0].message.content
    except Exception as e:
        cevap = f"⚠️ Teknik bir sorun oluştu: {str(e)}"

    # Veri Hazırlama
    tablo_df = pd.DataFrame({"Kaynak Maddeler": [d.page_content[:150] + "..." for d in docs]})
    kaynaklar = [d.page_content for d in docs]

    return cevap, tablo_df, kaynaklar

# --- CHAT ARAYÜZÜ ---
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Yönetmelik hakkında bir soru sorun..."):
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Yönetmelikler taranıyor..."):
            cevap, tablo, kaynaklar = okul_asistani_sorgula(prompt)
            st.markdown(cevap)
            
            if tablo is not None:
                with st.expander("📍 Dayanak Maddeler ve Kaynaklar"):
                    st.table(tablo)
                    for i, k in enumerate(kaynaklar):
                        st.info(f"Kaynak {i+1}: {k}")
            
            st.session_state.conversation.append({"role": "assistant", "content": cevap})
