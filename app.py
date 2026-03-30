import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import re
import pandas as pd

# 🔐 .env yükle
load_dotenv()

# 🔑 API KEY
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    api_key = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=api_key)

# 🎯 Başlık
st.title("MEB Yönetmelik Asistanı - Sohbet Hafızalı")
st.info("⚠️ Sadece MEB yönetmeliği ile ilgili sorular sorabilirsiniz. Uygunsuz sorular yanıtlanmayacaktır.")

# 🧠 VECTOR DB
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory="okul_asistani_gpt_db",
        embedding_function=embeddings
    )
    return db

vector_db = load_vector_db()

# 🗂️ Session State ile sohbet geçmişini tut
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# 🔄 Tekrarlanan sayıları ve boşlukları temizle
def temizle_cevap(cevap):
    cevap = re.sub(r'(\b\d{2,4}\b)(?:/\s*\1)+', r'\1', cevap)
    cevap = re.sub(r'\s{2,}', ' ', cevap)
    return cevap

# ❌ Uygunsuz soruları engelle
def uygunsuz_mu(soru):
    anahtar_kelimeler = ["küfür", "argo", "siyaset", "ülke", "din", "ırk", "cinsiyet"]
    soru_lower = soru.lower()
    return any(anahtar in soru_lower for anahtar in anahtar_kelimeler)

# 🤖 SORGULAMA
def okul_asistani_sorgula(soru):
    if uygunsuz_mu(soru):
        return "⚠️ Bu soru uygun değil. Lütfen yalnızca MEB yönetmeliği ile ilgili resmi sorular sorun.", None, None

    # 🔍 arama sorgusu
    arama_sorgusu = f"{soru} meb yönetmelik maddesi devamsızlık şartları"

    # 🔥 vektör DB arama
    docs = vector_db.similarity_search_with_score(arama_sorgusu, k=3)
    docs = sorted(docs, key=lambda x: x[1])[:3]
    docs = [doc[0] for doc in docs]

    if not docs:
        return "Veri bulunamadı.", None, None

    # 🔥 bağlam oluştur
    baglam = "\n\n".join([doc.page_content[:500] for doc in docs])

    # 🤖 AI çağrısı için mesajlar
    messages = [
        {"role": "system", "content": """
Sen MEB yönetmeliği uzmanısın.
Kurallar:
- Sadece verilen bağlama göre cevap ver
- Eğer madde varsa belirt
- En yakın bilgiyi kullan
- "Bulunamadı" deme
- Cevapta küfür, hakaret, siyaset, din, ırk, cinsiyet ile ilgili içerik olamaz
- Cevap sadece resmi MEB yönetmeliği ile ilgili olmalı
- Anlamsız tekrarlar ve saçma ifadeler kullanma
"""}
    ]

    # Önceki sohbeti ekle
    for msg in st.session_state.conversation:
        messages.append(msg)

    messages.append({"role": "user", "content": f"{baglam}\n\nSoru: {soru}"})

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=500
    )

    cevap = chat_completion.choices[0].message.content
    cevap = temizle_cevap(cevap)

    # 📝 Session state güncelle
    st.session_state.conversation.append({"role": "user", "content": soru})
    st.session_state.conversation.append({"role": "assistant", "content": cevap})

    # 📚 Kaynak ekleme ve tablo oluşturma
    kaynaklar = [doc.page_content[:200] for doc in docs]
    tablo_data = {"Madde Özeti": [doc.page_content[:100]+"..." for doc in docs]}
    tablo_df = pd.DataFrame(tablo_data)

    return cevap, tablo_df, kaynaklar

# 💬 Chat Arayüzü
for msg in st.session_state.conversation:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

if prompt := st.chat_input("Sorunuzu yazın:"):
    with st.spinner("Yanıt hazırlanıyor..."):
        # Kullanıcının mesajını chat'te göster
        with st.chat_message("user"):
            st.markdown(prompt)
        cevap, tablo_df, kaynaklar = okul_asistani_sorgula(prompt)
        with st.chat_message("assistant"):
            st.markdown(cevap)
            if tablo_df is not None:
                st.markdown("📊 **İlgili Madde Tablosu:**")
                st.table(tablo_df)
            if kaynaklar is not None:
                st.markdown("📚 **Kaynaklar:**")
                for k in kaynaklar:
                    st.markdown(f"- {k}")
