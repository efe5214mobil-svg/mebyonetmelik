# app.py
import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import re
import pandas as pd
from PIL import Image

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

# 🏫 Sınıf Ders Programı Görseli (Sidebar)
st.sidebar.header("📌 Sınıf Ders Programı")
dersprogram_klasor = "dersprogram_dosyasi"

# Klasördeki tüm PNG dosyalarını tarayıp sınıf isimleri ve dosya yolu eşlemesi oluştur
siniflar = []
dosya_dict = {}  # "sınıf" -> "dosya yolu"

if os.path.exists(dersprogram_klasor):
    for dosya in os.listdir(dersprogram_klasor):
        if dosya.lower().endswith(".png"):
            sinif = dosya.replace(".png", "").upper().replace(" ", "")
            siniflar.append(sinif)
            dosya_dict[sinif] = os.path.join(dersprogram_klasor, dosya)
    siniflar = sorted(siniflar)
else:
    st.sidebar.warning(f"📂 Klasör bulunamadı: {dersprogram_klasor}")

# Sidebar selectbox'ta 9A, 10 gibi gösterimleri "9 - A" gibi yap
def secim_gosterim_func(s):
    if len(s) == 2:
        return f"{s[0]} - {s[1]}"
    return s

secim_gosterim = [secim_gosterim_func(s) for s in siniflar]

if siniflar:
    secim_index = st.sidebar.selectbox(
        "Sınıfı seçin:",
        range(len(secim_gosterim)),
        format_func=lambda x: secim_gosterim[x]
    )
    secim = siniflar[secim_index]
    dosya_yolu = dosya_dict[secim]

    if os.path.exists(dosya_yolu):
        img = Image.open(dosya_yolu)
        st.sidebar.image(img, caption=f"{secim_gosterim[secim_index]} Ders Programı", use_column_width=True)
    else:
        st.sidebar.warning(f"{secim_gosterim[secim_index]} için görsel bulunamadı!")
else:
    st.sidebar.warning("📂 Klasörde PNG dosyası bulunamadı!")

# 🧠 VECTOR DB yükle
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

    arama_sorgusu = f"{soru} meb yönetmelik maddesi devamsızlık şartları"

    docs = vector_db.similarity_search_with_score(arama_sorgusu, k=3)
    docs = sorted(docs, key=lambda x: x[1])[:3]
    docs = [doc[0] for doc in docs]

    if not docs:
        return "Veri bulunamadı.", None, None

    baglam = "\n\n".join([doc.page_content[:500] for doc in docs])

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

    for msg in st.session_state.conversation:
        messages.append(msg)

    messages.append({"role": "user", "content": f"{baglam}\n\nSoru: {soru}"})

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="openai/gpt-oss-120b",
            temperature=0,
            max_tokens=500
        )

        if hasattr(chat_completion.choices[0], "message") and hasattr(chat_completion.choices[0].message, "content"):
            cevap = chat_completion.choices[0].message.content
        elif hasattr(chat_completion.choices[0], "text"):
            cevap = chat_completion.choices[0].text
        else:
            cevap = "⚠️ Yanıt alınamadı."
    except Exception as e:
        cevap = f"⚠️ API çağrısında hata oluştu: {e}"

    cevap = temizle_cevap(cevap)

    st.session_state.conversation.append({"role": "user", "content": soru})
    st.session_state.conversation.append({"role": "assistant", "content": cevap})

    tablo_data = {"Madde Özeti": [doc.page_content[:100]+"..." for doc in docs]}
    tablo_df = pd.DataFrame(tablo_data)
    kaynaklar = [doc.page_content[:200] for doc in docs]

    return cevap, tablo_df, kaynaklar

# 💬 Chat Arayüzü
for msg in st.session_state.conversation:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

if prompt := st.chat_input("Sorunuzu yazın:"):
    with st.spinner("Yanıt hazırlanıyor..."):
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
