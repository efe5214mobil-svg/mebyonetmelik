import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import re

# 🎨 Sayfa Ayarları
st.set_page_config(page_title="MEB Yönetmelik Asistanı", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 50%, #2c3e50 100%);
        color: #ffffff;
    }
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTextInput input {
        background-color: #252525 !important;
        color: white !important;
        border: 1px solid #444 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 🔐 API
load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

# 🧠 VECTOR DB
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="okul_asistani_gpt_db", embedding_function=embeddings)

vector_db = load_vector_db()

# 💬 Hafıza
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# 🛡️ Basit filtre
def uygunsuz_mu(soru):
    yasakli = ["küfür", "argo", "hakaret", "porno", "sex"]
    s = soru.lower()
    return any(k in s for k in yasakli)

# 🤖 Sorgu
def okul_asistani_sorgula(soru):
    if uygunsuz_mu(soru):
        return "⚠️ Bu ifade uygun değil."

    docs = vector_db.similarity_search(soru, k=5)
    baglam = "\n\n".join([d.page_content for d in docs])

    messages = [
        {
            "role": "system",
            "content": """Sen MEB Yönetmelik Asistanısın.
Kurallar:
- Sadece verilen bağlama göre cevap ver
- Resmi dil kullan
- Kısa ve net cevap ver"""
        }
    ]

    messages.extend(st.session_state.conversation[-2:])
    messages.append({"role": "user", "content": f"{baglam}\n\nSoru: {soru}"})

    try:
        completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=800
        )
        return completion.choices[0].message.content
    except:
        return "⚠️ Şu an yanıt verilemiyor."

# 🎓 Başlık
st.title("🎓 MEB Yönetmelik Asistanı")

# 💬 Chat geçmişi
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 💬 Input
if prompt := st.chat_input("Sorunuzu yazın..."):
    st.session_state.conversation.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Yanıt hazırlanıyor..."):
            cevap = okul_asistani_sorgula(prompt)
            st.write(cevap)

    st.session_state.conversation.append({"role": "assistant", "content": cevap})
