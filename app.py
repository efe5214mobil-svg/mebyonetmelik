# app.py
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from rag import okul_asistani_sorgula  # rag.py'deki basit fonksiyon

# 🔐 .env yükle
load_dotenv()

# Başlık
st.title("MEB Yönetmelik Asistanı")
st.info("⚠️ Sadece MEB yönetmeliği ile ilgili sorular sorabilirsiniz.")

# VECTOR DB
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory="okul_asistani_gpt_db",
        embedding_function=embeddings
    )
    return db

vector_db = load_vector_db()

# Session state ile sohbet geçmişi
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Chat arayüzü
for msg in st.session_state.conversation:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

if prompt := st.chat_input("Sorunuzu yazın:"):
    with st.spinner("Yanıt hazırlanıyor..."):
        cevap, kaynaklar = okul_asistani_sorgula(prompt, vector_db)
        st.session_state.conversation.append({"role": "user", "content": prompt})
        st.session_state.conversation.append({"role": "assistant", "content": cevap})

        with st.chat_message("assistant"):
            st.markdown(cevap)
            if kaynaklar:
                st.markdown("📚 Kaynaklar:")
                for k in kaynaklar:
                    st.markdown(f"- {k}")
