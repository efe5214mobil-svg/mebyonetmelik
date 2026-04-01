import streamlit as st
import os
import re
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rag import okul_asistani_sorgula

# 🔐 Başlatma
load_dotenv()

@st.cache_resource
def veri_tabanini_yukle():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="okul_asistani_gpt_db", embedding_function=embeddings)

vektor_tabani = veri_tabanini_yukle()

# 🛡️ GELİŞMİŞ ÇELİK ZIRH SÜZGECİ
def suzgec_kontrolu(metin):
    # Karakter hilelerini temizle (1=i, 4=a vb.)
    harita = {'1':'i', '0':'o', '3':'e', '4':'a', '5':'s', '7':'t', '8':'b', '@':'a', '$':'s'}
    temiz = metin.lower()
    for eski, yeni in harita.items():
        temiz = temiz.replace(eski, yeni)
    
    # Boşlukları ve özel karakterleri silerek sıkıştır
    sıkı_metin = re.sub(r'[^a-z0-9çşğüöı]', '', temiz)
    
    # Kapsamlı Yasaklı Listesi
    yasakli = [
        "oc", "aq", "amk", "amq", "pic", "got", "sik", "amc", "yarrak", "orospu", 
        "seks", "porno", "gay", "lezbiyen", "lgbt", "erdogan", "tayyip", "siyaset", 
        "parti", "ataturk", "teror", "darbe", "bebegim", "askim"
    ]
    return any(kelime in sıkı_metin for kelime in yasakli)

# 🖌️ TASARIM (Mavi-Beyaz Temalı)
st.set_page_config(page_title="Okul Asistanı GPT", layout="centered")

st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }
    .ana-baslik { font-size: 2.5rem; font-weight: 800; text-align: center; color: #FFFFFF; }
    
    /* Mavi-Beyaz Yönlendirme Butonu */
    .yuzen-buton-alani {
        position: fixed;
        bottom: 85px; 
        right: 8%; 
        z-index: 9999;
    }
    .stLinkButton a {
        background-color: #1E90FF !important; /* Mavi Arka Plan */
        color: white !important;               /* Beyaz Yazı */
        border-radius: 30px !important;
        padding: 0.8rem 1.8rem !important;
        font-weight: 700 !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4) !important;
    }

    .kategori-kutusu {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        border-top: 4px solid #1E90FF;
        margin-bottom: 10px;
    }
    .kategori-basligi { font-weight: bold; color: #1E90FF; }
</style>
""", unsafe_allow_html=True)

# --- ARAYÜZ ---
st.markdown("<div class='ana-baslik'>📅 Okul Ders Programı Asistanı</div>", unsafe_allow_html=True)

# Yönlendirme Butonu (Tıklandığında MEB Yönetmelik sitesine gider)
st.markdown('<div class="yuzen-buton-alani">', unsafe_allow_html=True)
st.link_button("🏛️ MEB Yönetmelik", "https://meb-yonetmelik.streamlit.app/")
st.markdown('</div>', unsafe_allow_html=True)

# 💡 Hızlı Sorular (Öneri Kartları)
c1, c2, c3 = st.columns(3)
with c1: st.markdown('<div class="kategori-kutusu"><div class="kategori-basligi">🔍 Sınıflar</div>9-A Pazartesi dersleri neler?</div>', unsafe_allow_html=True)
with c2: st.markdown('<div class="kategori-kutusu"><div class="kategori-basligi">👨‍🏫 Hocalar</div>U.TRK dersi hangi saatte?</div>', unsafe_allow_html=True)
with c3: st.markdown('<div class="kategori-kutusu"><div class="kategori-basligi">📚 Dersler</div>Bilişim dersi hangi gün?</div>', unsafe_allow_html=True)

st.markdown("---")

if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if girdi := st.chat_input("Ders programı hakkında bir soru sorun..."):
    if suzgec_kontrolu(girdi):
        msg = st.error("⚠️ Güvenlik Süzgeci: Lütfen uygun bir dil kullanın.")
        time.sleep(2); msg.empty()
    else:
        st.session_state.messages.append({"role": "user", "content": girdi})
        with st.chat_message("user"): st.markdown(girdi)

        with st.chat_message("assistant"):
            with st.spinner("⏳ İnceleniyor..."):
                cevap, kaynaklar = okul_asistani_sorgula(girdi, vektor_tabani)
                st.markdown(cevap)
                if kaynaklar:
                    with st.expander("📌 Kaynak Bilgi"):
                        for k in kaynaklar: st.caption(k)
                st.session_state.messages.append({"role": "assistant", "content": cevap})
                st.rerun()
