import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from dotenv import load_dotenv
import os
import pandas as pd

# 🔐 API ve Çevre Değişkenleri
load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

# 🎨 Sayfa Yapılandırması
st.set_page_config(page_title="MEB Mevzuat Asistanı", page_icon="🏛️", layout="centered")

# 🖌️ CSS - Modern Arayüz ve Tablo Tasarımı
st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }
    .main-title { font-size: 2.2rem; font-weight: 800; text-align: center; margin-bottom: 0.5rem; }
    
    /* Rehber ve Cevap Tablosu Tasarımı */
    .guide-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 0.9rem;
    }
    .guide-table td, .guide-table th {
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 10px;
    }
    .guide-header {
        background-color: rgba(255, 75, 75, 0.1);
        font-weight: bold;
        color: #FF4B4B;
    }
    
    [data-testid="stChatMessage"] {
        border-radius: 15px;
        padding: 15px;
        border: 1px solid rgba(128, 128, 128, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# 🧠 Vektör Veritabanı Yükleme
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="okul_asistani_gpt_db", embedding_function=embeddings)

vector_db = load_vector_db()

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# ❌ Güvenlik Filtresi
def uygunsuz_mu(soru):
    yasakli = ["küfür", "argo", "siyaset", "parti", "din", "ırk", "hakaret"]
    return any(k in soru.lower() for k in yasakli)

# 🤖 Sorgulama Fonksiyonu (Llama 3.3 70B)
def sorgula(soru):
    docs = vector_db.similarity_search(soru, k=3)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    messages = [{
        "role": "system",
        "content": """Sen MEB yönetmelik uzmanısın. 
        Sana bir soru sorulduğunda cevabını mutlaka şu iki bölüme ayır:
        1. 'Açıklama': Sorunun net ve resmi cevabı.
        2. 'Yönetmelik Dayanağı': Cevabın hangi kurala dayandığının kısa özeti.

        [PDF KRİTİK VERİLER]:
        - Devamsızlık: Özürsüz 10, Toplam 30 gün.
        - Başarı: En az 50 puan.
        - Sınıf Geçme: Max 3 zayıf sorumlu, 6+ zayıf tekrar.
        - Ödül: Teşekkür 70+, Takdir 85+.
        - Ders: 40 dakika.
        """
    }]

    for msg in st.session_state.conversation[-4:]:
        messages.append(msg)

    messages.append({"role": "user", "content": f"BAĞLAM:\n{baglam}\n\nSORU: {soru}"})

    try:
        completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0
        )
        return completion.choices[0].message.content, docs
    except Exception as e:
        return f"Hata: {str(e)}", None

# --- ARAYÜZ ---
st.markdown("<div class='main-title'>🏛️ MEB Mevzuat Uzmanı</div>", unsafe_allow_html=True)

# 💡 Giriş Rehberi (Soru sormadan önce görünür)
if not st.session_state.conversation:
    st.markdown("#### 💡 Neler Sorabilirsiniz?")
    st.markdown("""
    <table class="guide-table">
        <tr class="guide-header"><td>📂 Kategori</td><td>❓ Örnek Soru</td></tr>
        <tr><td><b>Devamsızlık</b></td><td>Özürsüz devamsızlık sınırı nedir?</td></tr>
        <tr><td><b>Sınıf Geçme</b></td><td>Kaç zayıfla sınıf tekrarı yapılır?</td></tr>
        <tr><td><b>Ödüller</b></td><td>Takdir belgesi puan şartı nedir?</td></tr>
    </table>
    """, unsafe_allow_html=True)

# 💬 Sohbet Geçmişi
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ⌨️ Giriş Alanı
if prompt := st.chat_input("Sorunuzu yazın..."):
    if uygunsuz_mu(prompt):
        st.error("Uygunsuz içerik tespit edildi.")
        st.stop()

    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("İnceleniyor..."):
            cevap, kaynaklar = sorgula(prompt)
            
            # 📊 CEVAP VE REFERANS TABLOSU OLUŞTURMA
            st.markdown(cevap)
            
            if kaynaklar:
                st.markdown("### 📋 Yönetmelik Referans Tablosu")
                # DataFrame ile tablo oluşturma
                ref_data = {
                    "Kaynak": [f"Madde Kesiti {i+1}" for i in range(len(kaynaklar))],
                    "Resmi İçerik": [doc.page_content[:200] + "..." for doc in kaynaklar]
                }
                st.table(pd.DataFrame(ref_data))
            
            st.session_state.conversation.append({"role": "assistant", "content": cevap})
