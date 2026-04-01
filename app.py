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

# 🖌️ Modern CSS
st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }
    .main-title { font-size: 2.2rem; font-weight: 800; text-align: center; margin-bottom: 1.5rem; }
    
    /* 🚀 Sağ Altta Yüzen Buton */
    .floating-button-container {
        position: fixed;
        bottom: 85px; 
        right: 10%; 
        z-index: 999999;
    }

    .stLinkButton a {
        background-color: #FF8C00 !important;
        color: white !important;
        border-radius: 25px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 700 !important;
        text-decoration: none !important;
        border: 2px solid white !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }

    /* Öneri Kartları Stili */
    .category-box {
        background-color: rgba(128, 128, 128, 0.05);
        border-radius: 15px;
        padding: 18px;
        border-top: 4px solid #FF4B4B;
        height: 100%;
        transition: 0.3s;
    }
    .category-box:hover { transform: translateY(-5px); background-color: rgba(255, 75, 75, 0.05); }
    .category-title { font-weight: bold; color: #FF4B4B; margin-bottom: 10px; font-size: 1.15rem; }
    .category-item { font-size: 0.88rem; margin-bottom: 6px; color: #444; cursor: pointer; }
</style>
""", unsafe_allow_html=True)

# 🧠 Vektör Veritabanı
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="okul_asistani_gpt_db", embedding_function=embeddings)

vector_db = load_vector_db()

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# 🤖 Sorgulama Fonksiyonu (Tam Donanımlı)
def sorgula(soru):
    # Vektör DB'den en alakalı 3 maddeyi getir
    docs = vector_db.similarity_search(soru, k=3)
    baglam = "\n\n".join([doc.page_content for doc in docs])
    
    messages = [{
        "role": "system", 
        "content": """Sen MEB yönetmelik uzmanısın. Resmi ve ciddi bir dille cevap ver.
        
        [SABİT PDF VERİLERİ]:
        - Devamsızlık: Özürsüz 10, Toplam 30 gün. (İstisna durumlar: 60 gün)
        - Başarı Notu: Geçme sınırı 50 puan.
        - Sınıf Geçme: En fazla 3 dersten sorumlu geçilebilir, 6+ zayıfta sınıf tekrarı.
        - Ödül: Teşekkür belgesi 70.00+, Takdir belgesi 85.00+.
        - Disiplin: Kopya çekmek veya sigara içmek 'Kınama' cezası gerektirir.
        - Kayıt: Evli olanların kayıtları yapılamaz, kayıtlıyken evlenenlerin ilişiği kesilir.

        KURAL: Soru selamlaşma, şaka veya mevzuat dışıysa cevabın sonuna [TABLO_YOK] etiketini ekle. 
        Mevzuatla ilgiliyse cevabını ver ve etiket ekleme."""
    }]
    
    for msg in st.session_state.conversation[-4:]:
        messages.append(msg)
    
    messages.append({"role": "user", "content": f"BAĞLAM (PDF Maddeleri):\n{baglam}\n\nSORU: {soru}"})
    
    completion = client.chat.completions.create(
        messages=messages, 
        model="llama-3.3-70b-versatile", 
        temperature=0
    )
    return completion.choices[0].message.content, docs

# --- ARAYÜZ TASARIMI ---
st.markdown("<div class='main-title'>🏛️ MEB Mevzuat Uzmanı</div>", unsafe_allow_html=True)

# 🚀 SAĞ ALT BUTON
st.markdown('<div class="floating-button-container">', unsafe_allow_html=True)
st.link_button("📅 Sınıf Programı", "https://senin-linkin-buraya.com")
st.markdown('</div>', unsafe_allow_html=True)

# 💡 ÖNERİ SORU KARTLARI (Sohbet boşsa göster)
if not st.session_state.conversation:
    st.markdown("### 💡 Hızlı Öneriler")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="category-box"><div class="category-title">📜 Disiplin</div><div class="category-item">"Kopya çekmenin cezası nedir?"</div><div class="category-item">"Disiplin cezaları nelerdir?"</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="category-box"><div class="category-title">⏳ Devamsızlık</div><div class="category-item">"Kaç gün devamsızlık hakkım var?"</div><div class="category-item">"30 gün kuralı nedir?"</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="category-box"><div class="category-title">🎓 Başarı</div><div class="category-item">"Kaç zayıfla sınıfta kalınır?"</div><div class="category-item">"Takdir belgesi kaç puan?"</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# 💬 SOHBET AKIŞI
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("[TABLO_YOK]", ""))

# ⌨️ GİRİŞ ALANI
if prompt := st.chat_input("Yönetmelik hakkında bir soru sorun..."):
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("⚖️ Mevzuat inceleniyor..."):
            cevap, kaynaklar = sorgula(prompt)
            
            # Tablo gösterilsin mi kontrolü
            tablo_gizle = "[TABLO_YOK]" in cevap
            temiz_cevap = cevap.replace("[TABLO_YOK]", "").strip()
            
            st.markdown(temiz_cevap)
            
            # Eğer konu mevzuat ise tabloyu ve referans bağlantısını bas
            if kaynaklar and not tablo_gizle:
                st.markdown("---")
                st.markdown("📑 **İlgili Mevzuat Referansları**")
                
                # Tablo verisini hazırla
                ref_data = []
                for i, doc in enumerate(kaynaklar):
                    # Vektör DB'de sayfa numarası varsa çek, yoksa 'Belirtilmemiş' yaz
                    sayfa = doc.metadata.get('page', 'PDF Kaynağı')
                    ref_data.append({
                        "Kaynak": f"Madde {i+1}",
                        "İçerik Özeti": doc.page_content[:200] + "...",
                        "Konum": sayfa
                    })
                
                st.table(pd.DataFrame(ref_data))
            
            st.session_state.conversation.append({"role": "assistant", "content": temiz_cevap})
