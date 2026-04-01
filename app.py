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
    
    /* 🚀 Sınıf Programı Butonu (Sağ Altta Sabit) */
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
        border: 2px solid white !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        text-decoration: none !important;
        transition: 0.3s ease !important;
    }
    
    .stLinkButton a:hover {
        background-color: #E67E22 !important;
        transform: scale(1.05);
    }

    /* Rehber Kartı Stili */
    .category-box {
        background-color: rgba(128, 128, 128, 0.05);
        border-radius: 15px;
        padding: 18px;
        border-top: 4px solid #FF4B4B;
        height: 100%;
    }
    .category-title { font-weight: bold; color: #FF4B4B; margin-bottom: 10px; font-size: 1.15rem; }
    .category-item { font-size: 0.88rem; margin-bottom: 6px; color: #444; }
    
    @media (prefers-color-scheme: dark) {
        .category-item { color: #CCC; }
        .category-box { background-color: rgba(255, 255, 255, 0.05); }
    }
    
    [data-testid="stChatMessage"] { border-radius: 20px; border: 1px solid rgba(128, 128, 128, 0.15); }
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

# ❌ Güvenlik Filtresi
def uygunsuz_mu(soru):
    yasakli = ["küfür", "argo", "siyaset", "parti", "din", "ırk", "hakaret", "aptal", "salak"]
    return any(k in soru.lower() for k in yasakli)

# 🤖 Sorgulama Fonksiyonu
def sorgula(soru):
    docs = vector_db.similarity_search(soru, k=3)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    messages = [{
        "role": "system",
        "content": """Sen MEB yönetmelik uzmanısın. Ağırbaşlı ve resmi bir dille cevap ver.
        [PDF VERİLERİ]:
        - Devamsızlık: Özürsüz 10, Toplam 30 gün. (İstisna: 60 gün)
        - Başarı: Geçme notu 50.
        - Sınıf Geçme: Max 3 zayıf sorumlu geçer, 6+ zayıf tekrar.
        - Ödül: Teşekkür 70+, Takdir 85+.
        - Disiplin: Kopya/Sigara = Kınama.
        - Kayıt: Evli kayıt yapılamaz.

        ÖNEMLİ: Eğer soru yönetmelikle ilgili değilse veya saçmaysa, cevabına mutlaka 'KAPSAM_DISI' kelimesini ekle."""
    }]

    for msg in st.session_state.conversation[-4:]:
        messages.append(msg)
    messages.append({"role": "user", "content": f"BAĞLAM:\n{baglam}\n\nSORU: {soru}"})

    completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    return completion.choices[0].message.content, docs

# --- ARAYÜZ ---

# Başlık
st.markdown("<div class='main-title'>🏛️ MEB Mevzuat Uzmanı</div>", unsafe_allow_html=True)

# 🚀 SAĞ ALTTA YÜZEN BUTON
st.markdown('<div class="floating-button-container">', unsafe_allow_html=True)
st.link_button("📅 Sınıf Programı", "https://sinifprogrami.streamlit.app/")
st.markdown('</div>', unsafe_allow_html=True)

# 💡 Öneri Sorular (Sadece sohbet boşken görünür)
if not st.session_state.conversation:
    st.markdown("### 💡 Sorabileceğiniz Konular")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="category-box"><div class="category-title">📜 Disiplin</div><div class="category-item">• Kopya cezası?<br>• Sigara yasağı?<br>• Kınama nedir?</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="category-box"><div class="category-title">⏳ Devamsızlık</div><div class="category-item">• Özürsüz sınır?<br>• 30 gün kuralı?<br>• Sağlık izni?</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="category-box"><div class="category-title">🎓 Başarı</div><div class="category-item">• Kaç zayıfla kalınır?<br>• Takdir puanı?<br>• Nakil şartları?</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# 💬 Sohbet Akışı
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        # KAPSAM_DISI etiketini gizle
        st.markdown(msg["content"].replace("KAPSAM_DISI", ""))

# ⌨️ Giriş Alanı
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    
    # Uygunsuz içerik kontrolü
    if uygunsuz_mu(prompt):
        hata_mesaji = "⚠️ Lütfen topluluk kurallarına uygun ve sadece MEB yönetmeliği ile ilgili sorular sorunuz."
        st.session_state.conversation.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"): st.markdown(hata_mesaji)
        st.session_state.conversation.append({"role": "assistant", "content": hata_mesaji})
    
    else:
        st.session_state.conversation.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("⚖️ İnceleniyor..."):
                raw_cevap, kaynaklar = sorgula(prompt)
                
                # Tablo ve içerik kontrolü
                is_out_of_scope = "KAPSAM_DISI" in raw_cevap
                clean_cevap = raw_cevap.replace("KAPSAM_DISI", "").strip()
                
                st.markdown(clean_cevap)
                
                # Sadece mevzuat dahilindeyse tabloyu göster
                if kaynaklar and not is_out_of_scope:
                    st.markdown("---")
                    st.markdown("📑 **Yönetmelik Referans Tablosu**")
                    ref_df = pd.DataFrame({
                        "Kaynak": [f"Madde {i+1}" for i in range(len(kaynaklar))],
                        "İçerik": [doc.page_content[:250] + "..." for doc in kaynaklar]
                    })
                    st.table(ref_df)
                
                st.session_state.conversation.append({"role": "assistant", "content": clean_cevap})
