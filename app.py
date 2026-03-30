import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import re
import pandas as pd
from PIL import Image

# 🎨 Sayfa Ayarları ve Gradyanlı Koyu Tema
st.set_page_config(page_title="MEB Yönetmelik Asistanı", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 50%, #2c3e50 100%);
        color: #ffffff;
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.8) !important;
    }
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    /* Input alanını koyulaştır */
    .stTextInput input {
        background-color: #252525 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="okul_asistani_gpt_db", embedding_function=embeddings)

vector_db = load_vector_db()

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# --- SIDEBAR & KESİN İSİM TEMİZLEME (12 - A Formatı) ---
st.sidebar.header("📌 Sınıf Panosu")
dersprogram_klasor = "dersprogram_dosyasi"
dosya_haritasi = {}

if os.path.exists(dersprogram_klasor):
    dosyalar = [f for f in os.listdir(dersprogram_klasor) if f.lower().endswith(".png")]
    for d in dosyalar:
        # Uzantıyı ve gereksiz karakterleri temizle
        isim_ham = d.lower().replace(".png", "").replace(" ", "").replace("-", "").replace(".", "")
        # Rakam ve harf gruplarını bul (Örn: 12a)
        match = re.search(r"(\d+)([a-z]+)", isim_ham)
        if match:
            sayi = match.group(1)
            harf = match.group(2).upper()
            gosterim_adi = f"{sayi} - {harf}"
            dosya_haritasi[gosterim_adi] = os.path.join(dersprogram_klasor, d)

    # 12'den 9'a doğru büyükten küçüğe sırala
    def sirala_key(x):
        r = re.search(r'\d+', x)
        return -int(r.group()) if r else 0

    sirali_isimler = sorted(dosya_haritasi.keys(), key=sirala_key)

    if sirali_isimler:
        secilen_sinif = st.sidebar.selectbox("Sınıfı seçin:", sirali_isimler)
        st.sidebar.image(Image.open(dosya_haritasi[secilen_sinif]), caption=f"{secilen_sinif} Programı", use_container_width=True)
else:
    st.sidebar.error("Klasör bulunamadı!")

# 🛡️ GELİŞMİŞ GÜVENLİK VE ALAKASIZ SORU FİLTRESİ
def uygunsuz_mu(soru):
    # 1. Küfür, Argo, Siyaset, Din, Irk Filtresi
    yasakli_kelimeler = [
        "küfür", "argo", "siyaset", "din", "ırk", "dil", "mezhep", "parti", 
        "hükümet", "seçim", "türkiye", "ekonomi", "hakaret", "aptal", "gerizekalı"
    ]
    # 2. Saçma sapan/Alakasız sorular (Gündelik muhabbet)
    alakasiz_kelimeler = ["naber", "nasılsın", "günaydın", "selam", "kimsin", "napıyorsun", "hava", "maç", "fenerbahçe", "galatasaray", "beşiktaş"]
    
    soru_low = soru.lower()
    
    # Yasaklı kelime kontrolü
    if any(k in soru_low for k in yasakli_kelimeler):
        return True, "Uygunsuz içerik (küfür, siyaset, din vb.) tespit edildi."
    
    # Çok kısa ve alakasız soruları engelle (Yönetmelik asistanı olduğu için)
    if len(soru_low.split()) < 3 and any(k in soru_low for k in alakasiz_kelimeler):
        return True, "Lütfen sadece MEB yönetmeliği ile ilgili teknik sorular sorunuz."
    
    return False, ""

# 🤖 SORGULAMA (Detaycı & Güvenli)
def okul_asistani_sorgula(soru):
    hata_var, mesaj = uygunsuz_mu(soru)
    if hata_var:
        return f"⚠️ **Sistem Engeli:** {mesaj}", None

    # Vektör veritabanında ara
    docs = vector_db.similarity_search(soru, k=4)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        {
            "role": "system", 
            "content": """Sen ciddi bir MEB Yönetmelik Uzmanısın.
            SADECE sana verilen bağlamdaki dökümanlara göre cevap ver.
            Eğer soru yönetmelik, okul kuralları, sınavlar veya disiplinle ilgili değilse 'Bu soru yönetmelik kapsamı dışındadır' de.
            Cevaplarına mutlaka EVET veya HAYIR ile başla, sonra yönetmelik maddelerine dayanarak detaylıca açıkla.
            Dökümanda olmayan hiçbir bilgiyi uydurma. Siyaset, din veya günlük sohbet yapma."""
        }
    ]
    # Son 2 mesajı hafıza olarak gönder
    messages.extend(st.session_state.conversation[-2:])
    messages.append({"role": "user", "content": f"Bağlam:\n{baglam}\n\nSoru: {soru}"})

    try:
        completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=1000
        )
        cevap = completion.choices[0].message.content
        return cevap, docs
    except Exception as e:
        return f"Şu an yanıt veremiyorum. Hata: {str(e)}", None

# --- ANA EKRAN ---
st.title("🎓 MEB Yönetmelik Asistanı")
st.info("Sadece MEB Ortaöğretim Kurumları Yönetmeliği hakkında resmi bilgi verir.")

# Sohbet geçmişini ekrana bas
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Kullanıcı girişi
if prompt := st.chat_input("Sorunuzu buraya yazın (Örn: Geç kalma kuralı nedir?)..."):
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mevzuat inceleniyor..."):
            cevap, kaynaklar = okul_asistani_sorgula(prompt)
            st.markdown(cevap)
            
            if kaynaklar:
                with st.expander("📚 Kaynak Yönetmelik Metinleri"):
                    for k in kaynaklar:
                        st.caption(k.page_content)
        
        st.session_state.conversation.append({"role": "assistant", "content": cevap})
