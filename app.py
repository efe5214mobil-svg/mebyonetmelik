import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import re
import base64
import pandas as pd
from PIL import Image

# 🎨 Sayfa Ayarları ve Tema
st.set_page_config(page_title="MEB Yönetmelik Asistanı", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 50%, #2c3e50 100%);
        color: #ffffff;
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.8) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
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
    .stTable {
        background-color: rgba(255, 255, 255, 0.05);
        color: white;
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

# --- GÖRSEL ÖZET FONKSİYONU ---
def gorsel_mevzuat_ozeti(docs):
    data = []
    for doc in docs:
        content = doc.page_content
        madde_match = re.search(r"MADDE\s+\d+", content, re.IGNORECASE)
        madde_no = madde_match.group(0) if madde_match else "Genel Hüküm"
        temiz_icerik = content.replace("\n", " ").strip()
        ozet = temiz_icerik[:200] + "..." if len(temiz_icerik) > 200 else temiz_icerik
        data.append({"Dayanak": madde_no, "Resmi İçerik Özeti": ozet})
    
    df = pd.DataFrame(data)
    st.markdown("#### 📋 Mevzuat Analiz Çizelgesi")
    st.table(df)

# --- SIDEBAR & SINIFLAR (12->9 ve A->Z) ---
st.sidebar.header("📌 Sınıflar")
dersprogram_klasor = "dersprogram_dosyasi"
dosya_haritasi = {}

if os.path.exists(dersprogram_klasor):
    dosyalar = [f for f in os.listdir(dersprogram_klasor) if f.lower().endswith(".png")]
    for d in dosyalar:
        isim_ham = d.lower().replace(".png", "").replace(" ", "").replace("-", "").replace(".", "")
        match = re.search(r"(\d+)([a-z]+)", isim_ham)
        if match:
            sayi = match.group(1)
            harf = match.group(2).upper()
            gosterim_adi = f"{sayi} - {harf}"
            dosya_haritasi[gosterim_adi] = os.path.join(dersprogram_klasor, d)

    def sirala_mantigi(x):
        parcalar = re.search(r"(\d+) - ([A-Z]+)", x)
        if parcalar:
            return (-int(parcalar.group(1)), parcalar.group(2))
        return (0, x)

    sirali_isimler = sorted(dosya_haritasi.keys(), key=sirala_mantigi)
    if sirali_isimler:
        secilen_sinif = st.sidebar.selectbox("Sınıf Seçin:", sirali_isimler)
        st.sidebar.image(Image.open(dosya_haritasi[secilen_sinif]), use_container_width=True)

# 🛡️ MAKSİMUM GÜVENLİK FİLTRESİ (Base64)
def uygunsuz_mu(soru):
    # Bu liste; küfür, ırkçılık, +18 ve hakaret içeren tüm kelimelerin Base64 halidir.
    data_enc = "a3VmdXIsYXJnbyxzaXlhc2V0LGRpbixpcmssaGFrYXJldCxwYXJ0aSxzZXgsc2Vrcyxwb3JubyxjaXBsYWssbWVtZSxnb3Qsc2lrLGFtayxwaXBpLHRhY2l6LG11c3RlaGNlbixnYXksbGV6Yml5ZW4sZmV0aXNsdWssdmFnaW5hLHBlbmlzLGVzY29ydCxuYWJlcixuYXNpbHNpbixzZWxhbSxtYWMsaGF2YSxmZW5lcmJhaGNlLGdhbGF0YXNhcmF5LG9jLGFtLGdvdCxwaWNoLHNpY21payx5YXJyYWssaXJramksZmFzaXN0LGthem9rLGdhdnVy"
    yasakli_liste = base64.b64decode(data_enc).decode('utf-8').split(',')
    s = soru.lower().strip()
    
    for kelime in yasakli_liste:
        if kelime in s: # Kelime cümlenin herhangi bir yerinde geçiyorsa blokla
            return True, "⚠️ **Güvenlik Engeli:** Girdiğiniz ifade akademik etik kurallara, topluluk ilkelerine veya MEB yönetmelik kapsamına uygun değildir."
    return False, ""

# 🤖 SORGULAMA (MEB Yönetmelik Uzmanı)
def okul_asistani_sorgula(soru):
    hata, mesaj = uygunsuz_mu(soru)
    if hata: return mesaj, None

    docs = vector_db.similarity_search(soru, k=5)
    baglam = "\n\n".join([d.page_content for d in docs])

    messages = [
        {
            "role": "system", 
            "content": """Sen Türkiye Cumhuriyeti MEB Yönetmelik Asistanısın. 
            Görevin SADECE okul yönetmeliği, sınavlar, devamsızlık ve disiplin gibi konularda bilgi vermektir.
            
            KESİN KURALLAR:
            1. Siyaset, din, ırkçılık, cinsellik veya hakaret içeren hiçbir konuya girme.
            2. 'Evet' veya 'Hayır' diyerek başlama. Direkt resmi veriyi paylaş.
            3. 'Sorumluluk Sınavı' gibi Türkçe terimler kullan.
            4. Özürlü devamsızlık sınırı 20 GÜNDÜR. Toplam 30 gündür.
            5. Sınıf geçme barajı 50 ortalamadır. 3'ten fazla zayıfı olan (ortalama 50 altıysa) kalır."""
        }
    ]
    messages.extend(st.session_state.conversation[-2:])
    messages.append({"role": "user", "content": f"BAĞLAM:\n{baglam}\n\nSORU: {soru}"})

    try:
        completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=1000
        )
        return completion.choices[0].message.content, docs
    except:
        return "Sistem şu an meşgul, lütfen yönetmelik çerçevesinde tekrar deneyin.", None

# --- ANA EKRAN ---
st.title("🎓 MEB Yönetmelik Asistanı")
st.caption("Resmi Ortaöğretim Kurumları Yönetmeliği Veri Tabanı")

with st.expander("❓ Sıkça Sorulan Sorular"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Devamsızlık ve Sorumluluk**")
        st.write("- Özürlü devamsızlık hakkı kaç gündür?")
        st.write("- Ortalamam 50 altındaysa sorumlu geçer miyim?")
    with c2:
        st.markdown("**Disiplin ve Kurallar**")
        st.write("- Okulda telefon yakalatmanın cezası nedir?")
        st.write("- Nakil şartları nelerdir?")

for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if prompt := st.chat_input("Yönetmelik sorunuzu yazın..."):
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mevzuat inceleniyor..."):
            cevap, kaynaklar = okul_asistani_sorgula(prompt)
            st.markdown(cevap)
            
            if kaynaklar:
                with st.expander("📖 Dayanak Yönetmelik Maddeleri (Görsel Analiz)"):
                    gorsel_mevzuat_ozeti(kaynaklar)
                    st.divider()
                    for k in kaynaklar:
                        st.caption(f"📍 {k.page_content}")
        
        st.session_state.conversation.append({"role": "assistant", "content": cevap})
