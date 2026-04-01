import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbedembeddings 
from dotenv import load_dotenv
import os
import re
import time 

# 🔐 API ve Çevre Değişkenleri
load_dotenv()
api_anahtari = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
istemci = Groq(api_key=api_anahtari)

# 🎨 Sayfa Yapılandırması
st.set_page_config(page_title="MEB Mevzuat Asistanı", page_icon="🏛️", layout="centered")

# 🖌️ Modern Görünüm (CSS)
st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }
    .ana-baslik { font-size: 2.5rem; font-weight: 800; text-align: center; margin-bottom: 1.5rem; color: #FFFFFF; }
    
    .yuzen-buton-alani {
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

    .kategori-kutusu {
        background-color: rgba(128, 128, 128, 0.05);
        border-radius: 15px;
        padding: 18px;
        border-top: 4px solid #FF4B4B;
        height: 100%;
        margin-bottom: 10px;
    }
    .kategori-basligi { font-weight: bold; color: #FF4B4B; margin-bottom: 10px; font-size: 1.15rem; }
    .kategori-maddesi { font-size: 0.88rem; margin-bottom: 6px; color: #444; }
</style>
""", unsafe_allow_html=True)

# 🧠 Veri Tabanı Yükleme
@st.cache_resource
def veri_tabanini_yukle():
    gomme_modeli = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="okul_asistani_gpt_db", embedding_function=gomme_modeli)

vektor_tabani = veri_tabanini_yukle()

# 🛡️ ÇELİK ZIRHLI GÜVENLİK SÜZGECİ
def suzgec_kontrolu(metin):
    # Karakter dönüşümü (Hileleri engellemek için)
    karakter_haritasi = {'1': 'i', '0': 'o', '3': 'e', '4': 'a', '5': 's', '7': 't', '8': 'b', '@': 'a', '$': 's'}
    
    temiz_metin = metin.lower()
    for eski, yeni in karakter_haritasi.items():
        temiz_metin = temiz_metin.replace(eski, yeni)
    
    # Boşlukları ve sembolleri kaldırarak kontrol et
    sikistirilmis_metin = re.sub(r'[^a-z0-9çşğüöı]', '', temiz_metin)
    
    yasakli_kelimeler = [
        # Küfür, Argo, Laubali
        "oc", "aq", "amk", "amq", "pic", "got", "sik", "amc", "yarrak", "orospu", "bebegim", "askim",
        # Irkçılık & Ayrımcılık
        "nigga", "zenci", "cikolata", "irkci", "yahudi", "ermeni", "nazi",
        # Siyaset & Devlet Büyükleri
        "erdogan", "tayyip", "rte", "cumhurbaskani", "akp", "chp", "mhp", "siyaset", "parti", "darbe",
        # Tarihsel Hükümdarlar & Hanedanlar (İstediğin Kısım)
        "mahmud", "charles", "suleyman", "fatih", "kanuni", "padisah", "kral", "imparator", "osmanli", "bizans", "roma",
        "iimahmud", "ivcharles", "ataturk", "hitler", "stalin", "lenin",
        # Konu Dışı Akımlar
        "modernizm", "narsizm", "narsist", "nihilizm", "ideoloji", "1945", "1939", "savas",
        # Cinsellik
        "gay", "lezbiyen", "lgbt", "seks", "sex", "porno", "vajina", "penis"
    ]
    
    # Roma rakamı içeren kalıpları yakalamak için (Örn: "II. Mahmud" -> "iimahmud")
    return any(yasakli in sikistirilmis_metin for yasakli in yasakli_kelimeler)

if "sohbet_gecmisi" not in st.session_state:
    st.session_state.sohbet_gecmisi = []

# 🤖 Yanıt Oluşturucu
def cevap_olustur(soru):
    ilgili_belgeler = vektor_tabani.similarity_search(soru, k=5)
    kaynak_metin = "\n\n".join([belge.page_content for belge in ilgili_belgeler])
    
    iletiler = [{
        "role": "system", 
        "content": """Sen uzman bir MEB Mevzuat Asistanısın. 
        SADECE okul yönetmeliği, dersler, devamsızlık ve disiplin kuralları hakkında bilgi ver. 
        Tarih, siyaset, hükümdarlar veya felsefi akımlar hakkında konuşma. 
        Daima Türkçe ve profesyonel bir dil kullan."""
    }]
    
    for ileti in st.session_state.sohbet_gecmisi[-4:]:
        iletiler.append(ileti)
    
    iletiler.append({"role": "user", "content": f"KAYNAK:\n{kaynak_metin}\n\nSORU: {soru}"})
    
    yanit = istemci.chat.completions.create(
        messages=iletiler, 
        model="llama-3.3-70b-versatile", 
        temperature=0.1
    )
    return yanit.choices[0].message.content

# --- ARAYÜZ ---
st.markdown("<div class='ana-baslik'>🏛️ MEB Yönetmelik Asistanı</div>", unsafe_allow_html=True)

st.markdown('<div class="yuzen-buton-alani">', unsafe_allow_html=True)
st.link_button("📅 Sınıf Programı", "https://sinifprogrami.streamlit.app/")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### 💡 Hızlı Sorular")
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown('<div class="kategori-kutusu"><div class="kategori-basligi">📜 Kayıt & Disiplin</div><div class="kategori-maddesi">• Evlilik durumu?<br>• Kopya cezası?</div></div>', unsafe_allow_html=True)
with s2:
    st.markdown('<div class="kategori-kutusu"><div class="kategori-basligi">⏳ Devamsızlık</div><div class="kategori-maddesi">• 30 gün kuralı?<br>• Geç gelme sınırı?</div></div>', unsafe_allow_html=True)
with s3:
    st.markdown('<div class="kategori-kutusu"><div class="kategori-basligi">🎓 Başarı & Nakil</div><div class="kategori-maddesi">• Sınıf tekrarı?<br>• Beceri sınavı?</div></div>', unsafe_allow_html=True)
st.markdown("---")

for ileti in st.session_state.sohbet_gecmisi:
    with st.chat_message(ileti["role"]):
        st.markdown(ileti["content"])

if girdi := st.chat_input("Yönetmelik hakkında bir soru sorun..."):
    
    if suzgec_kontrolu(girdi):
        uyari_alani = st.empty()
        uyari_alani.error("⚠️ Uyarı: İletiniz yönetmelik dışı, siyasi veya tarihsel içerik barındırdığı için engellenmiştir.")
        time.sleep(2) 
        uyari_alani.empty() 
        st.rerun() 
    else:
        st.session_state.sohbet_gecmisi.append({"role": "user", "content": girdi})
        with st.chat_message("user"):
            st.markdown(girdi)

        with st.chat_message("assistant"):
            with st.spinner("⚖️ İnceleniyor..."):
                cevap = cevap_olustur(girdi)
                st.markdown(cevap)
                st.session_state.sohbet_gecmisi.append({"role": "assistant", "content": cevap})
