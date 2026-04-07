import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from dotenv import load_dotenv
import os
import re
import time 

# --- 1. AYARLAR VE YAPILANDIRMA ---
load_dotenv()
st.set_page_config(page_title="MEB Yönetmelik Asistanı", page_icon="🏫", layout="centered")

# CSS Tasarımı
st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }
    .ana-baslik { font-size: 2.5rem; font-weight: 800; text-align: center; margin-bottom: 1.5rem; color: #FFFFFF; }
    .kategori-kutusu { background-color: rgba(128,128,128, 0.05); border-radius: 15px; padding: 18px; border-top: 4px solid #FF4B4B; height: 100%; margin-bottom: 10px; }
    .kategori-kutusu2 { background-color: rgba(128,128,128, 0.05); border-radius: 15px; padding: 18px; border-top: 4px solid #0974e6; height: 100%; margin-bottom: 10px; }
    .kategori-kutusu3 { background-color: rgba(128,128,128, 0.05); border-radius: 15px; padding: 18px; border-top: 4px solid #05f250; height: 100%; margin-bottom: 10px; }
    .kategori-basligi { font-weight: bold; color: #FF4B4B; margin-bottom: 10px; font-size: 1.15rem; }
    .kategori-basligi2 { font-weight: bold; color: #0974e6; margin-bottom: 10px; font-size: 1.15rem; }
    .kategori-basligi3 { font-weight: bold; color: #05f250; margin-bottom: 10px; font-size: 1.15rem; }
    .kategori-maddesi { font-size: 0.88rem; margin-bottom: 6px; color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)

# --- 2. FONKSİYONLAR ---

@st.cache_resource
def veri_tabanini_yukle():
    # Klasör yoksa hata vermemesi için kontrol eklenebilir
    gomme_modeli = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="okul_asistani_gpt_db", embedding_function=gomme_modeli)

def suzgec_kontrolu(metin):
    karakter_haritasi = {'1': 'i', '0': 'o', '3': 'e', '4': 'a', '5': 's', '7': 't', '8': 'b', '@': 'a', '$': 's'}
    temiz_metin = metin.lower()
    for eski, yeni in karakter_haritasi.items():
        temiz_metin = temiz_metin.replace(eski, yeni)
    
    sikistirilmis_metin = re.sub(r'[^a-z0-9çşğüöı]', '', temiz_metin)
    yasakli_kelimeler = [
        "oc", "aq", "amk", "amq", "pic", "got", "sik", "amc", "yarrak", "orospu", "bebegim", "askim",
        "nigga", "zenci", "cikolata", "irkci", "yahudi", "ermeni", "nazi",
        "erdogan", "tayyip", "rte", "cumhurbaskani", "akp", "chp", "mhp", "siyaset", "parti", "darbe",
        "mahmud", "charles", "suleyman", "fatih", "kanuni", "padisah", "kral", "imparator", "osmanli",
        "ataturk", "hitler", "stalin", "lenin", "modernizm", "narsizm", "narsist", "nihilizm", 
        "ideoloji", "1945", "1939", "savas", "gay", "lezbiyen", "lgbt", "seks", "porno"
    ]
    return any(yasakli in sikistirilmis_metin for yasakli in yasakli_kelimeler)

def cevap_olustur(soru, vektor_db, istemci):
    ilgili_belgeler = vektor_db.similarity_search(soru, k=5)
    kaynak_metin = "\n\n".join([belge.page_content for belge in ilgili_belgeler])
    
    sistem_mesaji = """Sen uzman bir MEB Mevzuat Asistanısın. Sadece yönetmelik kurallarına göre cevap ver.
    1. Devamsızlık: Özürsüz 10, toplam 30 gün.
    2. Başarı: Geçme notu 50. 6 dersten fazla başarısızlıkta sınıf tekrarı.
    3. Disiplin: Kopya ve sigara kınama cezasıdır.
    4. Sorumluluk sınavları Birinci dönemin ilk haftası, ikinci dönemin ilk haftası ve son iki hafta içerisinde yapılır.
    5. İlgili soruyla alakalı araştır ve ona göre madde ver Örnek : "(Madde-X)" sonuna ekliceksin.
    Siyaset ve uygunsuz konulara girme."""

    iletiler = [{"role": "system", "content": sistem_mesaji}]
    for ileti in st.session_state.sohbet_gecmisi[-2:]:
        iletiler.append(ileti)
    iletiler.append({"role": "user", "content": f"BAĞLAM:\n{kaynak_metin}\n\nSORU: {soru}"})
    
    yanit = istemci.chat.completions.create(
        messages=iletiler, 
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )
    return yanit.choices[0].message.content

# --- 3. ANA DÖNGÜ VE GİRİŞ KONTROLÜ ---

if "sohbet_gecmisi" not in st.session_state:
    st.session_state.sohbet_gecmisi = []

# API Anahtarı Kontrolü
api_key = os.getenv("GROQ_API_KEY") or st.session_state.get("custom_api_key")

if not api_key:
    st.title("🏫 MEB Yönetmelik Asistanı")
    st.info("Lütfen devam etmek için Groq API anahtarınızı girin.")
    input_key = st.text_input("Groq API Key:", type="password")
    if st.button("Sistemi Başlat"):
        if input_key.startswith("gsk_"): # Basit bir kontrol
            st.session_state.custom_api_key = input_key
            st.success("Anahtar kabul edildi!")
            st.rerun()
        else:
            st.error("Geçersiz API anahtarı formatı.")
    st.stop()

# API İstemcisini ve DB'yi Başlat
istemci = Groq(api_key=api_key)
vektor_tabani = veri_tabanini_yukle()

# --- 4. ARAYÜZ (UI) ---
st.markdown("<div class='ana-baslik'>🏛️ MEB Yönetmelik Asistanı</div>", unsafe_allow_html=True)

# Hızlı Sorular Bölümü
st.markdown("### 💡 Hızlı Sorular")
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown('<div class="kategori-kutusu"><div class="kategori-basligi">📜 Kayıt & Disiplin</div><div class="kategori-maddesi">• Disiplin cezaları nelerdir?<br>• Kopya cezası?</div></div>', unsafe_allow_html=True)
with s2:
    st.markdown('<div class="kategori-kutusu2"><div class="kategori-basligi2">⏳ Devamsızlık</div><div class="kategori-maddesi">• 10/30 gün kuralı?<br>• Geç gelme sınırı?</div></div>', unsafe_allow_html=True)
with s3:
    st.markdown('<div class="kategori-kutusu3"><div class="kategori-basligi3">🎓 Başarı & Nakil</div><div class="kategori-maddesi">• Kaç zayıfla kalınır?<br>• Nakil dönemi?</div></div>', unsafe_allow_html=True)
st.markdown("---")

# Sohbet Geçmişini Görüntüle
for ileti in st.session_state.sohbet_gecmisi:
    with st.chat_message(ileti["role"]):
        st.markdown(ileti["content"])

# Kullanıcı Girişi
if girdi := st.chat_input("Yönetmelik hakkında bir soru sorun..."):
    if suzgec_kontrolu(girdi):
        st.error("⚠️ Uyarı: İletiniz uygunsuz içerik barındırdığı için engellenmiştir.")
    else:
        st.session_state.sohbet_gecmisi.append({"role": "user", "content": girdi})
        with st.chat_message("user"):
            st.markdown(girdi)

        with st.chat_message("assistant"):
            with st.spinner("⚖️ İnceleniyor..."):
                cevap = cevap_olustur(girdi, vektor_tabani, istemci)
                st.markdown(cevap)
                st.session_state.sohbet_gecmisi.append({"role": "assistant", "content": cevap})
                # Streamlit'in yanıtı hemen göstermesi için rerun bazen gerekebilir 
                # ama chat_message içinde genellikle gerekmez. İsteğe bağlı: st.rerun()
