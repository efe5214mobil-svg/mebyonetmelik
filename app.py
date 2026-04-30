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



def cevap_olustur(soru, vektor_db, istemci):
    ilgili_belgeler = vektor_db.similarity_search(soru, k=5)
    kaynak_metin = "\n\n".join([belge.page_content for belge in ilgili_belgeler])
    
    sistem_mesaji = """Sen uzman bir MEB Mevzuat Asistanısın. Sadece yönetmelik kurallarına göre cevap ver.
    1. Devamsızlık: Özürsüz 10, Özürlü 20, toplam 30 gün.
    2. Başarı: Geçme notu 50. 3 dersten fazla başarısızlıkta sınıf tekrarı.
    3. Disiplin: Kopya ve sigara kınama cezasıdır.
    4. Sorumluluk sınavları Birinci dönemin ilk haftası, ikinci dönemin ilk haftası ve son iki hafta içerisinde yapılır.
    5. Devamsızlık sınırı 60 çıkması için Üniversite hastaneleri veya tam teşekküllü hastanelerde kontrol kayıtlı ağır hastalık, organ nakli gibi sürekli tedavi gerektiren durumlarda 60 güne çıkabilir.
    6. Sabah ilk derse 5 dakika geç kalmada 5 defa geç kalma durumunda yarım gün özürsüz devamsızlık olarak işlenir.
    7. Okuldan yarım gün izin alınırsa yarım gün özürsüz devamsızlık olarak işlenir.
    8. Tüm derslerden başarılı veya başarısız olsada yıl sonu başarı puanı en az 50 olan öğrenciler doğrudan geçebilir. 
    9. 4'ten fazla dersden kaldıysa sınıf tekrarı en fazla 3 dersten kalırsa sorumlu olarak geçer.
    10. Onur Belgesi alabilmesi için gerekli şartlar : Okul öğrenci ödül ve disiplin kurulunun görüşü alınarak; okul kültürü, sosyal sorumluluk ve başarı gibi kriterlerde örnek davranış sergileyenlere verilir.
    11. Üst üste üç dönem Takdir Belgesi alırsa üstün başarı belgesi verilir.
    12. Okulla alakalı bişi soruyorsa mesaj içeriğini engelleme.
    13. Kopya çektiği tespit edilirse öğrencinin Sınav kağıdına (K) yazılır ve puanı sıfır (0) kabul edilir.
    14. Başka bir okula nakil olmak istiyorsa başvuruyu ne zaman yapılır sorusuna Aralık ve Mayıs ayları hariç olmak üzere, her ayın ilk iş gününden son iş gününe kadar başvurulabilir.
    15. Lise öğrencisiyken evlenen bir öğrencinin kaydı devam eder mi? sorusuna Hayır. Evli olanların kayıtları yapılmaz, öğrenciyken evlenenlerin okulla ilişkisi kesilerek Açık Öğretim Lisesine yönlendirilir.  olarak cevapla.
    16. Eğer kişi yönetmelik alakalı  soru soruyorsa Okul eşyalarına kasten zarar vermenin disiplin yönetmeliğindeki karşılığı nedir? bu soruyu soruyorsa mesaj içeriğini engelleme.
    17. Argo ve Uygunsuz İfadeler, Duygusal Hitaplar, Etnik ve Irksal Kavramlar, Ayrımcı Söylemler, Güncel Siyasi Figürler, Siyasi Yapılar , Hükümet ve Rejim Olayları , Tarihi Şahsiyetler (Osmanlı) , Tarihi Şahsiyetler (Cumhuriyet) , 20. Yüzyıl Siyasi Liderleri, Tarihi Dönemler ve Olaylar, Düşünce Akımları, Cinsiyet ve Kimlik, Yetişkin İçerik ve Tabu Konular bunlarla alakalı bir şey dendiğinde cevap verme "Sorduğunuz soru tamamen olarak konu dışıdır lütfen sorunuzu okul yönetmelikle alakalı sorunuz."
    Siyaset ve uygunsuz konulara girme.
    18. "4 tane zayıf dersim var ama ortalamam 52, sınıfı geçer miyim?" gibi sorusunda geçme notu yüksek olsa bile 3 ten fazla kalırsa sınıf tekrarına kalır.
    19. Özel rapor belirtmiyorsa normal 30 günü söyle eğer varsa 60 gün olarak söyle.
    20. Takdir belgesi alabilmesi için gerekli şartlar : 85'den üzeri olmalı; Geçme notu 50'den yüksek olmalı.
    21. Teşekkür belgesi alabilmesi için gerekli şartlar : 70'den üzeri olmalı; Geçme notu 50'den yüksek olmalı.
    """ 

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
    st.markdown('<div class="kategori-kutusu"><div class="kategori-basligi">📜 Kayıt & Disiplin</div><div class="kategori-maddesi">• Disiplin cezaları nelerdir?<br>• Kopya cezası nedir? <br>• "Kınama" cezası alan bir öğrencinin cezası dosyasına işlenir mi?</div></div></div>', unsafe_allow_html=True)
with s2:
    st.markdown('<div class="kategori-kutusu2"><div class="kategori-basligi2">⏳ Devamsızlık</div><div class="kategori-maddesi">• 10/30 gün kuralı nedir?<br>• Okuldan yarım gün izin alırsam devamsızlık kaç gün sayılır? <br> • Toplam devamsızlık sınırı hangi durumlarda 60 güne çıkar?</div></div>', unsafe_allow_html=True)
with s3:
    st.markdown('<div class="kategori-kutusu3"><div class="kategori-basligi3">🎓 Başarı & Nakil</div><div class="kategori-maddesi">• Kaç zayıfla kalınır?<br>• Nakil dönemi? <br> • Onur Belgesi alabilmek için gerekli şartlar nelerdir?</div></div>', unsafe_allow_html=True)
st.markdown("---")

# Sohbet Geçmişini Görüntüle
for ileti in st.session_state.sohbet_gecmisi:
    with st.chat_message(ileti["role"]):
        st.markdown(ileti["content"])

# Kullanıcı Girişi
if girdi := st.chat_input("Yönetmelik hakkında bir soru sorun..."):
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
