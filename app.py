import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from dotenv import load_dotenv
import os
import re
import time 

# 🔐 API ve Çevre Değişkenleri
load_dotenv()
api_anahtari = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
istemci = Groq(api_key=api_anahtari)

# 🎨 Sayfa Yapılandırması
st.set_page_config(page_title="MEB Yönetmelik Asistanı", page_icon="🏫", layout="centered")

# 🖌️ Modern Görünüm (CSS)
st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }
    .ana-baslik { font-size: 2.5rem; font-weight: 800; text-align: center; margin-bottom: 1.5rem; color: #FFFFFF; }
    .kategori-kutusu {
        background-color: rgba(128,128,128, 0.05);
        border-radius: 15px;
        padding: 18px;
        border-top: 4px solid #FF4B4B;
        height: 100%;
        margin-bottom: 10px;
    }
    .kategori-kutusu2 {
        background-color: rgba(128,128,128, 0.05);
        border-radius: 15px;
        padding: 18px;
        border-top: 4px solid #0974e6;
        height: 100%;
        margin-bottom: 10px;
    }
    .kategori-kutusu3 {
        background-color: rgba(128,128,128, 0.05);
        border-radius: 15px;
        padding: 18px;
        border-top: 4px solid #05f250;
        height: 100%;
        margin-bottom: 10px;
    }
    .kategori-basligi { font-weight: bold; color: #FF4B4B; margin-bottom: 10px; font-size: 1.15rem; }
    .kategori-basligi2 { font-weight: bold; color: #0974e6; margin-bottom: 10px; font-size: 1.15rem; }
    .kategori-basligi3 { font-weight: bold; color: #05f250; margin-bottom: 10px; font-size: 1.15rem; }
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

if "sohbet_gecmisi" not in st.session_state:
    st.session_state.sohbet_gecmisi = []

# 🤖 Yanıt Oluşturucu
def cevap_olustur(soru):
    ilgili_belgeler = vektor_tabani.similarity_search(soru, k=5)
    kaynak_metin = "\n\n".join([belge.page_content for belge in ilgili_belgeler])
    
    iletiler = [{
        "role": "system", 
        "content": """Sen uzman bir MEB Mevzuat Asistanısın. Sadece aşağıdaki yönetmelik kurallarına göre cevap ver...""" # Kısaltılmıştır
    }]
    
    for ileti in st.session_state.sohbet_gecmisi[-2:]:
        iletiler.append(ileti)
    
    iletiler.append({"role": "user", "content": f"BAĞLAM:\n{kaynak_metin}\n\nSORU: {soru}"})
    
    yanit = istemci.chat.completions.create(
        messages=iletiler, 
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )
    return yanit.choices[0].message.content

# --- ARAYÜZ ---
st.markdown("<div class='ana-baslik'>🏛️ MEB Yönetmelik Asistanı</div>", unsafe_allow_html=True)

st.markdown("### 💡 Hızlı Sorular")
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown('<div class="kategori-kutusu"><div class="kategori-basligi">📜 Kayıt & Disiplin</div><div class="kategori-maddesi">• Evlilik durumu?<br>• Kopya cezası?</div></div>', unsafe_allow_html=True)
with s2:
    st.markdown('<div class="kategori-kutusu2"><div class="kategori-basligi2">⏳ Devamsızlık</div><div class="kategori-maddesi">• 10/30 gün kuralı?<br>• Geç gelme sınırı?</div></div>', unsafe_allow_html=True)
with s3:
    st.markdown('<div class="kategori-kutusu3"><div class="kategori-basligi3">🎓 Başarı & Nakil</div><div class="kategori-maddesi">• Kaç zayıfla kalınır?<br>• Nakil dönemi?</div></div>', unsafe_allow_html=True)
st.markdown("---")

for ileti in st.session_state.sohbet_gecmisi:
    with st.chat_message(ileti["role"]):
        st.markdown(ileti["content"])

if girdi := st.chat_input("Yönetmelik hakkında bir soru sorun..."):
    if suzgec_kontrolu(girdi):
        uyari_alani = st.empty()
        uyari_alani.error("⚠️ Uyarı: İletiniz yönetmelik dışı veya uygunsuz içerik barındırdığı için engellenmiştir.")
        time.sleep(2) 
        uyari_alani.empty() 
    else:
        st.session_state.sohbet_gecmisi.append({"role": "user", "content": girdi})
        with st.chat_message("user"):
            st.markdown(girdi)

        with st.chat_message("assistant"):
            with st.spinner("⚖️ İnceleniyor..."):
                cevap = cevap_olustur(girdi)
                st.markdown(cevap)
                st.session_state.sohbet_gecmisi.append({"role": "assistant", "content": cevap})
                st.rerun()

# --- 🛡️ GİRİŞ KONTROLÜ ---
kayitli_anahtar = anahtari_oku()

# Eğer dosya yoksa veya dosyadaki anahtar geçersizse giriş ekranını göster
if not api_anahtari_gecerli_mi(kayitli_anahtar):
    st.markdown("<h1 style='text-align: center;'>🏫 MEB Yönetmelik Asistanı</h1>", unsafe_allow_html=True)
    st.info("Sistemi ilk kez çalıştırıyorsunuz veya anahtarınız geçersiz. Lütfen geçerli bir Groq API Key giriniz.")
    
    input_key = st.text_input("Groq API Key:", type="password")
    if st.button("Sistemi Kur ve Başlat"):
        if api_anahtari_gecerli_mi(input_key):
            anahtari_kaydet(input_key) # Anahtarı dosyaya yaz (Kalcı hale getir)
            st.success("Anahtar doğrulandı ve kaydedildi! Başlatılıyor...")
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Hatalı API Anahtarı! Lütfen kontrol edin.")
    st.stop()

# --- 🚀 UYGULAMA ANA GÖVDESİ ---
# Buraya gelindiyse anahtar dosyada vardır ve doğrudur
istemci = Groq(api_key=anahtari_oku())

# 🖌️ Modern Görünüm (CSS)
st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }
    .ana-baslik { font-size: 2.5rem; font-weight: 800; text-align: center; margin-bottom: 1.5rem; color: #FFFFFF; }
    .kategori-kutusu, .kategori-kutusu2, .kategori-kutusu3 {
        background-color: rgba(128,128,128, 0.05);
        border-radius: 15px; padding: 18px; margin-bottom: 10px; height: 100%;
    }
    .kategori-kutusu { border-top: 4px solid #FF4B4B; }
    .kategori-kutusu2 { border-top: 4px solid #0974e6; }
    .kategori-kutusu3 { border-top: 4px solid #05f250; }
</style>
""", unsafe_allow_html=True)

# 🧠 Veri Tabanı Yükleme
@st.cache_resource
def veri_tabanini_yukle():
    gomme_modeli = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="okul_asistani_gpt_db", embedding_function=gomme_modeli)

vektor_tabani = veri_tabanini_yukle()

# --- ARAYÜZ ---
st.markdown("<div class='ana-baslik'>🏛️ MEB Yönetmelik Asistanı</div>", unsafe_allow_html=True)

if "sohbet_gecmisi" not in st.session_state:
    st.session_state.sohbet_gecmisi = []

# Hızlı Bilgi Kutuları
c1, c2, c3 = st.columns(3)
with c1: st.markdown('<div class="kategori-kutusu"><b>📜 Mevzuat</b><br>Kayıt ve Kabul</div>', unsafe_allow_html=True)
with c2: st.markdown('<div class="kategori-kutusu2"><b>⏳ Devamsızlık</b><br>10/30 Gün Sınırı</div>', unsafe_allow_html=True)
with c3: st.markdown('<div class="kategori-kutusu3"><b>🎓 Sınıf Geçme</b><br>Zayıf Sayısı</div>', unsafe_allow_html=True)

# Sohbet Alanı
for ileti in st.session_state.sohbet_gecmisi:
    with st.chat_message(ileti["role"]):
        st.markdown(ileti["content"])

if girdi := st.chat_input("Yönetmelik sorunuzu yazın..."):
    st.session_state.sohbet_gecmisi.append({"role": "user", "content": girdi})
    with st.chat_message("user"):
        st.markdown(girdi)

    with st.chat_message("assistant"):
        with st.spinner("⚖️ İnceleniyor..."):
            # RAG İşlemi (Similarity Search)
            ilgili_belgeler = vektor_tabani.similarity_search(girdi, k=4)
            kaynak = "\n".join([b.page_content for b in ilgili_belgeler])
            
            yanit = istemci.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Sen bir MEB Mevzuat asistanısın. Sadece yönetmeliklere uygun cevap ver."},
                    {"role": "user", "content": f"Bağlam: {kaynak}\nSoru: {girdi}"}
                ],
                model="llama-3.3-70b-versatile"
            )
            cevap = yanit.choices[0].message.content
            st.markdown(cevap)
            st.session_state.sohbet_gecmisi.append({"role": "assistant", "content": cevap})
            st.rerun()

# Sidebar - Ayarları Sıfırlama
with st.sidebar:
    st.write("### 🛠️ Ayarlar")
    if st.button("Sistemi Sıfırla (Anahtarı Sil)"):
        if os.path.exists(KEY_FILE):
            os.remove(KEY_FILE)
        st.session_state.clear()
        st.rerun()        padding: 18px;
        border-top: 4px solid #FF4B4B;
        height: 100%;
        margin-bottom: 10px;
    }

        .kategori-kutusu2 {
        background-color: rgba(128,128,128, 0.05);
        border-radius: 15px;
        padding: 18px;
        border-top: 4px solid #0974e6;
        height: 100%;
        margin-bottom: 10px;
    }

        .kategori-kutusu3 {
        background-color: rgba(128,128,128, 0.05);
        border-radius: 15px;
        padding: 18px;
        border-top: 4px solid #05f250;
        height: 100%;
        margin-bottom: 10px;
    }
    .kategori-basligi { font-weight: bold; color: #FF4B4B; margin-bottom: 10px; font-size: 1.15rem; }
    .kategori-basligi2 { font-weight: bold; color: #0974e6; margin-bottom: 10px; font-size: 1.15rem; }
    .kategori-basligi3 { font-weight: bold; color: #05f250; margin-bottom: 10px; font-size: 1.15rem; }
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
    karakter_haritasi = {'1': 'i', '0': 'o', '3': 'e', '4': 'a', '5': 's', '7': 't', '8': 'b', '@': 'a', '$': 's'}
    temiz_metin = metin.lower()
    for eski, yeni in karakter_haritasi.items():
        temiz_metin = temiz_metin.replace(eski, yeni)
    
    # Hileleri engellemek için metni sıkıştır
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

if "sohbet_gecmisi" not in st.session_state:
    st.session_state.sohbet_gecmisi = []

# 🤖 Yanıt Oluşturucu (TÜM YÖNETMELİK BURADA)
def cevap_olustur(soru):
    ilgili_belgeler = vektor_tabani.similarity_search(soru, k=5)
    kaynak_metin = "\n\n".join([belge.page_content for belge in ilgili_belgeler])
    
    iletiler = [{
        "role": "system", 
        "content": """Sen uzman bir MEB Mevzuat Asistanısın. Sadece aşağıdaki yönetmelik kurallarına göre cevap ver:

        1. GENEL BİLGİLER:
        - Ders saati okulda 40 dk, işletmelerde 60 dk'dır.
        - Ders yılı: Başlangıçtan kesildiği tarihe kadardır.
        - Kayıt: 18 yaşını bitirmemiş olmak şarttır.
        - Evlilik: Evli olanların kaydı yapılmaz, öğrenciyken evlenenler Açık Lise'ye aktarılır.
        - Hazırlık: Üst üste 2 yıl başarısız olan 9. sınıfa (hazırlık olmayan) nakledilir.

        2. DEVAMSIZLIK:
        - Özürsüz sınır 10 gündür. 10 günü geçen başarısız sayılır.
        - Toplam sınır (özürlü+özürsüz) 30 gündür.
        - İstisna: Organ nakli, ağır hastalık gibi hallerde toplam sınır 60 gündür.
        - Geç gelme: Sadece 1. ders saati için geçerlidir, sonrası devamsızlıktır.

        3. SINIF GEÇME & BAŞARI:
        - Yazılı sınav: Her dersten en az 2 yazılı yapılır.
        - Başarı puanı: Geçme notu en az 50'dir.
        - Sorumlu geçme: En fazla 3 dersten zayıfı olan sorumlu geçer.
        - Sınıf tekrarı: Başarısız ders sayısı 6'yı geçerse sınıf tekrar edilir.
        - Beceri sınavı: %80 sınav puanı + %20 iş dosyası.

        4. NAKİL & DERS SEÇİMİ:
        - Nakil: Aralık ve Mayıs hariç her ayın ilk iş günü başvurulur.
        - Ders seçimi (9. Sınıf): Ders yılının ilk haftasında yapılır.

        5. DİSİPLİN & ÖDÜL:
        - Cezalar: Kınama, kısa süreli uzaklaştırma, okul değiştirme, örgün eğitim dışına çıkarma.
        - Kopya ve Sigara: 'Kınama' cezası gerektirir.
        - Teşekkür: 70,00 - 84,99 arası ortalama.
        - Takdir: 85,00 ve üzeri ortalama.

        KESİN YASAKLAR: Siyaset, tarih, felsefi akımlar ve uygunsuz hitaplara cevap verme. Daima profesyonel Türkçe kullan."""
    }]
    
    for ileti in st.session_state.sohbet_gecmisi[-2:]:
        iletiler.append(ileti)
    
    iletiler.append({"role": "user", "content": f"BAĞLAM:\n{kaynak_metin}\n\nSORU: {soru}"})
    
    yanit = istemci.chat.completions.create(
        messages=iletiler, 
        model="llama-3.3-70b-versatile", # llama-3.3-70b-versatile #moonshotai/kimi-k2-instruct-0905
        temperature=0.1
    )
    return yanit.choices[0].message.content

# --- ARAYÜZ ---
st.markdown("<div class='ana-baslik'>🏛️ MEB Yönetmelik Asistanı</div>", unsafe_allow_html=True)


st.markdown("### 💡 Hızlı Sorular")
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown('<div class="kategori-kutusu"><div class="kategori-basligi">📜 Kayıt & Disiplin</div><div class="kategori-maddesi">• Evlilik durumu?<br>• Kopya cezası?</div></div>', unsafe_allow_html=True)
with s2:
    st.markdown('<div class="kategori-kutusu2"><div class="kategori-basligi2">⏳ Devamsızlık</div><div class="kategori-maddesi">• 10/30 gün kuralı?<br>• Geç gelme sınırı?</div></div>', unsafe_allow_html=True)
with s3:
    st.markdown('<div class="kategori-kutusu3"><div class="kategori-basligi3">🎓 Başarı & Nakil</div><div class="kategori-maddesi">• Kaç zayıfla kalınır?<br>• Nakil dönemi?</div></div>', unsafe_allow_html=True)
st.markdown("---")

for ileti in st.session_state.sohbet_gecmisi:
    with st.chat_message(ileti["role"]):
        st.markdown(ileti["content"])

if girdi := st.chat_input("Yönetmelik hakkında bir soru sorun..."):
    
    if suzgec_kontrolu(girdi):
        uyari_alani = st.empty()
        uyari_alani.error("⚠️ Uyarı: İletiniz yönetmelik dışı veya uygunsuz içerik barındırdığı için engellenmiştir.")
        time.sleep(2) 
        uyari_alani.empty() 
    else:
        st.session_state.sohbet_gecmisi.append({"role": "user", "content": girdi})
        with st.chat_message("user"):
            st.markdown(girdi)

        with st.chat_message("assistant"):
            with st.spinner("⚖️ İnceleniyor..."):
                cevap = cevap_olustur(girdi)
                st.markdown(cevap)
                st.session_state.sohbet_gecmisi.append({"role": "assistant", "content": cevap})
                st.rerun()
