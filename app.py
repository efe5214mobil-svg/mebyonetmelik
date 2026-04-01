import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from dotenv import load_dotenv
import os

# 🔐 .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

# 🎨 Sayfa
st.set_page_config(page_title="MEB Mevzuat Asistanı", page_icon="🏛️", layout="centered")

# 🎨 Stil
st.markdown("""
<style>
.stApp { font-family: 'Inter', sans-serif; }
[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# 🧠 VECTOR DB
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="okul_asistani_gpt_db", embedding_function=embeddings)

vector_db = load_vector_db()

# 🗂️ Hafıza
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# 🔍 AUTOCOMPLETE
oneriler = [
    "Devamsızlık sınırı nedir?",
    "Kaç gün devamsızlıkta kalınır?",
    "Sınıf geçme şartları nelerdir?",
    "Kaç dersten kalınca sınıfta kalınır?",
    "Takdir ve teşekkür şartları nelerdir?",
    "Sigara içmenin cezası nedir?",
    "Nakil başvurusu ne zaman yapılır?",
    "Ders süresi kaç dakikadır?"
]

def autocomplete(text):
    if not text:
        return []
    return [o for o in oneriler if text.lower() in o.lower()]

# ❌ UYGUNSUZ SORU KONTROL
def uygunsuz_mu(soru):
    yasakli = [
        "küfür","argo","salak","aptal","mal",
        "siyaset","cumhurbaşkanı","parti",
        "din","allah","tanrı",
        "ırk","kürt","türk","arap"
    ]
    soru_lower = soru.lower()
    return any(kelime in soru_lower for kelime in yasakli)

# 🤖 MODEL
def sorgula(soru):
    docs = vector_db.similarity_search(soru, k=4)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    messages = [{
        "role": "system",
        "content": """
Sen MEB (Milli Eğitim Bakanlığı) yönetmeliği uzmanısın.

GÖREV:
- Sadece yönetmeliğe göre cevap ver
- Kısa, net ve resmi ol
- Maddeler halinde yazabilirsin

YASAK:
- Küfür, argo, siyaset, din, ırk içeren cevap verme
- Uydurma bilgi verme

KURALLAR:

GENEL:
- Ders süresi: 40 dk (okul), 60 dk (işletme)

DEVAMSIZLIK:
- Özürsüz 10 gün → kalır
- Toplam 30 gün
- İstisna 60 gün

SINIF GEÇME:
- 3 derse kadar sorumlu
- 6 üstü → sınıf tekrarı

BAŞARI:
- Geçme notu: 50

DİSİPLİN:
- Sigara ve kopya = Kınama

BELGE:
- Teşekkür: 70-84.99
- Takdir: 85+

NAKİL:
- Aralık ve Mayıs hariç ay başı

STRATEJİ:
- Direkt cevabı ver
- Gereksiz uzatma
"""
    }]

    for msg in st.session_state.conversation[-4:]:
        messages.append(msg)

    messages.append({"role": "user", "content": f"{baglam}\n\nSoru: {soru}"})

    completion = client.chat.completions.create(
        messages=messages,
        model="openai/gpt-oss-120b",
        temperature=0.1,
        max_tokens=600
    )

    return completion.choices[0].message.content, docs

# 🎯 Başlık
st.title("🏛️ MEB Mevzuat Asistanı")
st.caption("Sadece yönetmeliğe uygun cevap verir")

# 💬 Chat geçmişi
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ✍️ Input
user_input = st.text_input("Sorunuzu yazın:")

# 🔍 Autocomplete
suggestions = autocomplete(user_input)
for s in suggestions:
    if st.button(f"🔎 {s}"):
        user_input = s

# 💬 Chat input
chat_input = st.chat_input("Sorunuzu buraya yazın...")

prompt = chat_input if chat_input else user_input

# 🚀 SORGULAMA
if prompt and prompt.strip() != "":

    # ❌ uygunsuz kontrol
    if uygunsuz_mu(prompt):
        hata_mesaji = "❌ Üzgünüm, sorduğunuz konu dışıdır. Lütfen MEB yönetmeliğine göre soru sorunuz."
        
        st.session_state.conversation.append({"role": "user", "content": prompt})
        st.session_state.conversation.append({"role": "assistant", "content": hata_mesaji})

        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.markdown(hata_mesaji)

    else:
        st.session_state.conversation.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Yanıt hazırlanıyor..."):
            cevap, kaynaklar = sorgula(prompt)

            st.session_state.conversation.append({"role": "assistant", "content": cevap})

            with st.chat_message("assistant"):
                st.markdown(cevap)

                if kaynaklar:
                    with st.expander("📄 Kaynaklar"):
                        for doc in kaynaklar:
                            st.caption(doc.page_content[:300] + "...")
