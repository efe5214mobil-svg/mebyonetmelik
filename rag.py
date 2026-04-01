from groq import Groq
import os

# API anahtarının yüklü olduğundan emin olun
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def okul_asistani_sorgula(soru, vector_db):
    # 1. Benzerlik araması
    docs = vector_db.similarity_search(soru, k=3)
    
    # 2. Bağlamı birleştirme
    baglam = "\n\n".join([doc.page_content for doc in docs])
    
    # 3. Groq API çağrısı
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system", 
                "content": "Sen bir okul asistanısın. Cevabını sadece verilen bağlama göre ver. Bağlamda bilgi yoksa 'Bilmiyorum' de."
            },
            {
                "role": "user", 
                "content": f"Bağlam:\n{baglam}\n\nSoru: {soru}"
            }
        ],
        model="openai/gpt-oss-120b", # Model ismi güncellendi
        temperature=0,
    )
    
    cevap = chat_completion.choices[0].message.content
    kaynaklar = [doc.page_content[:200] for doc in docs]

    return cevap, kaynaklar # Girintileme düzeltildi
