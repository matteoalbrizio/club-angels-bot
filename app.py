import json
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

st.set_page_config(page_title="Club Finder", layout="centered")

st.title("Club degli Investitori - Finder")
st.write("Trova le persone del Club più rilevanti in base alle loro competenze.")

# Carica il database
with open("club_people.json", "r", encoding="utf-8") as f:
    PEOPLE = json.load(f)

# OpenAI
api_key = st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key)

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

def build_embeddings():
    bios = [p.get("bio", "") for p in PEOPLE]
    vectors = []

    for i in range(0, len(bios), 64):
        batch = bios[i:i+64]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([d.embedding for d in resp.data])

    return np.array(vectors)

def top_k(query, vectors, k=5):
    q = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    sims = cosine_similarity([q], vectors)[0]
    idx = np.argsort(-sims)[:k]
    return idx

if not api_key:
    st.error("Manca la chiave OpenAI")
    st.stop()

with st.spinner("Preparazione database (solo la prima volta)..."):
    VECTORS = build_embeddings()

query = st.text_input("Cosa stai cercando? (es: FinTech, SaaS, Energy, Esperti di Formula Uno)")

if query:
    with st.spinner("Sto cercando i profili più rilevanti..."):
        idxs = top_k(query, VECTORS)

        candidates = []
        for i in idxs:
            p = PEOPLE[i]
            candidates.append({
                "name": p["name"],
                "url": p["url"],
                "bio": p["bio"][:1000]
            })

        context = "\n\n".join([
            f"Nome: {c['name']}\nLink: {c['url']}\nBio: {c['bio']}"
            for c in candidates
        ])

        system = """
Sei un assistente che suggerisce persone del Club degli Investitori.

Usa SOLO le biografie fornite.
Non inventare competenze.

Rispondi in italiano con:
- 3-5 persone consigliate
- breve motivazione
- link
"""

        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Richiesta: {query}\n\nBiografie:\n{context}"}
            ],
            temperature=0.2
        )

    st.write(resp.choices[0].message.content)