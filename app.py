import streamlit as st
import pandas as pd
import faiss
import numpy as np
import openai
import requests

# --- API Setup ---
openai.api_key = "sk-or-v1-e320933bc3a373f7eeeeea668e6aa58918292170625f7d372453b6a1a0330176Y"
openai.api_base = "https://openrouter.ai/api/v1"

MODEL = "google/gemma-7b-it"  # You can change to any supported OpenRouter model

# --- Embedding with OpenRouter (Optional) ---
# We'll simulate embedding via TF-IDF instead, since OpenRouter doesn't expose embeddings for all models

# --- Vector Store Using TF-IDF ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFStore:
    def __init__(self, texts):
        self.texts = texts
        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform(texts)

    def query(self, question, top_k=3):
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.vectors).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [self.texts[i] for i in top_indices]

# --- Streamlit App ---
st.set_page_config(page_title="Chat with Your Data (OpenRouter)", layout="wide")
st.title("🔍 Chat with Your Data using OpenRouter + RAG (Simulated)")

uploaded_file = st.file_uploader("📤 Upload a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Preview:", df.head())

    # Chunk the data
    text_chunks = df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()
    store = TFIDFStore(text_chunks)
    st.success("✅ Data loaded and indexed.")

    question = st.text_input("💬 Ask a question about your data:")
    if question:
        matches = store.query(question)
        context = "\n".join(matches)

        # Prompt Template
        prompt = f"""You are a data assistant. Use only the data below to answer the user's question.

        Data:
        {context}

        Question: {question}
        Answer:"""

        # OpenRouter call
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response["choices"][0]["message"]["content"]
            st.markdown("### 🤖 Answer:")
            st.write(answer)

            with st.expander("🔍 Retrieved Context Chunks"):
                for i, chunk in enumerate(matches):
                    st.markdown(f"**Chunk {i+1}:** {chunk}")
        except Exception as e:
            st.error(f"OpenRouter Error: {e}")
