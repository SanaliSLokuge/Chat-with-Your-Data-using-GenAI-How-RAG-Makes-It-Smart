# 📚 Imports
import pandas as pd
import docx
import PyPDF2
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Config ---
st.set_page_config(page_title="AI-Powered File Q&A", layout="wide")

# --- Constants ---
API_KEY = st.secrets["openrouter"]["api_key"]
MODEL = "openai/gpt-4o-mini"

# --- Helper Functions ---
def chunk_text_by_words(text, chunk_size=100):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
        if len(words[i:i + chunk_size]) > 10
    ]

def extract_text_from_file(uploaded_file):
    name = uploaded_file.name
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        return df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()
    elif name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(uploaded_file)
        text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        return chunk_text_by_words(text)
    elif name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        text = "\n".join([p.text for p in doc.paragraphs])
        return chunk_text_by_words(text)
    else:
        st.error("❌ Unsupported file type.")
        return []

def retrieve_context(question, chunks, vectorizer, doc_vectors, top_k=3):
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, doc_vectors).flatten()
    indices = scores.argsort()[::-1][:top_k]
    return [chunks[i] for i in indices]

def ask_openrouter(context_chunks, question):
    context = "\n".join(context_chunks)
    prompt = f"""You are a helpful data analyst. Use ONLY the data below to answer the question.
Data:
{context}

Question: {question}
Answer:"""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    json_data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful data analyst."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.2,
        "top_p": 1
    }

    response = requests.post(url, headers=headers, json=json_data)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

# --- UI ---
st.title("📄 AI-Powered Document Q&A")
uploaded_file = st.file_uploader("Upload a CSV, PDF, or DOCX file", type=["csv", "pdf", "docx"])

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")
    with st.spinner("🔍 Extracting and indexing..."):
        text_chunks = extract_text_from_file(uploaded_file)
        if text_chunks:
            vectorizer = TfidfVectorizer()
            doc_vectors = vectorizer.fit_transform(text_chunks)
            st.success(f"✅ {len(text_chunks)} text chunks indexed.")

            question = st.text_input("Ask a question about your file:")
            if question:
                matched_chunks = retrieve_context(question, text_chunks, vectorizer, doc_vectors)
                with st.expander("🔎 Top matching chunks"):
                    for i, chunk in enumerate(matched_chunks, 1):
                        st.markdown(f"**Chunk {i}:** {chunk[:500]}...")

                with st.spinner("🤖 Generating answer..."):
                    try:
                        answer = ask_openrouter(matched_chunks, question)
                        st.success("✅ Answer:")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"API Error: {e}")
