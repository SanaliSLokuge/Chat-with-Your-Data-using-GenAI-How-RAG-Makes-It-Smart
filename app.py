# 📚 Imports
import pandas as pd
import docx
import PyPDF2
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Config ---
st.set_page_config(page_title="🧠 AI-Powered Document Q&A Blog", layout="wide")

# --- Style ---
st.markdown("""
    <style>
        body { background-color: #f5f7fa; }
        .title { color: #0a2f5c; font-size: 40px; font-weight: bold; }
        .step-header { background-color: #eaf2ff; padding: 10px; border-radius: 8px; margin-top: 20px; }
        .highlight { color: #1a73e8; font-weight: 600; }
        .important { background-color: #fef3c7; padding: 10px; border-left: 5px solid #facc15; }
        .chunk-box { background-color: #f0f4f8; padding: 10px; border-radius: 6px; margin-bottom: 10px; }
        .question-box { background-color: #e0f7fa; padding: 10px; border-radius: 6px; }
        .answer-box { background-color: #e7f5e1; padding: 15px; border-radius: 10px; border: 1px solid #a3d977; }
    </style>
""", unsafe_allow_html=True)

# --- Constants ---
API_KEY = st.secrets["api_key"]
MODEL = "openai/gpt-4o-mini"

# --- UI: Blog Intro ---
st.markdown('<div class="title">🧠 Build Your Own AI-Powered Document Analyst</div>', unsafe_allow_html=True)
st.markdown("""
Welcome to the **AI-Powered Document Q&A** interactive blog!  
In this hands-on walkthrough, you’ll learn how to:

- 🧾 Extract text from **PDF**, **Word**, and **CSV** files  
- 🔎 Build a search index with **TF-IDF**  
- 🤖 Use **LLMs (like GPT-4o-mini)** to answer questions about your files  

Upload a document and start exploring!
""")

# --- Step 1: Upload File ---
st.markdown('<div class="step-header"><h3>📂 Step 1: Upload Your Document</h3></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a CSV, PDF, or DOCX file", type=["csv", "pdf", "docx"])

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
        st.markdown("### 📊 Preview of Uploaded CSV")
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

def ask_openrouter(context_chunks, question, temperature, max_tokens):
    context = "\n".join(context_chunks)
    prompt = f"""You are a helpful data analyst. Use ONLY the data below to answer the question. 
Please go deep into why, how, and when — and provide point-form answers.

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
            {"role": "system", "content": "You are a helpful data analyst. If it's a dataset, analyze trends and insights. If it's a document, assess purpose, gaps, and improvements."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1
    }

    response = requests.post(url, headers=headers, json=json_data)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

# --- Main Execution ---
if uploaded_file:
    st.success(f"✅ Uploaded: `{uploaded_file.name}`")
    with st.spinner("🔍 Extracting and indexing text..."):
        text_chunks = extract_text_from_file(uploaded_file)

    if text_chunks:
        st.markdown('<div class="step-header"><h3>🧠 Step 2: Indexing with TF-IDF</h3></div>', unsafe_allow_html=True)
        st.markdown("""
TF-IDF stands for **Term Frequency - Inverse Document Frequency**.  
It helps identify important terms by comparing word frequency across text chunks.

Creating a vector representation of your document...
""")
        vectorizer = TfidfVectorizer()
        doc_vectors = vectorizer.fit_transform(text_chunks)
        st.success(f"✅ {len(text_chunks)} text chunks processed.")

        # Step 3: Ask Questions
        st.markdown('<div class="step-header"><h3>💬 Step 3: Ask a Question About Your File</h3></div>', unsafe_allow_html=True)
        st.markdown("Try asking a question about the content in your uploaded file.")

        sample_qs = [
            "Can I become a machine learning engineer?",
            "What are my strongest skills?",
            "Summarize my technical experience.",
        ]
        col1, col2 = st.columns([2, 1])
        with col1:
            question = st.text_input("📝 Type your question here:")
        with col2:
            selected = st.selectbox("🎯 Sample questions", sample_qs)
            if st.button("Use sample"):
                question = selected

        with st.expander("⚙️ Advanced LLM Settings"):
            temperature = st.slider("🎨 Temperature (creativity)", 0.0, 1.0, 0.2)
            max_tokens = st.slider("🧾 Max tokens", 100, 2000, 1000)

        if question:
            matched_chunks = retrieve_context(question, text_chunks, vectorizer, doc_vectors)
            with st.expander("🔍 Top Matching Chunks"):
                for i, chunk in enumerate(matched_chunks, 1):
                    st.markdown(f'<div class="chunk-box"><b>Chunk {i}:</b> {chunk[:400]}...</div>', unsafe_allow_html=True)

            with st.spinner("🤖 Querying the AI..."):
                try:
                    answer = ask_openrouter(matched_chunks, question, temperature, max_tokens)
                    st.markdown('<div class="answer-box"><b>✅ Answer:</b><br>' + answer + '</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"API Error: {e}")
