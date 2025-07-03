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

# --- Constants ---
API_KEY = st.secrets["api_key"]
MODEL = "openai/gpt-4o-mini"

# --- UI: Blog Intro ---
st.title("🧠 Build Your Own AI-Powered Document Analyst")
st.markdown("""
Welcome to the **AI-Powered Document Q&A** interactive blog!  
In this hands-on walkthrough, you’ll learn how to:

- Extract text from PDF, Word, and CSV files  
- Build a basic search index 
- Use **LLMs (like GPT-4o-mini)** to answer questions based on your files  

Follow along and upload your own document to try it live.
""")

# --- Step 1: Upload File ---
st.header("📂 Step 1: Upload Your Document")
uploaded_file = st.file_uploader("Choose a CSV, PDF, or DOCX file", type=["csv", "pdf", "docx"])

# --- Helper: Chunking & Parsing ---
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
            {"role": "system", "content": "You are a helpful data analyst.If it is a dataset you can analyse trends and other patterns/ insights from the dataset. If it a pdf, then understand the purpose and the relatable things in the document and check the standard or how to imporve it."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1
    }

    response = requests.post(url, headers=headers, json=json_data)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

# --- Proceed if File is Uploaded ---
if uploaded_file:
    st.success(f"✅ Uploaded: `{uploaded_file.name}`")
    with st.spinner("🔍 Extracting and indexing text..."):
        text_chunks = extract_text_from_file(uploaded_file)

    if text_chunks:
        st.header("🧠 Step 2: Indexing with TF-IDF")
        st.markdown("""
TF-IDF stands for *Term Frequency - Inverse Document Frequency*.  
It’s a simple algorithm that helps us score text relevance by comparing word frequencies across your document.

We’ll now create a vector representation of each chunk of text from your file.
""")
        vectorizer = TfidfVectorizer()
        doc_vectors = vectorizer.fit_transform(text_chunks)
        st.success(f"✅ {len(text_chunks)} chunks of text processed and indexed.")

        # --- Sample Questions ---
        st.header("💬 Step 3: Ask a Question About Your File")
        st.markdown("Try asking a question about the content you uploaded!")

        sample_qs = [
            "Can I become a machine learning engineer?",
            "What are my strongest skills?",
            "Summarize my technical experience.",
        ]
        col1, col2 = st.columns([2, 1])
        with col1:
            question = st.text_input("Type your question here:")
        with col2:
            selected = st.selectbox("Or pick a sample:", sample_qs)
            if st.button("Use sample question"):
                question = selected

        # --- LLM Settings ---
        with st.expander("⚙️ Advanced Settings"):
            temperature = st.slider("LLM Temperature (creativity)", 0.0, 1.0, 0.2)
            max_tokens = st.slider("Max Tokens", 100, 2000, 1000)

        if question:
            matched_chunks = retrieve_context(question, text_chunks, vectorizer, doc_vectors)
            with st.expander("🔍 Top Matching Chunks"):
                for i, chunk in enumerate(matched_chunks, 1):
                    st.markdown(f"**Chunk {i}:** {chunk[:400]}...")

            with st.spinner("🤖 Asking the AI..."):
                try:
                    answer = ask_openrouter(matched_chunks, question, temperature, max_tokens)
                    st.success("✅ Answer Generated:")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"API Error: {e}")
