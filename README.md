# SmartDocQ – AI-Powered Document Analyst

**DocuSage** is a smart Streamlit-based app that lets you ask questions about any uploaded document (CSV, PDF, or DOCX) using vector search and an LLM (GPT-4o-mini via OpenRouter).

## 🚀 Features

- 📂 Upload and parse **PDF**, **Word**, and **CSV** documents
- 🔍 Automatically chunk content and build a **TF-IDF** search index
- 🧠 Ask questions and get **LLM-powered answers** from relevant content
- ⚙️ Customize response creativity and length with **advanced controls**

## 📸 Demo

![Demo](demo.gif) *(Add your own GIF or screenshot)*

## 🔧 Tech Stack

- `Streamlit` – UI & frontend
- `scikit-learn` – TF-IDF + similarity search
- `PyPDF2`, `python-docx`, `pandas` – File parsing
- `OpenRouter` – LLM API backend (GPT-4o-mini or any supported model)

## 📁 How to Run Locally

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-username/docsage.git
   cd docsage
