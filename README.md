# 📄 PDF Chat Assistant

Chat with any PDF using Generative AI. Upload a document, ask questions, and get answers grounded only in that document, with the source passages shown.

## Features
- Upload any text-based PDF
- Ask questions in a chat interface
- Answers come only from the document (no made-up facts)
- Shows which parts of the document were used

## How it works (RAG)
1. **Extract** text from the PDF with `pypdf`
2. **Chunk** the text into overlapping pieces
3. **Retrieve** the most relevant chunks with TF-IDF + cosine similarity
4. **Generate** an answer with Google Gemini, using only the retrieved chunks

## Tech stack
Python · Streamlit · Google Gemini API · scikit-learn · pypdf

## Run it locally
```bash
git clone https://github.com/sarashaheryar897/pdf-chat-assistant.git
cd pdf-chat-assistant
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
Get a free API key at https://aistudio.google.com, copy `.env.example` to `.env`, and paste your key. Then:
```bash
streamlit run app.py
```

## Future improvements
- Semantic search with embeddings and a vector database
- Support for multiple PDFs and scanned documents (OCR)
