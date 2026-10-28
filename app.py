"""
PDF Chat Assistant
Upload a PDF and ask questions about it. Answers come only from the document.

How it works (RAG = Retrieval-Augmented Generation):
  1. Extract text from the PDF.
  2. Split the text into small overlapping chunks.
  3. Find the chunks most relevant to the question (TF-IDF similarity).
  4. Send only those chunks + the question to the Gemini model.
"""

import os
import time

import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()  # reads GEMINI_API_KEY from a local .env file

MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]  # main model, then a lighter backup
RETRIES = 3                     # attempts per model when the server is busy
CHUNK_SIZE = 800                # characters per chunk
CHUNK_OVERLAP = 150             # shared characters between neighbouring chunks
TOP_K = 4                       # how many chunks to send to the model


# ---------- Step 1: read the PDF ----------
def extract_text(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


# ---------- Step 2: split into chunks ----------
def split_into_chunks(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ---------- Step 3: find the most relevant chunks ----------
def find_relevant_chunks(question: str, chunks: list[str], vectorizer, matrix) -> list[str]:
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, matrix)[0]
    best = scores.argsort()[::-1][:TOP_K]
    return [chunks[i] for i in best if scores[i] > 0]


# ---------- Step 4: ask Gemini ----------
def ask_gemini(client, question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant that answers questions about a document.
Use ONLY the context below. If the answer is not in the context, say
"I couldn't find that in the document." Keep answers clear and concise.

CONTEXT:
{context}

QUESTION: {question}"""
    last_error = None
    for model in MODELS:
        for attempt in range(RETRIES):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text
            except Exception as e:
                last_error = e
                busy = any(code in str(e) for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
                if not busy:
                    raise                      # a real error (e.g. wrong API key): stop now
                time.sleep(2 ** attempt)       # wait 1s, 2s, 4s before trying again
    raise last_error


# ---------- The web app ----------
st.set_page_config(page_title="PDF Chat Assistant", page_icon="📄")
st.title("📄 PDF Chat Assistant")
st.caption("Upload a PDF and ask questions. Answers come only from your document.")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY is missing. Add it to a .env file (see README).")
    st.stop()
client = genai.Client(api_key=api_key)

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    # Process each new file once and keep it in the session
    if st.session_state.get("file_name") != uploaded.name:
        with st.spinner("Reading your PDF..."):
            text = extract_text(uploaded)
            chunks = split_into_chunks(text)
            if not chunks:
                st.error("No readable text found. Scanned PDFs need OCR first.")
                st.stop()
            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform(chunks)
        st.session_state.update(
            file_name=uploaded.name, chunks=chunks,
            vectorizer=vectorizer, matrix=matrix, messages=[],
        )
        st.success(f"Ready! Split into {len(chunks)} chunks.")

    # Show the chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # New question
    if question := st.chat_input("Ask something about the document"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                s = st.session_state
                relevant = find_relevant_chunks(question, s.chunks, s.vectorizer, s.matrix)
                if not relevant:
                    answer = "I couldn't find that in the document."
                else:
                    try:
                        answer = ask_gemini(client, question, relevant)
                    except Exception as e:
                        answer = ("The AI service is busy right now. Please try again in a minute." if "503" in str(e) or "429" in str(e) else f"Something went wrong calling the AI model: {e}")
            st.markdown(answer)
            if relevant:
                with st.expander("Sources used from the document"):
                    for i, chunk in enumerate(relevant, 1):
                        st.markdown(f"**Chunk {i}:** {chunk[:300]}...")
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a PDF to get started.")
