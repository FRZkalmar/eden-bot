# eden_bot_web.py — FastAPI version of Eden Bot

import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk
import os

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Eden Bot Web RAG")

# -----------------------------
# NLTK setup
# -----------------------------
def ensure_nltk():
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource)
ensure_nltk()

# -----------------------------
# RAG Utilities
# -----------------------------
def extract_text_from_epub(epub_path):
    if not os.path.exists(epub_path):
        raise FileNotFoundError(f"File not found: {epub_path}")
    book = epub.read_epub(epub_path)
    text_parts = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        clean_text = soup.get_text(separator=" ", strip=True)
        if clean_text:
            text_parts.append(clean_text)
    text = "\n".join(text_parts).strip()
    if not text:
        raise RuntimeError("No readable text found in EPUB.")
    return text

def split_text(text, chunk_size=600, overlap=120):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:]
            current_chunk = overlap_text + " " + sentence
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def create_vectors(chunks):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.9
    )
    vectors = vectorizer.fit_transform(chunks)
    return vectorizer, vectors

def get_top_chunks(question, chunks, vectorizer, vectors, top_n=5):
    question_vector = vectorizer.transform([question])
    similarity = cosine_similarity(question_vector, vectors).flatten()
    top_indices = similarity.argsort()[-top_n:][::-1]
    return [chunks[i] for i in top_indices]

def ask_ollama(question, context):
    prompt_text = f"""
You are answering questions about a document.

Use ONLY the provided context.
If the answer is not explicitly stated in the context, respond exactly:
Not found in document.

Do not use prior knowledge.

Context:
{context}

Question: {question}
Answer:
"""
    response = ollama.chat(
        model="llama3.2:latest",
        options={"temperature": 0.2},
        messages=[{"role": "user", "content": prompt_text}],
    )
    return response["message"]["content"]

# -----------------------------
# Load EPUB once at startup
# -----------------------------
EPUB_PATH = "48laws.epub"  # put your book here
print("Loading book...")
text = extract_text_from_epub(EPUB_PATH)
chunks = split_text(text)
vectorizer, vectors = create_vectors(chunks)
print(f"Loaded {len(chunks)} chunks from {EPUB_PATH}")

# -----------------------------
# FastAPI request model
# -----------------------------
class QuestionRequest(BaseModel):
    question: str

# -----------------------------
# Ask endpoint
# -----------------------------
@app.post("/ask")
def ask(request: QuestionRequest):
    try:
        top_chunks = get_top_chunks(request.question, chunks, vectorizer, vectors, top_n=5)
        context = "\n\n".join(top_chunks)
        answer = ask_ollama(request.question, context)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))