# eden_bot.py — Robust Local EPUB RAG with Ollama

import ollama
import argparse
import os
import re
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL = "llama3.2:latest"
CHUNK_SIZE = 800
OVERLAP_SENTENCES = 2
TOP_N = 5
MIN_SIMILARITY = 0.05
MAX_CONTEXT_CHARS = 4000


# ============================================================
# NLTK Setup
# ============================================================

def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


# ============================================================
# EPUB Extraction
# ============================================================

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


# ============================================================
# Sentence-Based Chunking
# ============================================================

def split_text(text, chunk_size=CHUNK_SIZE, overlap_sentences=OVERLAP_SENTENCES):
    sentences = sent_tokenize(text)
    chunks = []
    i = 0

    while i < len(sentences):
        current_chunk = []
        current_length = 0
        j = i

        while j < len(sentences) and current_length + len(sentences[j]) <= chunk_size:
            current_chunk.append(sentences[j])
            current_length += len(sentences[j])
            j += 1

        chunk_text = " ".join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

        i = max(j - overlap_sentences, i + 1)

    return chunks


# ============================================================
# TF-IDF Vectorization (Improved)
# ============================================================

def create_vectors(chunks):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",  # KEEP single-character tokens like "1"
        max_df=0.95
    )

    vectors = vectorizer.fit_transform(chunks)
    return vectorizer, vectors


# ============================================================
# Query Normalization
# ============================================================

def normalize_question(question):
    """
    If user asks 'What is Law 1?' convert to:
    'What is the title and explanation of Law 1?'
    """
    match = re.search(r"\blaw\s+(\d+)\b", question.lower())
    if match:
        number = match.group(1)
        return f"What is the title and explanation of Law {number}?"
    return question


# ============================================================
# Retrieval
# ============================================================

def get_top_chunks(question, chunks, vectorizer, vectors,
                   top_n=TOP_N, min_score=MIN_SIMILARITY):

    question_vector = vectorizer.transform([question])
    similarity = cosine_similarity(question_vector, vectors).flatten()

    sorted_indices = similarity.argsort()[::-1]

    results = []
    for idx in sorted_indices[:top_n]:
        score = similarity[idx]
        if score >= min_score:
            results.append((chunks[idx], score))

    return results


# ============================================================
# Context Builder
# ============================================================

def build_context(top_chunks):
    context = ""
    for chunk, _ in top_chunks:
        if len(context) + len(chunk) > MAX_CONTEXT_CHARS:
            break
        context += chunk + "\n\n"
    return context.strip()


# ============================================================
# Ollama Call (Improved Prompt)
# ============================================================

def ask_ollama(question, context, model):
    prompt = f"""
You are answering questions about a document.

Use ONLY the provided context.

If the answer can be directly determined or reasonably inferred 
from the context (including structured titles like "LAW 1"),
return it clearly.

If the answer truly does not appear in the context, respond exactly:
Not found in document.

Do not use prior knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model=model,
        options={"temperature": 0.2},
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"].strip()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Eden Bot Pro v2 - EPUB RAG with Ollama")
    parser.add_argument("epub_path", help="Path to EPUB file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--debug", action="store_true", help="Show similarity scores")

    args = parser.parse_args()

    ensure_nltk()

    print("\nExtracting text...")
    text = extract_text_from_epub(args.epub_path)
    print("Text length:", len(text))

    print("Splitting text into chunks...")
    chunks = split_text(text)
    print("Number of chunks:", len(chunks))

    print("Creating TF-IDF vectors...")
    vectorizer, vectors = create_vectors(chunks)

    print("\nEden Bot Pro v2 is ready! (type 'quit' to exit)")

    while True:
        question = input("\nAsk: ").strip()

        if not question:
            continue

        if question.lower() == "quit":
            break

        normalized_question = normalize_question(question)

        top_chunks = get_top_chunks(
            normalized_question,
            chunks,
            vectorizer,
            vectors
        )

        if not top_chunks:
            print("\nNo relevant context found.")
            continue

        if args.debug:
            print("\n--- Retrieval Debug ---")
            for _, score in top_chunks:
                print(f"Similarity: {score:.4f}")
            print("-----------------------")

        context = build_context(top_chunks)

        answer = ask_ollama(normalized_question, context, args.model)

        print("\nAnswer:\n", answer)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nError:", e)