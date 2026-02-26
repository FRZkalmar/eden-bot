# eden_bot.py — Generic Local EPUB RAG with Ollama (Book-Agnostic)

import ollama
import argparse
import os
import re
import time
import threading
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
# UI Helpers
# ============================================================

def log_step(message):
    print(f"→ {message}")

def print_header():
    print("\n" + "─" * 40)
    print(" Eden Bot (Universal Book Mode)")
    print("─" * 40)

def section(title):
    print("\n" + "─" * 40)
    print(title)
    print("─" * 40)

def moving_dots(message, stop_event):
    dots = ""
    while not stop_event.is_set():
        dots = "." if dots == "..." else dots + "."
        print(f"\r→ {message}{dots}   ", end="", flush=True)
        time.sleep(0.4)
    print("\r" + " " * 60 + "\r", end="", flush=True)


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
# Sentence-Based Chunking (Edge Case Safe)
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

        # Edge case: single long sentence
        if not current_chunk and j < len(sentences):
            current_chunk.append(sentences[j])
            j += 1

        chunk_text = " ".join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

        i = max(j - overlap_sentences, i + 1)

    return chunks


# ============================================================
# TF-IDF Vectorization
# ============================================================

def create_vectors(chunks):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
        max_df=0.95
    )

    vectors = vectorizer.fit_transform(chunks)
    return vectorizer, vectors


# ============================================================
# Generic Heading-Aware Normalization
# ============================================================

def normalize_question(question):
    pattern = r"\b(chapter|part|section|law|act|article)\s+(\d+)\b"
    match = re.search(pattern, question.lower())

    if match:
        label = match.group(1).capitalize()
        number = match.group(2)
        return f"What is the title and explanation of {label} {number}?"

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
# Ollama Call (Enhanced Depth Version)
# ============================================================

def ask_ollama(question, context, model):
    prompt = f"""
You are answering questions about a book.

CRITICAL RULES:
- Use ONLY the provided context.
- Do NOT use prior knowledge.
- If the answer truly does not appear in the context, respond exactly:
  Not found in document.
- Do NOT guess.
- Do NOT generalize beyond the text.

DEPTH REQUIREMENTS:
- Provide a thorough and detailed explanation.
- Use the book's wording, phrasing, and terminology whenever possible.
- If helpful, quote short relevant phrases from the context.
- Combine ideas from multiple parts of the context when appropriate.
- Explain not only WHAT the text says, but HOW and WHY it explains it.
- Clarify definitions, implications, and relationships described in the text.
- Avoid short summaries. Provide structured, well-developed answers.

Structure your answer clearly using paragraphs.

Context:
{context}

Question:
{question}

Answer:
"""

    section("ANSWER")

    response = ollama.chat(
        model=model,
        options={"temperature": 0.2},
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in response:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)

    print("\n")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Eden Bot - Generic EPUB RAG")
    parser.add_argument("epub_path", help="Path to EPUB file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--debug", action="store_true", help="Show similarity scores")

    args = parser.parse_args()

    print_header()
    ensure_nltk()

    print("\nExtracting text...")
    text = extract_text_from_epub(args.epub_path)
    log_step(f"Text extracted ({len(text):,} characters)")

    print("\nSplitting text into chunks...")
    chunks = split_text(text)
    log_step(f"{len(chunks)} chunks created")

    print("\nCreating TF-IDF vectors...")
    vectorizer, vectors = create_vectors(chunks)
    log_step("Vector index built")

    print("\nEden Bot is ready! (type 'quit' to exit)")

    while True:
        question = input("\nAsk: ").strip()

        if not question:
            continue

        if question.lower() == "quit":
            break

        section("QUESTION")
        print(question)

        normalized_question = normalize_question(question)

        print("\nRetrieving relevant context...")
        top_chunks = get_top_chunks(
            normalized_question,
            chunks,
            vectorizer,
            vectors
        )

        if not top_chunks:
            print("No relevant context found.")
            continue

        log_step(f"Retrieved {len(top_chunks)} relevant sections")

        if args.debug:
            section("DEBUG: Similarity Scores")
            for chunk, score in top_chunks:
                print(f"[Score: {score:.4f}] {chunk[:200]}...\n")

        context = build_context(top_chunks)

        ask_ollama(normalized_question, context, args.model)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nError:", e)