# eden_bot_v3.py
#FRZKALMAR
# Eden Bot — Hybrid Semantic + Lexical RAG (Safe, Cached, High-Quality)

import ollama
import argparse
import os
import re
import numpy as np
import hashlib
import nltk
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL = "llama3.2:latest"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 800
OVERLAP_SENTENCES = 2

TOP_N = 8
MMR_LAMBDA = 0.85
SIMILARITY_FALLBACK = 0.18
MIN_SIMILARITY_THRESHOLD = 0.05

MAX_CONTEXT_CHARS = 6000


# ============================================================
# Utilities
# ============================================================

def log_step(msg):
    print(f"→ {msg}")

def section(title):
    print("\n" + "─" * 40)
    print(title)
    print("─" * 40)

def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

def hash_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()[:16]


# ============================================================
# EPUB Extraction
# ============================================================

def extract_text_from_epub(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    book = epub.read_epub(path)
    parts = []

    for item in book.get_items_of_type(ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)

    full_text = "\n".join(parts).strip()

    if not full_text:
        raise RuntimeError("No readable text found in EPUB.")

    return full_text


# ============================================================
# Chunking
# ============================================================

def split_text(text):
    sentences = sent_tokenize(text)
    chunks = []
    i = 0

    while i < len(sentences):
        current = []
        length = 0
        j = i

        while j < len(sentences) and length + len(sentences[j]) <= CHUNK_SIZE:
            current.append(sentences[j])
            length += len(sentences[j])
            j += 1

        if not current and j < len(sentences):
            current.append(sentences[j])
            j += 1

        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)

        i = max(j - OVERLAP_SENTENCES, i + 1)

    return chunks


# ============================================================
# Embeddings (Book-Specific Caching)
# ============================================================

def embed_text(text, model):
    return np.array(
        ollama.embeddings(model=model, prompt=text)["embedding"]
    )

def embed_chunks(chunks, model, cache_file):

    if os.path.exists(cache_file):
        log_step("Loading cached embeddings...")
        return np.load(cache_file)

    log_step("Creating embeddings (first run only)...")

    embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"\rEmbedding {i+1}/{len(chunks)}", end="", flush=True)
        embeddings.append(embed_text(chunk, model))

    embeddings = np.vstack(embeddings)
    np.save(cache_file, embeddings)

    print("\n→ Embeddings cached.")
    return embeddings


# ============================================================
# Hybrid Retrieval (Semantic + Lexical + MMR)
# ============================================================

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def lexical_score(question, chunk):
    question_lower = question.lower()
    chunk_lower = chunk.lower()

    score = 0.0

    # Exact phrase boost
    if question_lower in chunk_lower:
        score += 0.35

    # Word overlap boost
    q_words = set(re.findall(r"\w+", question_lower))
    c_words = set(re.findall(r"\w+", chunk_lower))

    if q_words:
        overlap = q_words.intersection(c_words)
        score += 0.25 * (len(overlap) / len(q_words))

    return score

def retrieve(question, question_emb, chunk_embs, chunks, debug=False):

    similarities = np.array([
        cosine_sim(question_emb, emb) for emb in chunk_embs
    ])

    # Apply lexical boosting
    for i in range(len(chunks)):
        similarities[i] += lexical_score(question, chunks[i])

    max_sim = similarities.max()

    if debug:
        section("DEBUG: Similarity Stats")
        print(f"Top similarity after boost: {max_sim:.4f}")

    if max_sim < SIMILARITY_FALLBACK:
        log_step("Weak match detected. Using top similarity fallback.")
        return np.argsort(similarities)[::-1][:TOP_N]

    selected = []
    candidates = list(range(len(chunk_embs)))

    while len(selected) < TOP_N and candidates:
        scores = []

        for idx in candidates:
            relevance = similarities[idx]

            if relevance < MIN_SIMILARITY_THRESHOLD:
                continue

            if not selected:
                diversity = 0
            else:
                diversity = max(
                    cosine_sim(chunk_embs[idx], chunk_embs[s])
                    for s in selected
                )

            mmr_score = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * diversity
            scores.append((idx, mmr_score))

        if not scores:
            break

        best_idx = max(scores, key=lambda x: x[1])[0]
        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected


# ============================================================
# Context Builder
# ============================================================

def build_context(indices, chunks):
    indices = sorted(indices)
    context = ""

    for idx in indices:
        chunk = chunks[idx]
        if len(context) + len(chunk) > MAX_CONTEXT_CHARS:
            break
        context += chunk + "\n\n"

    return context.strip()


# ============================================================
# LLM Answer
# ============================================================

def ask_ollama(question, context, model):

    prompt = f"""
You are answering questions about a book.

CRITICAL RULES:
- Use ONLY the provided context.
- Do NOT use outside knowledge.
- If answer is not present, respond exactly:
  Not found in document.
- Do NOT guess.

DEPTH REQUIREMENTS:
- Provide a thorough explanation.
- Quote short relevant phrases when useful.
- Combine ideas from multiple sections if needed.
- Explain relationships and implications described in the text.
- Avoid shallow summaries.

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
        print(chunk["message"]["content"], end="", flush=True)

    print("\n")


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(description="Eden Bot v3 — Hybrid EPUB RAG")
    parser.add_argument("epub_path", help="Path to EPUB file")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    ensure_nltk()

    print("\nExtracting text...")
    text = extract_text_from_epub(args.epub_path)
    log_step(f"{len(text):,} characters extracted")

    print("\nSplitting into chunks...")
    chunks = split_text(text)
    log_step(f"{len(chunks)} chunks created")

    file_hash = hash_file(args.epub_path)
    cache_file = f"eden_embed_{file_hash}.npy"

    chunk_embeddings = embed_chunks(chunks, args.embed_model, cache_file)

    print("\nEden Bot v3 ready! (type 'quit' to exit)")

    while True:
        question = input("\nAsk: ").strip()

        if not question:
            continue
        if question.lower() == "quit":
            break

        section("QUESTION")
        print(question)

        question_emb = embed_text(question, args.embed_model)

        indices = retrieve(
            question,
            question_emb,
            chunk_embeddings,
            chunks,
            debug=args.debug
        )

        if not indices:
            print("No relevant context found.")
            continue

        context = build_context(indices, chunks)

        ask_ollama(question, context, args.model)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nError:", e)