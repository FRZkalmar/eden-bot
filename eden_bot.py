# eden_bot.py — Stable Ollama RAG Version

import ollama
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import sys

nltk.download("punkt")


# -----------------------------
# Step 1: Extract text from EPUB
# -----------------------------
def extract_text_from_epub(epub_path):
    try:
        book = epub.read_epub(epub_path)
    except Exception as e:
        print("Error reading EPUB:", e)
        sys.exit()

    text = ""

    # Correct way to get readable content
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text += soup.get_text(separator=" ", strip=True) + "\n"

    text = text.strip()

    if not text:
        print("No readable text found in EPUB.")
        sys.exit()

    return text


# -----------------------------
# Step 2: Split text into chunks
# -----------------------------
def split_text(text, chunk_size=800):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    if not chunks:
        print("Text splitting failed. No chunks created.")
        sys.exit()

    return chunks


# -----------------------------
# Step 3: Vectorize chunks
# -----------------------------
def create_vectors(chunks):
    try:
        vectorizer = TfidfVectorizer(
            stop_words=None,      # prevent empty vocabulary issues
            lowercase=True
        )
        vectors = vectorizer.fit_transform(chunks)
    except ValueError as e:
        print("Vectorization failed:", e)
        sys.exit()

    return vectorizer, vectors


# -----------------------------
# Step 4: Retrieve top N chunks
# -----------------------------
def get_top_chunks(question, chunks, vectorizer, vectors, top_n=3):
    question_vector = vectorizer.transform([question])
    similarity = cosine_similarity(question_vector, vectors)
    indices = similarity.flatten().argsort()[-top_n:][::-1]
    return [chunks[i] for i in indices]


# -----------------------------
# Step 5: Ask Ollama
# -----------------------------
def ask_ollama(question, context):
    prompt_text = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say "Not found in document."

Context:
{context}

Question: {question}
Answer:
"""

    response = ollama.chat(
        model="llama3.2:latest",  # change if needed
        messages=[{"role": "user", "content": prompt_text}],
    )

    return response["message"]["content"]


# -----------------------------
# Main Program
# -----------------------------
def main():
    epub_path = input("Enter EPUB file path: ").strip()

    print("\nExtracting text...")
    text = extract_text_from_epub(epub_path)
    print("Text length:", len(text))

    print("Splitting text into chunks...")
    chunks = split_text(text)
    print("Number of chunks:", len(chunks))

    print("Creating TF-IDF vectors...")
    vectorizer, vectors = create_vectors(chunks)

    print("\nEden Bot is ready! Ask your questions (type 'quit' to exit).")

    while True:
        question = input("\nAsk: ").strip()

        if question.lower() == "quit":
            break

        top_chunks = get_top_chunks(
            question, chunks, vectorizer, vectors, top_n=3
        )

        context = "\n\n".join(top_chunks)

        answer = ask_ollama(question, context)

        print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()