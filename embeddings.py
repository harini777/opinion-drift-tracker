from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────
# WHAT IS HAPPENING HERE?
#
# SentenceTransformer loads a pre-trained ML model.
# "Pre-trained" means someone already trained it on
# hundreds of millions of sentences — we just use it.
# This is called "transfer learning" — a huge concept in ML.
#
# 'all-MiniLM-L6-v2' is the model name. It's small, fast,
# and produces 384-dimensional vectors. Perfect for us.
# ─────────────────────────────────────────────────────────
def load_model():
    print("Loading SBERT model...")
    print("(First time only: downloads ~80MB. After that it's cached.)\n")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded!\n")
    return model


# ─────────────────────────────────────────────────────────
# GENERATE EMBEDDINGS
# ─────────────────────────────────────────────────────────
def generate_embeddings(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    """
    Converts every post in your DataFrame into a vector.

    Args:
        df    : your scraped HN data (from Phase 1)
        model : the loaded SBERT model

    Returns:
        A 2D numpy array of shape (num_posts, 384)
        Think of it as a table: each row is one post,
        each column is one dimension of its meaning.
    """
    texts = df['text'].tolist()  # extract just the text column as a list

    print(f"Converting {len(texts)} posts into meaning-vectors...")
    print("Each post becomes 384 numbers capturing its semantic meaning.\n")

    # .encode() is the core call — this is where the ML happens
    # show_progress_bar=True prints a progress bar so you can watch it work
    embeddings = model.encode(texts, show_progress_bar=True)

    print(f"\nDone! Shape of embeddings: {embeddings.shape}")
    print(f"  → {embeddings.shape[0]} posts, each with {embeddings.shape[1]} dimensions\n")

    return embeddings


# ─────────────────────────────────────────────────────────
# SANITY CHECK — does it actually understand meaning?
# ─────────────────────────────────────────────────────────
def sanity_check(model: SentenceTransformer):
    """
    This is the most important step for LEARNING.
    We test the model on sentences we understand
    to verify it actually captures semantic similarity.

    Cosine similarity score:
        1.0  = identical meaning
        0.7+ = very similar
        0.3  = loosely related
        0.0  = completely different
       -1.0  = opposite meaning
    """
    print("── Sanity Check: Does SBERT understand meaning? ──────────")
    print("We'll compare sentence pairs and check similarity scores.\n")

    sentence_pairs = [
        # Should be HIGH similarity (same idea, different words)
        ("AI will take all our jobs",
         "Artificial intelligence is going to replace human workers"),

        # Should be MEDIUM similarity (related topic)
        ("Python is the best programming language",
         "I prefer coding in JavaScript"),

        # Should be LOW similarity (completely different)
        ("AI will take all our jobs",
         "I had pasta for dinner last night"),
    ]

    # Collect all sentences and embed them together (more efficient)
    all_sentences = [s for pair in sentence_pairs for s in pair]
    all_embeddings = model.encode(all_sentences)

    for i, (s1, s2) in enumerate(sentence_pairs):
        # cosine_similarity measures the angle between two vectors
        # We reshape because sklearn expects 2D arrays
        e1 = all_embeddings[i * 2].reshape(1, -1)
        e2 = all_embeddings[i * 2 + 1].reshape(1, -1)
        score = cosine_similarity(e1, e2)[0][0]

        print(f"Sentence A: \"{s1}\"")
        print(f"Sentence B: \"{s2}\"")
        print(f"Similarity: {score:.3f}  {'✅ HIGH' if score > 0.6 else '🟡 MEDIUM' if score > 0.3 else '❌ LOW'}")
        print()


# ─────────────────────────────────────────────────────────
# SAVE EMBEDDINGS
# ─────────────────────────────────────────────────────────
def save_embeddings(embeddings: np.ndarray, topic: str):
    """
    Saves embeddings as a .npy file (numpy's native format).
    Much faster than CSV for numerical arrays.
    Phase 3 (clustering) will load this file.
    """
    filename = topic[:30].replace(" ", "_").replace("/", "-")
    filepath = f"data/{filename}_embeddings.npy"
    np.save(filepath, embeddings)
    print(f"Embeddings saved to {filepath}")
    print(f"File size: {embeddings.nbytes / 1024:.1f} KB\n")
    return filepath


# ─────────────────────────────────────────────────────────
# RUN TO TEST
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Step 1: Load your scraped data from Phase 1 ──
    import os
    csv_files = [f for f in os.listdir("data") if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in data/ folder.")
        print("Run hn_scraper.py first to scrape some data!")
    else:
        print("Available datasets:")
        for i, f in enumerate(csv_files):
            print(f"  {i+1}. {f}")

        choice = int(input("\nWhich file to use? Enter number: ")) - 1
        filepath = f"data/{csv_files[choice]}"
        topic    = csv_files[choice].replace(".csv", "")

        df = pd.read_csv(filepath)
        print(f"\nLoaded {len(df)} posts from {filepath}\n")

        # ── Step 2: Load SBERT model ──
        model = load_model()

        # ── Step 3: Sanity check — understand what embeddings do ──
        sanity_check(model)

        # ── Step 4: Generate embeddings for all your posts ──
        embeddings = generate_embeddings(df, model)

        # ── Step 5: Save for Phase 3 ──
        save_embeddings(embeddings, topic)

        print("Phase 2 complete! Your posts are now meaning-vectors.")
        print("Next: Phase 3 will cluster these into opinion groups.")