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
    User types two sentences and sees how similar they are.
    Demonstrates that SBERT understands meaning, not just words.
    """
    print("── Semantic Similarity Checker ───────────────────────────")
    print("Type two sentences and see how similar they are in meaning.")
    print("(This uses the same model that will cluster your HN posts)\n")

    while True:
        s1 = input("Sentence 1: ").strip()
        s2 = input("Sentence 2: ").strip()

        e1, e2 = model.encode([s1, s2])
        score = cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1))[0][0]

        if score > 0.6:
            label = "✅ Very similar meaning"
        elif score > 0.3:
            label = "🟡 Loosely related"
        else:
            label = "❌ Unrelated"

        print(f"\nSimilarity score: {score:.3f}  —  {label}\n")

        again = input("Try another pair? (y/n): ").strip().lower()
        if again != 'y':
            break

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

        
        