import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import umap.umap_ as umap
import plotly.express as px
import plotly.graph_objects as go
import os

# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
def load_data(topic: str):
    """
    Loads your scraped posts + their embeddings.
    Both files were saved in earlier phases.
    """
    filename = topic[:30].replace(" ", "_").replace("/", "-")

    csv_path = f"data/{filename}.csv"
    emb_path = f"data/{filename}_embeddings.npy"

    df         = pd.read_csv(csv_path)
    embeddings = np.load(emb_path)

    print(f"Loaded {len(df)} posts and their embeddings ({embeddings.shape})\n")
    return df, embeddings


# ─────────────────────────────────────────────────────────
# ELBOW METHOD — find the right number of clusters
# ─────────────────────────────────────────────────────────
def find_optimal_k(embeddings: np.ndarray, max_k: int = 10):
    """
    Runs K-Means for k=2 to max_k and plots inertia.

    Inertia = sum of distances from each point to its
    cluster centre. Lower = more compact clusters.

    The 'elbow' in the plot is where adding more clusters
    stops giving meaningful improvement — that's your K.
    """
    print("Running elbow method to find optimal number of clusters...")
    print("(Testing K=2 through K=10)\n")

    # Normalize embeddings — makes distance calculations more stable
    # Think of it as putting all vectors on the same scale
    normed = normalize(embeddings)

    inertias = []
    k_values = range(2, max_k + 1)

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(normed)
        inertias.append(km.inertia_)
        print(f"  K={k}  inertia={km.inertia_:.2f}")

    # Plot the elbow curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(k_values),
        y=inertias,
        mode='lines+markers',
        marker=dict(size=8, color='#7c3aed'),
        line=dict(color='#7c3aed', width=2)
    ))
    fig.update_layout(
        title="Elbow Method — Find Optimal K",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia (lower = more compact)",
        template="plotly_dark",
        width=700, height=400
    )
    fig.show()

    print("\nLook at the chart — find where the curve bends (the elbow).")
    print("That K value gives you the best cluster quality.\n")

    return inertias


# ─────────────────────────────────────────────────────────
# RUN K-MEANS
# ─────────────────────────────────────────────────────────
def run_kmeans(embeddings: np.ndarray, k: int) -> np.ndarray:
    """
    Runs K-Means with your chosen K.

    Returns an array of cluster labels — one per post.
    e.g. [0, 2, 1, 0, 3, 2, ...] means:
      post 0 → cluster 0
      post 1 → cluster 2
      post 2 → cluster 1
      etc.
    """
    print(f"Running K-Means with K={k}...")

    normed = normalize(embeddings)

    # random_state=42 makes results reproducible
    # n_init=10 runs K-Means 10 times and picks the best result
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(normed)

    # Show how many posts landed in each cluster
    unique, counts = np.unique(labels, return_counts=True)
    print("\nCluster sizes:")
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} posts")

    return labels


# ─────────────────────────────────────────────────────────
# UMAP VISUALISATION — see your clusters in 2D
# ─────────────────────────────────────────────────────────
def visualise_clusters(embeddings: np.ndarray, labels: np.ndarray, df: pd.DataFrame):
    """
    UMAP reduces 384 dimensions → 2 dimensions so we can
    plot every post as a dot on a 2D chart.

    Similar posts stay close together even in 2D.
    Each colour = one cluster = one opinion narrative.

    You can hover over any dot to read the actual post!
    """
    print("\nReducing dimensions with UMAP for visualisation...")
    print("(Compressing 384 dimensions → 2 dimensions)\n")

    reducer = umap.UMAP(
        n_components=2,    # reduce to 2D
        random_state=42,   # reproducible layout
        n_neighbors=15,    # how much local vs global structure to preserve
        min_dist=0.1       # how tightly to pack points
    )

    coords_2d = reducer.fit_transform(embeddings)

    # Build a DataFrame for plotting
    plot_df = pd.DataFrame({
        'x'      : coords_2d[:, 0],
        'y'      : coords_2d[:, 1],
        'cluster': [f"Cluster {l}" for l in labels],
        'text'   : df['text'].str[:120] + "...",  # truncate for hover
        'score'  : df['score']
    })

    fig = px.scatter(
        plot_df,
        x='x', y='y',
        color='cluster',
        hover_data={'text': True, 'score': True, 'x': False, 'y': False},
        title="Opinion Clusters — Hover over any dot to read the post",
        template="plotly_dark",
        width=900, height=600
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.show()

    print("Each colour = one opinion cluster.")
    print("Dots close together = posts with similar meaning.")
    print("Hover over any dot to read the actual HN post!\n")


# ─────────────────────────────────────────────────────────
# SHOW SAMPLE POSTS PER CLUSTER
# ─────────────────────────────────────────────────────────
def show_cluster_samples(df: pd.DataFrame, labels: np.ndarray, k: int):
    """
    Prints 3 sample posts from each cluster so you can
    understand what opinion each cluster represents.
    This is how you manually interpret what the algorithm found.
    """
    df = df.copy()
    df['cluster'] = labels

    print("── Sample posts per cluster ──────────────────────────────\n")
    for cluster_id in range(k):
        cluster_posts = df[df['cluster'] == cluster_id]['text'].tolist()
        print(f"CLUSTER {cluster_id} ({len(cluster_posts)} posts)")
        print("-" * 50)
        for post in cluster_posts[:3]:
            print(f"  • {post[:200]}")
        print()


# ─────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────
def save_results(df: pd.DataFrame, labels: np.ndarray, topic: str):
    """Saves posts with their cluster labels for Phase 4."""
    df = df.copy()
    df['cluster'] = labels
    filename = topic[:30].replace(" ", "_").replace("/", "-")
    filepath = f"data/{filename}_clustered.csv"
    df.to_csv(filepath, index=False)
    print(f"Saved clustered data to {filepath}")
    return filepath


# ─────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Pick dataset ──
    csv_files = [f for f in os.listdir("data") if f.endswith(".csv")]
    print("Available datasets:")
    for i, f in enumerate(csv_files):
        print(f"  {i+1}. {f}")

    choice = int(input("\nWhich dataset? Enter number: ")) - 1
    topic  = csv_files[choice].replace(".csv", "")

    # ── Load data + embeddings ──
    df, embeddings = load_data(topic)

    # ── Step 1: Find optimal K ──
    find_optimal_k(embeddings)
    k = int(input("Enter your chosen K (look at the elbow in the chart): "))

    # ── Step 2: Run K-Means ──
    labels = run_kmeans(embeddings, k)

    # ── Step 3: Visualise ──
    visualise_clusters(embeddings, labels, df)

    # ── Step 4: Read sample posts per cluster ──
    show_cluster_samples(df, labels, k)

    # ── Step 5: Save for Phase 4 ──
    save_results(df, labels, topic)
