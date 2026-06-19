import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
import plotly.express as px
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Import your own modules
from hn_scraper import scrape_hn
from genai_layer import label_cluster, generate_report, get_cluster_samples

load_dotenv()

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Opinion Drift Tracker",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────────────────────────
# CACHE HEAVY OBJECTS
# so the model doesn't reload every time user clicks a button
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.title("🧠 Opinion Drift Tracker")
st.markdown(
    "Discover how people on **Hacker News** actually think about any topic — "
    "clustered by meaning, explained by AI."
)
st.divider()

# ─────────────────────────────────────────────────────────
# SIDEBAR — controls
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    max_posts = st.slider(
        "Posts to scrape",
        min_value=50, max_value=300, value=150, step=50,
        help="More posts = better clusters but slower"
    )
    k = st.slider(
        "Number of opinion clusters (K)",
        min_value=2, max_value=8, value=5,
        help="How many distinct opinion groups to find"
    )
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. 🔍 Scrapes HN live")
    st.markdown("2. 🧬 SBERT converts text → meaning vectors")
    st.markdown("3. 🎯 K-Means finds opinion clusters")
    st.markdown("4. 🗺️ UMAP visualises in 2D")
    st.markdown("5. 🤖 Groq AI labels and reports")

# ─────────────────────────────────────────────────────────
# MAIN INPUT
# ─────────────────────────────────────────────────────────
topic = st.text_input(
    "Enter a topic to analyse",
    placeholder="e.g. AI is overhyped, Is Python dying, Remote work future",
    help="Try controversial tech topics for the most interesting clusters"
)

run_btn = st.button("🔍 Analyse Opinions", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────
# MAIN PIPELINE — runs when button is clicked
# ─────────────────────────────────────────────────────────
if run_btn and topic.strip():

    # ── Step 1: Scrape ──
    with st.status("🔍 Scraping Hacker News...", expanded=True) as status:
        st.write(f"Searching for: *{topic}*")
        df = scrape_hn(topic, max_results=max_posts)
        st.write(f"✅ Collected **{len(df)} posts and comments**")
        status.update(label="Scraping complete!", state="complete")

    if len(df) < 20:
        st.error("Not enough posts found. Try a broader topic.")
        st.stop()

    # ── Step 2: Embeddings ──
    with st.status("🧬 Generating meaning vectors...", expanded=True) as status:
        model = load_model()
        st.write("Converting posts to 384-dimensional semantic vectors...")
        embeddings = model.encode(df['text'].tolist(), show_progress_bar=False)
        st.write(f"✅ **{embeddings.shape[0]} posts** → each a vector of **{embeddings.shape[1]} numbers**")
        status.update(label="Embeddings ready!", state="complete")

    # ── Step 3: Clustering ──
    with st.status("🎯 Discovering opinion clusters...", expanded=True) as status:
        normed = normalize(embeddings)
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(normed)
        sil    = silhouette_score(normed, labels)
        df['cluster'] = labels
        st.write(f"✅ Found **{k} clusters** — Silhouette score: **{sil:.3f}**")
        st.caption("Silhouette score: closer to 1.0 = well-separated clusters")
        status.update(label="Clusters found!", state="complete")

    # ── Step 4: UMAP ──
    with st.status("🗺️ Building 2D opinion map...", expanded=True) as status:
        reducer  = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        coords   = reducer.fit_transform(embeddings)
        status.update(label="Map ready!", state="complete")

    # ── Step 5: GenAI labels ──
    with st.status("🤖 Asking AI to label clusters...", expanded=True) as status:
        cluster_labels = {}
        for cid in sorted(df['cluster'].unique()):
            samples = get_cluster_samples(df, cid)
            label   = label_cluster(cid, samples)
            cluster_labels[cid] = label
            st.write(f"Cluster {cid}: *{label}*")
        status.update(label="Labels generated!", state="complete")

    # ── Step 6: Discourse report ──
    with st.status("📝 Writing discourse report...", expanded=True) as status:
        report = generate_report(topic, cluster_labels, df)
        status.update(label="Report ready!", state="complete")

    st.divider()

    # ─────────────────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────────────────

    # ── Cluster size breakdown ──
    st.subheader("📊 Opinion Cluster Breakdown")
    col1, col2 = st.columns([1, 1])

    with col1:
        # Metrics per cluster
        for cid, label in cluster_labels.items():
            count = len(df[df['cluster'] == cid])
            pct   = round(count / len(df) * 100, 1)
            st.metric(label=f"Cluster {cid}", value=label, delta=f"{pct}% of posts")

    with col2:
        # Pie chart
        sizes  = [len(df[df['cluster'] == c]) for c in cluster_labels]
        labels_list = list(cluster_labels.values())
        fig_pie = px.pie(
            values=sizes,
            names=labels_list,
            title="Share of each opinion cluster",
            template="plotly_dark"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── UMAP scatter plot ──
    st.subheader("🗺️ 2D Opinion Map")
    st.caption("Every dot is a real HN post. Hover to read it. Same colour = same opinion cluster.")

    plot_df = pd.DataFrame({
        'x'      : coords[:, 0],
        'y'      : coords[:, 1],
        'cluster': [cluster_labels[l] for l in labels],
        'text'   : df['text'].str[:150] + "...",
        'score'  : df['score']
    })

    fig = px.scatter(
        plot_df,
        x='x', y='y',
        color='cluster',
        hover_data={'text': True, 'score': True, 'x': False, 'y': False},
        template="plotly_dark",
        height=550
    )
    fig.update_traces(marker=dict(size=7, opacity=0.8))
    st.plotly_chart(fig, use_container_width=True)

    # ── Discourse report ──
    st.subheader("📝 State of Discourse")
    st.info(report)

    # ── Sample posts per cluster ──
    st.subheader("💬 Sample Posts Per Cluster")
    for cid, label in cluster_labels.items():
        with st.expander(f"Cluster {cid}: {label}"):
            cluster_posts = df[df['cluster'] == cid]['text'].head(5).tolist()
            for post in cluster_posts:
                st.markdown(f"- {post[:250]}")

    # ── Save report ──
    os.makedirs("data", exist_ok=True)
    fname = topic[:30].replace(" ", "_")
    df.to_csv(f"data/{fname}_clustered.csv", index=False)

elif run_btn and not topic.strip():
    st.warning("Please enter a topic first!")

else:
    # ── Landing state ──
    st.markdown("### 👆 Enter a topic above to get started")
    st.markdown("**Example topics to try:**")
    cols = st.columns(3)
    examples = [
        "AI is overhyped", "Is Python dying",
        "Remote work productivity", "Crypto is dead",
        "React vs Vue", "Open source sustainability"
    ]
    for i, ex in enumerate(examples):
        with cols[i % 3]:
            if st.button(ex, use_container_width=True):
                st.session_state['topic'] = ex