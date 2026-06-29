import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import plotly.express as px

# Robust import for UMAP
umap_available = True
try:
    import umap
except ImportError:
    try:
        from umap import umap_ as umap
    except ImportError:
        umap_available = False

# Import your own modules
from hn_scraper import scrape_hn
from genai_layer import label_cluster, generate_report, get_cluster_samples

load_dotenv()

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Opinion Drift Tracker // Console",
    page_icon="🖥️",
    layout="wide"
)

# ─────────────────────────────────────────────────────────
# LOAD STYLE SHEET
# ─────────────────────────────────────────────────────────
if os.path.exists("style.css"):
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CACHE HEAVY OBJECTS
# so the model doesn't reload every time user clicks a button
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# ─────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────
if 'topic' not in st.session_state:
    st.session_state['topic'] = ""
if 'auto_run' not in st.session_state:
    st.session_state['auto_run'] = False

# ─────────────────────────────────────────────────────────
# HEADER / MASTHEAD
# ─────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="masthead-container">
        <h1 class="masthead-title">Opinion Drift Tracker</h1>
        <p class="masthead-subtitle">// SEMANTIC PIPELINE & ACTIVE DISCOURSE REPORT: HACKER NEWS</p>
        <div class="masthead-meta">
            <span>SYS_STATUS: ACTIVE</span>
            <span>BUILD: 2026.06.29</span>
            <span>CONSOLE: V1.0.0</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────
# SIDEBAR — controls
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    max_posts = st.slider(
        "Scrape Sample Size",
        min_value=50, max_value=300, value=150, step=50,
        help="Higher counts yield richer patterns but increase scraping and embedding duration."
    )
    k = st.slider(
        "Semantic Cluster Resolution (K)",
        min_value=2, max_value=8, value=5,
        help="The target number of distinct opinion themes to identify."
    )
    st.divider()
    st.markdown("### 📰 Pipeline Methodology")
    st.markdown(
        """
        1. **Live Scraping:** Scrapes active threads and comment branches matching your query.
        2. **Semantic Encoding:** Converts post text into 384-dimensional dense vectors using SBERT.
        3. **Clustering Analysis:** Applies K-Means on normalized vectors to identify cluster themes.
        4. **Dimensionality Reduction:** Projects high-dimensional vectors to 2D using UMAP.
        5. **AI Synthesis:** Generates human-like thematic labels and reports via Groq AI.
        """
    )

# ─────────────────────────────────────────────────────────
# MAIN INPUT
# ─────────────────────────────────────────────────────────
topic = st.text_input(
    "Enter a topic to analyse",
    value=st.session_state['topic'],
    placeholder="e.g. AI is overhyped, Is Python dying, Remote work future",
    help="Try controversial tech topics for the most interesting clusters"
)

# Keep session state updated with typed inputs
st.session_state['topic'] = topic

run_btn = st.button("🔍 Run Pipeline Search", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────
# MAIN PIPELINE — runs when button is clicked or auto-triggered
# ─────────────────────────────────────────────────────────
should_run = run_btn or st.session_state.get('auto_run', False)

if should_run and topic.strip():
    # Reset auto run flag immediately
    st.session_state['auto_run'] = False

    # Check for UMAP package availability before starting
    if not umap_available:
        st.error(
            "⚠️ Dependency Error: The 'umap-learn' package is not installed in the active python environment. "
            "Please install it using: `pip install umap-learn` or run the app via the virtual environment: "
            "`./venv/bin/streamlit run app.py`"
        )
        st.stop()

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

    # Cyber Neon Palette Colors
    custom_colors = ["#00F0FF", "#8B5CF6", "#10B981", "#EC4899", "#FBBF24", "#3B82F6", "#EF4444", "#8C96A3"]

    # ── Cluster size breakdown ──
    st.markdown("## 📊 SEMANTIC CLUSTER ANALYSIS", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<h3 style='margin-bottom: 1rem; color: #F4F0EA; font-family: Georgia, serif;'>THEMATIC BUCKETS</h3>", unsafe_allow_html=True)
        # Custom infographic cards
        for cid, label in cluster_labels.items():
            count = len(df[df['cluster'] == cid])
            pct   = round(count / len(df) * 100, 1)
            color_class = ["card-cyan", "card-violet", "card-lime", "card-pink", "card-yellow"][cid % 5]
            accent_color = custom_colors[cid % len(custom_colors)]
            st.markdown(
                f"""
                <div class="editorial-card {color_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 0.3rem; margin-bottom: 0.5rem;">
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: {accent_color}; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1rem;">CLUSTER_{cid}</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #8C96A3; font-weight: 600;">{count} POSTS • {pct}%</span>
                    </div>
                    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 700; color: #F4F0EA; line-height: 1.35; letter-spacing: 0.01em;">
                        {label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col2:
        # Donut chart
        sizes  = [len(df[df['cluster'] == c]) for c in cluster_labels]
        labels_list = list(cluster_labels.values())
        
        fig_pie = px.pie(
            values=sizes,
            names=labels_list,
            color_discrete_sequence=custom_colors[:len(labels_list)],
            hole=0.45
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace", color="#8C96A3"),
            title=dict(
                text="<b>CLUSTER SHARE DIST</b>",
                font=dict(family="Space Grotesk, sans-serif", size=18, color="#00F0FF")
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=50, b=50, l=0, r=0)
        )
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent',
            marker=dict(line=dict(color='#07080A', width=2))
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── UMAP scatter plot ──
    st.markdown("<h2 style='margin-top: 2rem;'>🗺️ 2D SEMANTIC VECTOR PROJECT</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #8C96A3; margin-top: -0.5rem; margin-bottom: 1.5rem; font-family: \"JetBrains Mono\", monospace; font-size: 0.82rem;'>"
        "// EACH NODE IS A HACKER NEWS ELEMENT. CLUSTERING REVEALS SENTIMENT DENSITIES."
        "</p>",
        unsafe_allow_html=True
    )

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
        color_discrete_sequence=custom_colors[:len(cluster_labels)],
        height=550
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#8C96A3"),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0, 240, 255, 0.03)",
            zeroline=False,
            showticklabels=False,
            title=None
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0, 240, 255, 0.03)",
            zeroline=False,
            showticklabels=False,
            title=None
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=10, b=50, l=10, r=10)
    )
    fig.update_traces(
        marker=dict(
            size=8,
            opacity=0.85,
            line=dict(color='#07080A', width=0.5)
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Discourse report ──
    st.markdown("<h2 style='margin-top: 3rem;'>📝 AI DISCOURSE SYNTHESIS</h2>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="report-container">
            {report}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Sample posts per cluster ──
    st.markdown("<h2 style='margin-top: 3rem;'>💬 REPRESENTATIVE DISCOURSE EXTRACTS</h2>", unsafe_allow_html=True)
    for cid, label in cluster_labels.items():
        with st.expander(f"CLUSTER {cid} // {label}"):
            cluster_posts = df[df['cluster'] == cid]['text'].dropna().head(5).tolist()
            for post in cluster_posts:
                clean_post = post.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
                st.markdown(
                    f"""
                    <div class="quote-card">
                        <div class="quote-text">{clean_post[:350]}...</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ── Save report ──
    os.makedirs("data", exist_ok=True)
    fname = topic[:30].replace(" ", "_")
    df.to_csv(f"data/{fname}_clustered.csv", index=False)

elif run_btn and not topic.strip():
    st.warning("Please enter a topic first!")

else:
    # ── Landing state ──
    st.markdown("<h3 style='text-align: center; margin-top: 3rem; font-family: \"Space Grotesk\", sans-serif; color: #00F0FF;'>📰 SYSTEM SEARCH PRESETS</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8C96A3; margin-bottom: 2rem; font-family: \"Plus Jakarta Sans\"; font-size: 0.9rem;'>Select an active investigation query to launch the live semantic parser</p>", unsafe_allow_html=True)
    
    cols = st.columns(3)
    examples = [
        {"title": "AI is overhyped", "desc": "Is the artificial intelligence bubble nearing its limit?", "meta": "150 POSTS"},
        {"title": "Is Python dying", "desc": "Debating performance vs. developer velocity in 2026.", "meta": "150 POSTS"},
        {"title": "Remote work productivity", "desc": "Analyzing corporate mandates vs. engineer autonomy.", "meta": "150 POSTS"},
        {"title": "Crypto is dead", "desc": "Parsing the post-hype utility of blockchain technology.", "meta": "150 POSTS"},
        {"title": "React vs Vue", "desc": "The endless frontend war and architectural fatigue.", "meta": "150 POSTS"},
        {"title": "Open source sustainability", "desc": "How maintainers are coping with licensing and burnout.", "meta": "150 POSTS"}
    ]
    
    for i, ex in enumerate(examples):
        with cols[i % 3]:
            accent_color = ["#8B5CF6", "#00F0FF", "#10B981"][i % 3]
            st.markdown(
                f"""
                <div style="border: 1px solid rgba(0, 240, 255, 0.08); background: rgba(15, 17, 21, 0.7); border-radius: 6px; padding: 1.25rem; height: 130px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 0.5rem; border-left: 3px solid {accent_color}; box-shadow: 0 4px 12px rgba(0,0,0,0.35);">
                    <div>
                        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; color: #F4F0EA; font-weight: 700; line-height: 1.2; letter-spacing: 0.02em;">{ex['title']}</div>
                        <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.78rem; color: #8C96A3; margin-top: 0.3rem; line-height: 1.35;">{ex['desc']}</div>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: {accent_color}; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700;">{ex['meta']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button(f"RUN SEARCH: {ex['title']} →", key=f"ex_btn_{i}", use_container_width=True):
                st.session_state['topic'] = ex['title']
                st.session_state['auto_run'] = True
                st.rerun()