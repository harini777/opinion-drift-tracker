# Opinion Drift Tracker

A tool that scrapes live Hacker News data, uses sentence embeddings and K-Means clustering to discover opinion clusters without any pre-defined labels, and generates an AI report on the state of discourse.

## What it does

Type any topic → the app scrapes Hacker News live → SBERT embeddings convert posts into high-dimensional semantic meaning-vectors → K-Means clustering partitions the vector space into distinct opinion themes → a GenAI layer auto-labels each cluster and writes an analytical "state of discourse" report.

## Tech Stack

- **Data** — Hacker News comments and stories via Algolia API (no auth required)
- **NLP** — Sentence-BERT (`sentence-transformers`) for generating 384-dimensional dense semantic vectors
- **ML** — K-Means clustering (`scikit-learn`) for unsupervised grouping + UMAP (`umap-learn`) for 2D semantic projection
- **GenAI** — Groq API (LLaMA 3.3 70B) for cluster label synthesis and report generation
- **Frontend & Visuals** — Streamlit & Plotly Express

## Project Structure

```
opinion-drift-tracker/
│
├── hn_scraper.py        # Phase 1: Scrapes HN posts via Algolia API
├── embeddings.py        # Phase 2: Converts posts to 384-dim vectors using SBERT
├── clustering.py        # Phase 3: K-Means clustering + silhouette analysis + UMAP
├── genai_layer.py       # Phase 4: Groq LLM labels clusters and generates discourse report
├── app.py               # Streamlit web app (with Cyber-Modern UI Overhaul)
├── style.css            # Custom CSS style overrides (glowing glass panels, modern tech fonts)
├── run.sh               # Shell helper script to launch Streamlit in the virtual environment
├── data/                # Local data storage — excluded from Git
├── .env                 # API keys — excluded from Git
├── .gitignore
└── requirements.txt
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/opinion-drift-tracker.git
cd opinion-drift-tracker
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

## Usage

### Phase Execution via CLI
You can run the pipeline scripts step-by-step from the command line:

```bash
# Step 1 — Scrape Hacker News
python3 hn_scraper.py

# Step 2 — Generate embeddings
python3 embeddings.py

# Step 3 — Cluster opinions
python3 clustering.py

# Step 4 — Label clusters and generate report
python3 genai_layer.py
```

### Running the Web Dashboard
Launch the unified Streamlit interface:

```bash
# Recommended: Launches within the configured virtual environment to prevent import conflicts
./run.sh

# Or run directly if venv is activated
streamlit run app.py
```

## How it works

### Scraping
The Algolia Hacker News API is queried for stories and comments matching the user's topic. No authentication is required. Scraped posts are parsed, filtered, and saved as a CSV to the local `data/` folder.

### Embeddings (Semantic Processing)
Each post is converted into a 384-dimensional vector using `all-MiniLM-L6-v2` from Sentence-BERT. Unlike traditional word-matching (TF-IDF or bag-of-words), SBERT models capture deep contextual semantics. Posts containing different words but sharing similar sentiments or themes will map to vectors that are close to each other in vector space.

### Clustering (Unsupervised Machine Learning)
K-Means clustering groups the normalized embeddings into distinct opinion themes. The pipeline uses silhouette coefficient checks to evaluate the separation of the groups. UMAP (Uniform Manifold Approximation and Projection) reduces the 384-dimensional vectors down to 2 dimensions, enabling you to inspect the layout of clusters as an interactive 2D scatter plot in Plotly.

### GenAI Layer (Automatic Thematic Synthesis)
Representative comments from the center of each K-Means cluster are gathered and sent to Groq's LLaMA 3.3 70B model. The LLM labels each group with a descriptive 4-to-7 word title and synthesizes a 150–200 word "State of Discourse" summary detailing dominant narratives, conflicting views, and where the overall conversation is headed.

## Key concepts covered

- **REST APIs & JSON Parsing:** Fetching and clean-up of live community data.
- **Unsupervised Machine Learning:** Semantic partitioning of raw natural language datasets without prior labels.
- **Dense Vector Embeddings:** Sentence-BERT semantic encoding for contextual text understanding.
- **Dimensionality Reduction:** Compressing high-dimensional embeddings for 2D visual analysis using UMAP.
- **Prompt Engineering with LLMs:** Synthesizing precise categorical labels and reports from numeric cluster samples.
- **Streamlit Development:** Creating premium, responsive machine learning dashboard interfaces.

## Roadmap

- [ ] Deploy to Hugging Face Spaces with a live public URL
- [ ] Add drift timeline — track how opinion clusters shift over time
- [ ] Support multiple data sources beyond Hacker News

