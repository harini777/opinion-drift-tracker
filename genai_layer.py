"""
Phase 4 — GenAI Layer
Uses Groq (fast) to label clusters and generate discourse report.
"""

import os
import glob
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def load_clustered_data() -> pd.DataFrame:
    files = glob.glob("data/*_clustered.csv")
    if not files:
        print("No clustered CSV found. Run clustering.py first.")
        exit()

    print("\nAvailable clustered datasets:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")

    choice = int(input("\nWhich dataset? Enter number: ")) - 1
    df = pd.read_csv(files[choice])
    print(f"Loaded {len(df)} posts across {df['cluster'].nunique()} clusters.\n")
    return df


def get_cluster_samples(df: pd.DataFrame, cluster_id: int, n: int = 8) -> str:
    posts = df[df["cluster"] == cluster_id]["text"].dropna().head(n).tolist()
    return "\n".join(f"{i+1}. {p[:200]}" for i, p in enumerate(posts))


def label_cluster(cluster_id: int, samples: str) -> str:
    prompt = f"""You are analyzing online discussion posts grouped by semantic similarity.

Here are {len(samples.splitlines())} posts from one group:

{samples}

Give this group a short, specific label (4-7 words) that captures the dominant opinion or narrative.
Reply with ONLY the label. No explanation, no punctuation at the end."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30
    )
    return response.choices[0].message.content.strip()


def generate_report(topic: str, cluster_labels: dict, df: pd.DataFrame) -> str:
    cluster_summary = ""
    total = len(df)
    for cid, label in cluster_labels.items():
        count = len(df[df["cluster"] == cid])
        pct = round(count / total * 100, 1)
        sample = df[df["cluster"] == cid]["text"].dropna().iloc[0][:150]
        cluster_summary += f'\nCluster "{label}" ({pct}% of posts):\n  e.g. "{sample}"\n'

    prompt = f"""You are an analyst summarizing public opinion from Hacker News discussions.

Topic: "{topic}"
Total posts analyzed: {total}

Here are the opinion clusters discovered by an ML algorithm:
{cluster_summary}

Write a 150-200 word plain-English "State of Discourse" report that:
- Describes the dominant narratives people hold about this topic
- Notes which viewpoints are most and least common
- Highlights any interesting tensions or contradictions between clusters
- Ends with one sentence on what this suggests about where the conversation is heading

Write in a clear, analytical tone. No bullet points — flowing paragraphs only."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400
    )
    return response.choices[0].message.content.strip()


def save_report(topic: str, labels: dict, report: str):
    filename = f"data/{topic}_report.txt"
    with open(filename, "w") as f:
        f.write(f"OPINION DRIFT REPORT: {topic.upper()}\n")
        f.write("=" * 60 + "\n\n")
        f.write("CLUSTER LABELS\n")
        f.write("-" * 30 + "\n")
        for cid, label in labels.items():
            f.write(f"  Cluster {cid}: {label}\n")
        f.write("\n\nSTATE OF DISCOURSE\n")
        f.write("-" * 30 + "\n")
        f.write(report)
    print(f"\nReport saved to {filename}")


if __name__ == "__main__":
    df = load_clustered_data()
    topic = df["topic"].iloc[0] if "topic" in df.columns else "unknown_topic"

    print("Labeling clusters with Groq...\n")
    cluster_labels = {}
    for cid in sorted(df["cluster"].unique()):
        samples = get_cluster_samples(df, cid)
        label = label_cluster(cid, samples)
        cluster_labels[cid] = label
        count = len(df[df["cluster"] == cid])
        print(f"  Cluster {cid} ({count} posts): {label}")

    print("\nGenerating State of Discourse report...\n")
    report = generate_report(topic, cluster_labels, df)
    print(report)

    save_report(topic, cluster_labels, report)
    print("\nPhase 4 complete. Run app.py next for the full web UI.")