import requests          # makes HTTP calls — like your browser, but in Python
import pandas as pd      # stores data in a table (like Excel)
from datetime import datetime, timezone
import time

# ─────────────────────────────────────────────────────────
# NO CREDENTIALS NEEDED — Algolia's HN API is fully public
# Just a URL → you get data back. That's it.
# ─────────────────────────────────────────────────────────

BASE_URL = "https://hn.algolia.com/api/v1/search"


# ─────────────────────────────────────────────────────────
# CLEAN TEXT
# ─────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    HN comments come with HTML tags like <p> and &amp;
    This strips them out so we get plain readable text.
    """
    if not text:
        return ""
    # Remove common HTML tags
    import re
    text = re.sub(r'<[^>]+>', ' ', text)     # strip <p>, <a href=...>, etc.
    text = text.replace('&amp;', '&')         # fix HTML entities
    text = text.replace('&gt;', '>')
    text = text.replace('&lt;', '<')
    text = text.replace('&#x27;', "'")
    text = text.replace('&quot;', '"')
    text = " ".join(text.split())             # collapse whitespace
    return text.strip()


# ─────────────────────────────────────────────────────────
# SINGLE API CALL
# ─────────────────────────────────────────────────────────
def fetch_page(query: str, content_type: str, page: int) -> dict:
    """
    Makes one API call to Algolia's HN search endpoint.

    Args:
        query        : the search term e.g. "AI is overhyped"
        content_type : "comment" or "story"
        page         : which page of results (0, 1, 2...)

    Returns:
        The raw JSON response as a Python dictionary
    """
    # These are called "query parameters" — they go after the ? in a URL
    params = {
        "query"       : query,
        "tags"        : content_type,   # filter to comments or stories only
        "hitsPerPage" : 50,             # 50 results per page
        "page"        : page
    }

    # requests.get() is like typing a URL in your browser
    # params get added to the URL automatically: ?query=...&tags=...
    response = requests.get(BASE_URL, params=params)

    # .raise_for_status() throws an error if something went wrong
    # (e.g. 404 not found, 500 server error)
    response.raise_for_status()

    # .json() converts the response text into a Python dictionary
    return response.json()


# ─────────────────────────────────────────────────────────
# MAIN SCRAPER
# ─────────────────────────────────────────────────────────
def scrape_hn(topic: str, max_results: int = 200) -> pd.DataFrame:
    """
    Scrapes Hacker News stories + comments about any topic.

    Args:
        topic       : what to search for e.g. "is AI overhyped"
        max_results : how many total posts to collect

    Returns:
        A pandas DataFrame with columns:
        id, text, source, platform, created_at, score, url
    """
    records = []

    print(f"\n Scraping Hacker News for: '{topic}'...")
    print(" No API key needed — this is the beauty of open APIs!\n")

    # ── We scrape two types of content ──
    # Stories = the original post/link someone shared
    # Comments = the discussion underneath (richer opinions)
    for content_type in ["story", "comment"]:

        collected = 0
        page      = 0
        per_type  = max_results // 2   # split evenly between stories + comments

        while collected < per_type:

            # Fetch one page of results
            data = fetch_page(topic, content_type, page)

            # 'hits' is HN's word for "results"
            hits = data.get("hits", [])

            # If no results came back, we've hit the end — stop
            if not hits:
                break

            for hit in hits:
                # ── Extract text depending on content type ──
                if content_type == "story":
                    # Stories have a title and sometimes a body (self-posts)
                    text = (hit.get("title") or "") + " " + (hit.get("story_text") or "")
                else:
                    # Comments have a comment_text field
                    text = hit.get("comment_text") or ""

                text = clean_text(text)

                # Skip anything too short to carry a real opinion
                if len(text) < 40:
                    continue

                # ── Parse the timestamp ──
                # HN gives us time as a Unix timestamp (seconds since 1970)
                # We convert it to a readable datetime object
                created_raw = hit.get("created_at_i") or 0
                created_at  = datetime.fromtimestamp(created_raw, tz=timezone.utc)

                # ── Build one row of our table ──
                records.append({
                    "id"        : hit.get("objectID"),
                    "text"      : text,
                    "source"    : "Hacker News",
                    "platform"  : "hackernews",
                    "created_at": created_at,
                    "score"     : hit.get("points") or 0,  # upvotes on HN
                    "url"       : hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
                })

                collected += 1
                if collected >= per_type:
                    break

            page += 1

            # Be polite — don't hammer the API
            time.sleep(0.3)

    # ── Convert to DataFrame ──
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset="id")
    df = df.reset_index(drop=True)

    print(f" Collected {len(df)} stories + comments")
    print(f" Date range: {df['created_at'].min().strftime('%Y-%m')} → {df['created_at'].max().strftime('%Y-%m')}\n")

    return df


# ─────────────────────────────────────────────────────────
# SAVE TO CSV
# ─────────────────────────────────────────────────────────
def save_data(df: pd.DataFrame, topic: str) -> str:
    """Saves scraped data to the data/ folder as a CSV file."""
    filename = topic[:30].replace(" ", "_").replace("/", "-")
    filepath = f"data/{filename}.csv"
    df.to_csv(filepath, index=False)
    print(f" Saved to {filepath}")
    print(f" Open this file in VS Code to see your data!\n")
    return filepath


# ─────────────────────────────────────────────────────────
# RUN TO TEST
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    topic = input("Enter a topic to search (e.g. 'AI is overhyped'): ")

    df = scrape_hn(topic, max_results=200)

    filepath = save_data(df, topic)

    print("── Preview of your data ──────────────────────────────")
    print(df[["source", "score", "created_at", "text"]].head(10).to_string(index=False))
    print(f"\nTotal rows collected : {len(df)}")