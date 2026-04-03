#!/usr/bin/env python3
"""
Collect metadata (stars, about, README) for top repos in the test set,
and save results to disk.

Usage: python Repo_EDA.py [N=5] [M=100]
  N  — number of top repos to print (by stars)
  M  — minimum sample count filter
"""

import base64
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
M = int(sys.argv[2]) if len(sys.argv) > 2 else 100
OUTPUT_DIR = Path(__file__).parent.parent / "dataset" / "repo_metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GH_TOKEN = os.getenv("GH_TOKEN")
SESSION = requests.Session()
if GH_TOKEN:
    SESSION.headers.update({"Authorization": f"token {GH_TOKEN}"})
    print("Authenticated with GitHub token.")
else:
    print("No GH_TOKEN found — using unauthenticated requests (60 req/hr limit).")


def fetch_repo_metadata(repo: str) -> dict:
    """Fetch stars, about, and README for a repo (owner/name format)."""
    meta = {"repo": repo, "stars": 0, "about": None, "readme": None}
    try:
        resp = SESSION.get(f"https://api.github.com/repos/{repo}", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            meta["stars"] = data.get("stargazers_count", 0)
            meta["about"] = data.get("description")
    except Exception:
        return meta

    try:
        resp = SESSION.get(f"https://api.github.com/repos/{repo}/readme", timeout=10)
        if resp.status_code == 200:
            encoded = resp.json().get("content", "")
            meta["readme"] = base64.b64decode(encoded).decode("utf-8", errors="replace")
    except Exception:
        pass

    return meta


print(f"\nTop {N} repos (min {M} samples) by GitHub stars\n")
print("=" * 70)

all_results = {}

for lang in ["java", "python"]:
    lang_path = OUTPUT_DIR / f"{lang}_repo_metadata.json"
    dataset = load_dataset("google/code_x_glue_ct_code_to_text", lang)
    df = pd.DataFrame(dataset["test"])
    unique_repos = set(df["repo"].unique())

    # Load cached records
    cached: dict[str, dict] = {}
    if lang_path.exists():
        with open(lang_path) as f:
            for record in json.load(f):
                cached[record["repo"]] = record

    missing = unique_repos - cached.keys()
    print(f"\n{lang.upper()}: {len(unique_repos)} repos total, "
          f"{len(cached)} cached, {len(missing)} to fetch")

    # Fetch only missing repos
    newly_fetched: dict[str, dict] = {}
    if missing:
        for repo in tqdm(sorted(missing), desc="Fetching metadata"):
            newly_fetched[repo] = fetch_repo_metadata(repo)

    # Merge and rebuild sample_count from current df
    repo_meta: dict[str, dict] = {}
    for repo in unique_repos:
        base = cached.get(repo) or newly_fetched.get(repo, {"repo": repo, "stars": 0, "about": None, "readme": None})
        repo_meta[repo] = {
            **base,
            "language": lang,
            "sample_count": int((df["repo"] == repo).sum()),
        }

    # Save if anything new was fetched
    if missing:
        records = list(repo_meta.values())
        with open(lang_path, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(records)} records → {lang_path}")
    else:
        print("All metadata loaded from cache, nothing to save.")

    all_results[lang] = list(repo_meta.values())

    # Filter by min samples, then sort by stars, take top N
    filtered = [m for m in repo_meta.values() if m["sample_count"] >= M]
    top = sorted(filtered, key=lambda m: m["stars"], reverse=True)[:N]

    print(f"Total samples in test: {len(df)}")
    print(f"Repos with >= {M} samples: {len(filtered)}\n")

    for i, meta in enumerate(top, 1):
        about = (meta["about"] or "")[:60]
        print(
            f"{i:2d}. {meta['repo']:40s} ⭐ {meta['stars']:6d} stars | "
            f"{meta['sample_count']:3d} samples | {about}"
        )

# Save combined
combined_path = OUTPUT_DIR / "all_repo_metadata.json"
with open(combined_path, "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\nCombined results → {combined_path}")

print("\n" + "=" * 70)

# --- Plot: distribution of sample counts per repo ---
fig, axes = plt.subplots(1, len(all_results), figsize=(7 * len(all_results), 5), sharey=False)
if len(all_results) == 1:
    axes = [axes]

for ax, (lang, records) in zip(axes, all_results.items()):
    counts = [r["sample_count"] for r in records]
    ax.hist(counts, bins=100, edgecolor="white", linewidth=0.5)
    ax.axvline(M, color="red", linestyle="--", linewidth=1.2, label=f"M={M} filter")
    ax.set_title(f"{lang.upper()} — {len(records)} repos")
    ax.set_xlabel("Samples per repo")
    ax.set_ylabel("Number of repos")
    ax.legend()

fig.suptitle("Distribution of sample counts per repo (test set)", fontsize=13)
fig.tight_layout()

plot_path = OUTPUT_DIR / "sample_distribution.png"
fig.savefig(plot_path, dpi=150)
print(f"Plot saved → {plot_path}")
plt.show()
