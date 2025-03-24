# Reddit API data collection tool with post tracking

"""
This file will "scrape" posts from forums on Reddit by interacting with Reddit's Python API "PRAW"
Posts are sought by most recent. There used to be a way of searching further back by date, but apparently Reddit doesn't work this way any more.
So, instead, the script compiles the most recent mental health posts from the specified forums.
It can be run periodically to enlarge the dataset.
The script will only download posts that haven't previously been accessed, by making reference to their ID's, which are saved in a .txt file in the data folder.
"""

import praw
import json
import os
import time
from datetime import datetime, timezone

# Fill in your Reddit app credentials:
reddit = praw.Reddit(
    client_id="dVGQfNLECbg765wP6m9nkw",
    client_secret="_1DtxzYyS5zmp_9bGKzanUnMNCHd6A",
    user_agent="MentalHealthNLP"
)

# Subreddits to scrape
subreddits = ["mentalhealth", "depression", "anxiety", "mentalillness"]
max_posts_per_subreddit = 1000
posts_per_category = max_posts_per_subreddit // 3
categories = {
    "hot": lambda sub: sub.hot(limit=posts_per_category),
    "new": lambda sub: sub.new(limit=posts_per_category),
    "top": lambda sub: sub.top(limit=posts_per_category),
}

# Load previously seen post IDs
seen_ids_path = "data/raw/seen_ids.txt"
seen_ids = set()
if os.path.exists(seen_ids_path):
    with open(seen_ids_path, "r") as f:
        seen_ids = set(f.read().splitlines())

# Load existing data (if any)
output_path = "data/raw/reddit_posts.json"
all_posts = []
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        all_posts = json.load(f)

# Determine scrape run number
existing_scrape_runs = [post.get("scrape_run", 0) for post in all_posts]
current_scrape_run = max(existing_scrape_runs, default=0) + 1

# Start scraping
new_posts = []

for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    print(f"\n🔎 Scraping r/{subreddit_name}...")

    for category_name, fetch_function in categories.items():
        print(f"  📂 Fetching {posts_per_category} from '{category_name}'...")
        time.sleep(2)
        try:
            for post in fetch_function(subreddit):
                if (
                    post.id in seen_ids
                    or post.stickied
                    or len(post.selftext.strip()) < 50
                ):
                    continue

                seen_ids.add(post.id)
                new_post = {
                    "scrape_run": current_scrape_run,
                    "subreddit": subreddit_name,
                    "id": post.id,
                    "title": post.title,
                    "body": post.selftext,
                    "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(),
                    "url": post.url,
                    "score": post.score,
                    "category": category_name,
                }
                all_posts.append(new_post)
                new_posts.append(new_post)
        except Exception as e:
            print(f"    ⚠️ Skipped '{category_name}' due to error: {e}")
        time.sleep(2)

# Save updated dataset
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_posts, f, ensure_ascii=False, indent=2)

# Save updated seen_ids
with open(seen_ids_path, "w", encoding="utf-8") as f:
    f.write("\n".join(seen_ids))

print(f"\n✅ Done! {len(new_posts)} new posts collected.")
print("Note: though the script is written to scrape up to 333 from 'hot', 'new' and 'top' in four forums - meaning a total of ~4,000 posts (333 * 3 * 4), the actual number returned may be lower for a number of reasons, including that posts under 50 characters are excluded, and that some subreddits may not have much recent content. Despite precautions, the script may also hit a temporary rate limit or Reddit API issue.")
print(f"📦 Total dataset size: {len(all_posts)} posts")
print(f"📁 Saved to: {output_path}")