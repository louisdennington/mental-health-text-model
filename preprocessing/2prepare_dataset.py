# prepare_dataset.py - SBERT-friendly text preprocessing

import json
import os
import re

# Define SBERT-style light cleaner
def clean_text_sbert(text):
    text = text.strip()  # Remove leading/trailing whitespace
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"\s+", " ", text)  # Normalize multiple spaces
    return text

# Load original dataset
input_path = "data/raw/reddit_posts.json"
output_path = "data/processed/cleaned_posts.json"

with open(input_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Clean text
cleaned_data = []

for post in raw_data:
    combined_text = post["title"].strip() + " " + post["body"].strip()
    cleaned_post = {
        "id": post["id"],
        "subreddit": post["subreddit"],
        "category": post["category"],
        "text": clean_text_sbert(combined_text)
    }
    cleaned_data.append(cleaned_post)

# Save cleaned dataset
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"✅ SBERT-friendly cleaning complete. Saved {len(cleaned_data)} posts to {output_path}")