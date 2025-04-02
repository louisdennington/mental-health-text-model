import json
import os
import re

def clean_text_sbert(text):
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    
    # Remove common Redditisms
    reddit_terms = ["tw", "vent", "rant", "update", "throwra", "ama", "crosspost", "cross-post"]
    for term in reddit_terms:
        text = re.sub(rf"\b{term}\b", "", text)

    # Remove mental health labels that could bias clustering - this part seemed to remove too much information, resulting in large amorphous clusters
    #keyword_patterns = [
    #    r"\bocd\b", r"\bptsd\b", r"\bcptsd\b", r"\bbpd\b",
    #    r"\baddict(?:ed|ion)?\b", r"\beating disorder\b", r"\banorexia\b", r"\bbulimia\b",
    #    r"\badhd\b", r"\bautism\b", r"\bautistic\b", r"\bpsychosis\b", r"\bpsychotic\b",
    #    r"\bdepression\b",  # keep "depressed", "depressing", etc.
    #    r"\bemotional neglect\b",
    #    r"\banxious attachment\b", r"\bavoidant attachment\b"
    #]
    #for pattern in keyword_patterns:
    #    text = re.sub(pattern, "", text)

    # Final whitespace cleanup
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

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