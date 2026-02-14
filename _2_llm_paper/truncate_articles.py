"""
Truncate articles in cached dataframes to MAX_ARTICLE_TOKENS tokens
and save the truncated versions back to cache.
"""

import pandas as pd
from transformers import AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login
import os
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_ARTICLE_TOKENS = 7192

# ── Auth & tokenizer ───────────────────────────────────────────────────────
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print("Tokenizer loaded.\n")

# ── Load cached dataframes ─────────────────────────────────────────────────
print("Loading cached dataframes...")
df_train = pd.read_pickle("_2_llm_paper/cache/df_train.pkl")
df_test  = pd.read_pickle("_2_llm_paper/cache/df_test.pkl")
df_eval  = pd.read_pickle("_2_llm_paper/cache/df_eval.pkl")

# ── Truncate ────────────────────────────────────────────────────────────────
def truncate_text(text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > MAX_ARTICLE_TOKENS:
        return tokenizer.decode(tokens[:MAX_ARTICLE_TOKENS], skip_special_tokens=True)
    return text

for name, df in [("train", df_train), ("test", df_test), ("eval", df_eval)]:
    print(f"Truncating {name} ({len(df)} articles)...")
    tqdm.pandas(desc=name)
    df['Full Text'] = df['Full Text'].progress_apply(truncate_text)

# ── Save ────────────────────────────────────────────────────────────────────
print("\nSaving truncated dataframes...")
os.makedirs("_2_llm_paper/cache_truncated", exist_ok=True)
df_train.to_pickle("_2_llm_paper/cache_truncated/df_train.pkl")
df_test.to_pickle("_2_llm_paper/cache_truncated/df_test.pkl")
df_eval.to_pickle("_2_llm_paper/cache_truncated/df_eval.pkl")
print("Done. Cached dataframes updated.")
