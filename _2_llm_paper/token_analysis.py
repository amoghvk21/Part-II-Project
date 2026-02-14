"""
Measure the token overhead of the prompt template (excluding article text).
Tests with 1 currency and all 10 currencies to show the range.
"""

from transformers import AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login
import os

from models_code.finetuned_llama import generate_prompt

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 8192
ALL_CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD', 'NOK', 'SEK']

# ── Auth & tokenizer ───────────────────────────────────────────────────────
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.model_max_length = MAX_SEQ_LENGTH
tokenizer.pad_token = '<|reserved_special_token_0|>'
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|reserved_special_token_0|>')
print("Tokenizer loaded.\n")

# ── Test with different currency counts ─────────────────────────────────────
print(f"{'Currencies':>12}  {'Overhead tokens':>16}  {'Remaining for article':>22}")
print("-" * 56)

for n in [1, 2, 3, 5, 10]:
    currencies = ALL_CURRENCIES[:n]

    # Build a fake row with empty article
    dummy_row = {
        'Title': '',
        'Full Text': '',
        'mentioned_currencies': currencies,
    }
    # Add dummy labels for training prompt
    for c in currencies:
        dummy_row[f'{c}_past_label'] = 'unchanged'
        dummy_row[f'{c}_future_label'] = 'unchanged'

    prompt = generate_prompt(dummy_row, tokenizer, is_training=True)
    n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

    print(f"{n:>12}  {n_tokens:>16}  {MAX_SEQ_LENGTH - n_tokens:>22}")

print(f"\nMax context window: {MAX_SEQ_LENGTH} tokens")


# results 
# 950 tokens for 10 currencies
# around to 1000