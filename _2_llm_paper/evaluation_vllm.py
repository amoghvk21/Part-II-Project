# %% [markdown]
# # Implementing "FX sentiment analysis with large language models" (Ballinari et al.)
# ## vLLM-accelerated evaluation
# Uses vLLM for significantly faster batched inference via PagedAttention and continuous batching.

# %% [markdown]
# ## Imports

# %%
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os
from huggingface_hub import login
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

print("imports done")

get_model_id = {
    'a': "meta-llama/Meta-Llama-3.1-8B-Instruct",
    'b': "meta-llama/Meta-Llama-3-70B-Instruct",

    'c': "meta-llama/Meta-Llama-3.1-8B",
    'd': "meta-llama/Meta-Llama-3-70B",

    'e': "meta-llama/Meta-Llama-4-Maverick-8B-Instruct",
    'f': "meta-llama/Meta-Llama-4-Scout-8B-Instruct",

    'g': "mistralai/Mistral-Nemo-Instruct",

    'h': "Qwen/Qwen3-32B-Chat",
}

# Model selection
MODEL = 'a'
MODEL_LOAD_DIR = 'finetuned_llama_8b'
MODEL_ID = get_model_id[MODEL]
LORA_ADAPTER_PATH = f"_2_llm_paper/models/{MODEL_LOAD_DIR}/model"
MAX_SEQ_LENGTH = 8192

USE_LORA = True   # Set to False for base model evaluation (no LoRA adapter)

# %% [markdown]
# ## Prompt Generation (same as finetuned_llama.py)

# %%
def generate_prompt(row, tokenizer):
    title = row.get('Title', '')
    text = row.get('Full Text', '')
    currencies = row.get('mentioned_currencies')

    target_currencies = ''
    for c in currencies:
        target_currencies += f'{c}_past: "appreciation, depreciation, or unchanged",\n'
        target_currencies += f'{c}_future: "appreciation, depreciation, or unchanged",\n'
    target_currencies = target_currencies.strip().rstrip(",") # Remove last comma

    # Same structure as per paper
    prompt = (
        f"Title: {title}\n"
        f"Text: {text}\n\n"
        "Instructions:\n"
        "Objective: For each mentioned currency, answer the following questions:\n"
        "- What has been the current/past movement of the currency (appreciation, depreciation, or unchanged)?\n"
        "- What is the future expectation for the currency (appreciation, depreciation, or unchanged)?\n\n"
        "You must answer these two questions for each of the following currencies mentioned in the article:\n"
        f"{target_currencies}\n\n"
        "Output Format:\n"
        "- Important: Provide your answer in separate rows for each currency as shown above.\n"
        "- Do not combine multiple currencies in the same row.\n"
        '- Each currency should have its own line with "_past" or "_future" specified.\n\n'
        "Example:\n"
        '- If the article states, "The EUR is expected to appreciate," the output should be:\n'
        '    EUR_past: "unchanged",\n'
        '    EUR_future: "appreciation"\n'
        '- If the article states, "EUR/USD depreciated last week," the output should be:\n'
        '    EUR_past: "depreciation",\n'
        '    USD_past: "appreciation"\n'
        '- If only future movements are mentioned for a currency, the past movement should be labelled as "unchanged" and vice versa.\n\n'
        "Currency Pair Interpretation:\n"
        "- If currencies are discussed in pairs, interpret as follows:\n"
        '    - If "EUR/USD appreciated," label EUR_past as "appreciation" and USD_past as "depreciation".\n'
        '    - If "EUR/USD depreciated," label EUR_past as "depreciation" and USD_past as "appreciation".\n\n'
        "Synonyms:\n"
        "- Recognize the following synonyms for each currency:\n"
        "- **EUR**: EUR, Euro\n"
        "- **USD**: USD, Dollar, Dollars, US Dollar, US-Dollar, U.S. Dollar, US Dollars, US-Dollars, U.S. Dollars, Greenback\n"
        "- **JPY**: JPY, Yen, Japanese Yen\n"
        "- **GBP**: GBP, Pound, Pounds, Sterling, British Pound, British Pounds\n"
        "- **AUD**: AUD, Australian Dollar, Australian Dollars, Aussie\n"
        "- **CAD**: CAD, Canadian Dollar, Canadian Dollars\n"
        "- **CHF**: CHF, Swiss Franc, Swiss Francs, Swissie\n"
        "- **NZD**: NZD, New Zealand Dollar, New Zealand Dollars, Kiwi\n"
        "- **NOK**: NOK, Norwegian Krone, Norwegian Kroner\n"
        "- **SEK**: SEK, Swedish Krona, Swedish Kronor\n\n"
        "Answer below in the given format:\n"
    )

    messages = [
        {"role": "user", "content": prompt}
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_response(response):
    """Parse a single model response into a sentiment dict."""
    if not response:
        return {}

    sentiment = {}
    for line in response.split('\n'):
        try:
            if line.strip():
                currency, label = line.split(':')
                currency = currency.strip()
                label = label.strip().strip('"').strip("'")
                sentiment[currency] = label
        except ValueError:
            return {}
    return sentiment


# %% [markdown]
# ## 3 Downstream Application - vLLM Accelerated
# Using model trained on pre 2020 data to backtest on 2020-2024 market

# %% [markdown]
# ### 3.1 Load Model with vLLM

# %%
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

# Load tokenizer for prompt formatting
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = '<|reserved_special_token_0|>'
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|reserved_special_token_0|>')
tokenizer.padding_side = "left"

# Load model with vLLM
# BitsAndBytes 4-bit quantization + LoRA adapter at inference time
if USE_LORA:
    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=MAX_SEQ_LENGTH,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        enforce_eager=True,       # Avoids flash_attn/cudagraph issues; remove once flash-attn is rebuilt
    )
    lora_request = LoRARequest("finetuned_adapter", 1, LORA_ADAPTER_PATH)
else:
    llm = LLM(
        model=MODEL_ID,
        max_model_len=MAX_SEQ_LENGTH,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        enforce_eager=True,       # Avoids flash_attn/cudagraph issues; remove once flash-attn is rebuilt
    )
    lora_request = None

print("vLLM model loaded.")

# %% [markdown]
# ### 3.2 Load data

# %%
df_news = pd.read_pickle("_2_llm_paper/cache/df_news.pkl")

df_news = df_news[
    (df_news['Date'] >= pd.to_datetime('2020-01-01')) &
    (df_news['Date'] < pd.to_datetime('2024-07-01'))
]

# Reset index
df_news = df_news.reset_index(drop=True)

# Drop the labels - we will predict these
df_news = df_news[['Title', 'Full Text', 'mentioned_currencies', 'Trading Date']]

df_news

# %% [markdown]
# ### 3.3 Make inferences for all articles using vLLM
# vLLM handles batching internally with continuous batching and PagedAttention.
# We submit ALL prompts at once and let vLLM schedule them optimally.

# %%
print("Generating prompts for all articles...")
prompts = []
for _, row in tqdm(df_news.iterrows(), total=len(df_news), desc="Formatting prompts"):
    prompts.append(generate_prompt(row, tokenizer))

print(f"Generated {len(prompts)} prompts. Starting vLLM inference...")

sampling_params = SamplingParams(
    temperature=0,       # greedy decoding (no randomness), same as do_sample=False
    max_tokens=150,      # ~100 tokens needed for 10 currencies × 2 labels
)

# vLLM processes all prompts with optimal batching automatically
outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

# Parse all responses
sentiment_predictions = []
for output in outputs:
    response = output.outputs[0].text.strip()
    sentiment_predictions.append(parse_response(response))

df_news['sentiment_predictions'] = sentiment_predictions

# Filter out rows with empty sentiment predictions (failed LLM responses)
initial_count = len(df_news)
df_news = df_news[df_news['sentiment_predictions'].apply(lambda x: len(x) > 0)]
df_news = df_news.reset_index(drop=True)
filtered_count = len(df_news)

print(f"Filtered out {initial_count - filtered_count} rows with empty sentiment predictions")
print(f"Remaining articles: {filtered_count}")

# %% [markdown]
# ### 3.4 Daily Sentiment Score Generation
# 
# 
# $$S_{i, t} = round(log(1+CountAppreciation_{i, t}) - log(1+CountDepreciation_{i, t}))$$
# 
# where $CountAppreciation_{i, t}$ is the number of articles published on day $t$ for which the model assigns the **future label** of currency $i$ to "appreciation"
# 
# $round$ function is defined below:
# $$
# \widehat{S}_{i,t} = 
# \begin{cases} 
# +1, & \text{if } S_{i,t} > 0, \\
# 0, & \text{if } S_{i,t} = 0, \\
# -1, & \text{if } S_{i,t} < 0.
# \end{cases}
# $$

# %%
# Initialise dict (currency, date)
data_for_S = {}

# Get list of currency codes
currency_codes = ['USD', 'EUR', 'JPY', 'GBP', 'CAD', 'AUD', 'CHF', 'SEK', 'NOK', 'NZD']

# Group by date
for date, group in df_news.groupby('Trading Date'):
    # For each currency
    for currency in currency_codes:
        # Count appreciation and depreciation for this currency on this date
        count_appreciation = 0
        count_depreciation = 0
        
        for _, row in group.iterrows():
            # Get the future label prediction for this currency
            future_key = f'{currency}_future'
            prediction = row['sentiment_predictions'].get(future_key, 'unchanged')
            
            if prediction == 'appreciation':
                count_appreciation += 1
            elif prediction == 'depreciation':
                count_depreciation += 1
        
        # Calculate S_{i, t} = log(1 + CountAppreciation) - log(1 + CountDepreciation)
        S_value = np.log(1 + count_appreciation) - np.log(1 + count_depreciation)

        # Round to 1 if positive, 0 if 0, -1 if negative
        if S_value > 0:
            S_value = 1
        elif S_value < 0:
            S_value = -1
        else:
            S_value = 0
        
        # Add to dict
        data_for_S[(currency, date)] = S_value


# Get list of dates
dates = sorted(set([date for _, date in data_for_S.keys()]))

# convert to a numpy matrix shape (currencies, dates)
S = np.array([[data_for_S[(currency, date)] for date in dates] for currency in currency_codes])

unique_dates_array = np.sort(pd.to_datetime(df_news['Trading Date'].unique()))
unique_dates_array = pd.to_datetime(unique_dates_array)
unique_dates_array = unique_dates_array.tz_localize(None)  # removes timezone info if present
unique_dates_array = unique_dates_array.normalize()        # sets time to midnight (removes time info)

print(S)
print()
print()
print(unique_dates_array)


# %% [markdown]
# ### 3.5 Get weights
# 
# $$
# w_{i,t:t+1} = 
# \begin{cases} 
# \frac{\widehat{S}_{i,t}}{\sum_{j:\widehat{S}_{j,t}>0} \widehat{S}_{j,t}}, & \text{if } \widehat{S}_{i,t} > 0 \text{ and } \sum_{j:\widehat{S}_{j,t}>0} \widehat{S}_{j,t} > 0 \\
# -\frac{\widehat{S}_{i,t}}{\sum_{j:\widehat{S}_{j,t}<0} |\widehat{S}_{j,t}|}, & \text{if } \widehat{S}_{i,t} < 0 \text{ and } \sum_{j:\widehat{S}_{j,t}<0} |\widehat{S}_{j,t}| > 0 \\
# 0, & \text{otherwise.}
# \end{cases}
# $$
# 
# Long on positive sentiment scores
# 
# Short negative sentiment scores
# 
# Value of each position is proprtional to sentiment score 
# 
# Value of all short positions are the same as all the long positions for any given day
# 
# Held for 1 day (close of day t until close of day t+1)
# 
# if no new article was published on the next day, the previous sentiment signal is retained

# %%
weights = np.zeros_like(S, dtype=float)

positive_mask = S > 0
negative_mask = S < 0

sum_positive = np.sum(S * positive_mask, axis=0, keepdims=True)
sum_negative_abs = np.sum(np.abs(S) * negative_mask, axis=0, keepdims=True)

valid_positive_sums = sum_positive > 0
valid_negative_sums = sum_negative_abs > 0

with np.errstate(divide='ignore', invalid='ignore'):
    positive_weights = np.where(valid_positive_sums, S / sum_positive, 0)
    negative_weights = np.where(valid_negative_sums, S / sum_negative_abs, 0)

weights = np.where(positive_mask & valid_positive_sums, positive_weights, weights)
weights = np.where(negative_mask & valid_negative_sums, negative_weights, weights)

weights

# %% [markdown]
# ### 3.6 Get the market data

# %%
df_log_returns = pd.read_pickle("_2_llm_paper/cache/df_log_returns.pkl")

df_log_returns = df_log_returns[df_log_returns.index.isin(unique_dates_array)]

print(df_log_returns.shape)
print(len(unique_dates_array))

# %% [markdown]
# ### 3.7 Execute strategy

# %%
df_log_returns_T = df_log_returns.transpose()
log_returns = df_log_returns_T.values   # turn it into a matrix

# Remove the last one as doesnt have future returns
W_active = weights[:, :-1]

# Allign for future returns
R_future = log_returns[:, 1:]

# Element wise multiplication
# Sum vertically for each day
daily_pnl = np.sum(W_active * R_future, axis=0)

# %% [markdown]
# ### 3.8 Evaluate results
# 
# Assume a zero cost portiolio so risk free rate is 0 
# 
# $$Annualized\ Return = mean\ daily\ return \times 252$$
# 
# $$Annualized\ Volatility = Standard\ Deviation\ of\ Daily\ Returns \times sqrt(252)$$
# 
# $$Sharpe\ Ratio = \frac{Annualized\ Return}{Annualized\ Volatility}$$
# 
# $$Maximum\ Drawdown =\ The\ largest\ peak\ to\ trough\ decline\ in\ cumulative\ returns$$
# 
# $$Transaction\ Costs = sum\ of\ absolute\ weight\ changes\ per\ day,\ averaged,\ then\ annualized

# %%
ANNUALIZATION_FACTOR = 252    # Number of trading days per year


# Annualized Return
ann_return = np.mean(daily_pnl) * ANNUALIZATION_FACTOR
ann_return_pc = ann_return * 100  # percent

# Annualized Volatility
ann_vol = np.std(daily_pnl, ddof=1) * np.sqrt(ANNUALIZATION_FACTOR)
ann_vol_pc = ann_vol * 100        # Convert to percentage

# Sharpe Ratio
if ann_vol != 0:
    sharpe_ratio = ann_return / ann_vol
else:
    sharpe_ratio = 0.0

# Maximum Drawdown
cumulative_returns = np.cumsum(daily_pnl)
running_max = np.maximum.accumulate(cumulative_returns)
drawdown = running_max - cumulative_returns
max_drawdown = np.max(drawdown)
max_drawdown_pc = max_drawdown * 100  # Convert to percentage

# Transaction Costs (Rebalancing Frequency)
weight_changes = np.abs(np.diff(weights, axis=1))  # Change from day t to t+1
daily_turnover = np.sum(weight_changes, axis=0)    # Total turnover per day
rebalancing_frequency = np.mean(daily_turnover) * ANNUALIZATION_FACTOR

print(f"Annualized Return:    {ann_return_pc:.2f}%")
print(f"Annualized Volatility:{ann_vol_pc:.2f}%")
print(f"Sharpe Ratio:         {sharpe_ratio:.2f}")
print(f"Max Drawdown:         {max_drawdown_pc:.2f}%")
print(f"Rebalancing Freq:     {rebalancing_frequency:.2f}")
