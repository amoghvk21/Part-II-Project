# %% [markdown]
# # Implementing "FX sentiment analysis with large language models" (Ballinari et al.)
# This paper can be found at 

# %% [markdown]
# ## Imports

# %%
# surpressing security warnings from loading models as all models i am loading are my own finetuned ones 
import transformers.utils.import_utils
import transformers.trainer
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.trainer.check_torch_load_is_safe = lambda: None


import numpy as np
import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os
from huggingface_hub import login
from trl import SFTTrainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import import_ipynb

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

# a) llama-3.1-8b-instruct             - used in paper
# b) llama-3.3-70b-instruct            - want to try bigger model

# c) llama-3.1-8b finetuned            - used in paper
# d) llama-3.3-70b finetuned           - want to try bigger model

# e) llama 4 maverick instruct      - 1M context window but more params             moE model can be tested     best in the space right now
# f) llama 4 scout instruct         -  10M context window but smaller params        MoE model can be tested     best in the space right now

# g) mistral-nemo-instruct finetuned       - used in paper and is a great alternative

# h) qwen 3 32b coder instruct finetuned         - good at following logic and rules and won't hallicinate json output, use ChatML

# i) finbert        - used in paper
# j) vader          - used in paper
# k) lm             - used in paper

MODEL = 'a'
MODEL_LOAD_DIR = 'finetuned_llama_8b'
MODEL_ID = get_model_id[MODEL]

# %% [markdown]
# ## 3 Downstream Application - NOT USING DDP
# Using model trained on pre 2020 data to backtest on 2020-2024 market

# %% [markdown]
# ### 3.1 Load Model

# %%
import models_code.finetuned_llama as finetuned_llama

model, tokenizer = finetuned_llama.load(MODEL_ID, MODEL_LOAD_DIR)

# %%
# import models_code.base_llama as base_llama

# model, tokenizer = base_llama.setup(MODEL_ID)

# %% [markdown]
# ### 3.2 Load data

# %%
import pandas as pd

df_news = pd.read_pickle("_2_llm_paper/cache/df_news.pkl")

# TODO check if its inclusive
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
# ### 3.3 Make inferences for all articles

# %%
print("Getting sentiment predictions for all articles...")

sentiment_predictions = []

for idx, row in tqdm(df_news.iterrows(), total=len(df_news), desc="Processing articles"):
    sentiment = finetuned_llama.get_sentiment(row, model, tokenizer)   # returns a dict
    sentiment_predictions.append(sentiment)

df_news['sentiment_predictions'] = sentiment_predictions

# Filter out rows with empty sentiment predictions (failed LLM responses)
initial_count = len(df_news)
df_news = df_news[df_news['sentiment_predictions'].apply(lambda x: len(x) > 0)]
df_news = df_news.reset_index(drop=True)
filtered_count = len(df_news)

print(f"Filtered out {initial_count - filtered_count} rows with empty sentiment predictions")
print(f"Remaining articles: {filtered_count}")

# %% [markdown]
# ### Make FAKE PREDICTIONS for testing

# %%
# # df_news = df_news.head(5000)

# import random
# random.seed(42)

# currency_codes = ['USD', 'EUR', 'JPY', 'GBP', 'CAD', 'AUD', 'CHF', 'SEK', 'NOK', 'NZD']

# def generate_fake_sentiment():
#     labels = ['appreciation', 'depreciation', 'unchanged']
#     # Future and past key naming as in further code
#     fake_dict = {}
#     for c in currency_codes:
#         fake_dict[f"{c}_future"] = random.choice(labels)
#         fake_dict[f"{c}_past"] = random.choice(labels)
#     return fake_dict

# df_news['sentiment_predictions'] = [generate_fake_sentiment() for _ in range(len(df_news))]

# df_news

# %% [markdown]
# ### 3.4 Daily Sentiment Score Generation
# 
# 
# $$S_{i, t} = round(log(1+CountAppreciation_{i, t}) - log(1+CountDepreciation_{i, t}))$$
# 
# where $CountAppreciation_{i, t}$ is the number of articles published on day $t$ for which the model assigns the **future label** of currency $i$ to “appreciation”
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
import numpy as np

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


