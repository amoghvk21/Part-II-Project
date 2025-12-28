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

# %% [markdown]
# ## Hyperparameters

# %%
get_model_id = {
    'a': "meta-llama/Meta-Llama-3-8B-Instruct",
    'b': "meta-llama/Meta-Llama-3-70B-Instruct",

    'c': "meta-llama/Meta-Llama-3-8B",
    'd': "meta-llama/Meta-Llama-3-70B",

    'e': "meta-llama/Meta-Llama-4-Maverick-8B-Instruct",
    'f': "meta-llama/Meta-Llama-4-Scout-8B-Instruct",

    'g': "mistralai/Mistral-Nemo-Instruct",

    'h': "Qwen/Qwen3-32B-Chat",
}

# %%
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

MODEL = 'c'
MODEL_LOAD_DIR = ''
MODEL_ID = get_model_id[MODEL]
MODEL_SAVE_DIR = ''

# %% [markdown]
# ## 1. Dataset Preprocessing

# %% [markdown]
# ### 1.1. Filtering
# - Load the datasets
# - Drop articles with <20 words
# - Remove duplicate articles 
# - Convert time to datetime

# %%
def filter_df(df):
    df = df[['Title', 'Date', 'Full Text', 'URL']]

    # Drop articles with <20 words
    df = df[df['Full Text'].str.split().str.len() >= 20]

    # Remove duplicate articles
    df = df.drop_duplicates(subset=['Full Text'])
    df = df.drop_duplicates(subset=['Title'])
    df = df.drop_duplicates(subset=['URL'])

    # Truncate articles to max 32,767 characters
    df['Full Text'] = df['Full Text'].str[:32767]

    return df
    

# %%
# Loading the dailyfx news articles dataset (Title,Author,Date,Full Text,URL)
dailyfx_news = pd.read_csv('_2_llm_paper/datasets/news_articles/dailyfx_articles_012011-062024.csv')
dailyfx_news = filter_df(dailyfx_news)

# Convert time to datetime
dailyfx_news['Date'] = pd.to_datetime(dailyfx_news['Date'], utc=True)

dailyfx_news['Date'] = (
    dailyfx_news['Date']                        
    .dt.tz_convert('America/New_York')     # Convert to NY time like paper
    .dt.tz_localize(None)                  # Remove timezone info to allow for merging later
    .dt.normalize()                        # Normalise to midnight
)

# Sort by date
dailyfx_news = dailyfx_news.sort_values(by='Date')

print(f"Loaded {len(dailyfx_news)} articles")

# %%
# Load the fxstreet news articles dataset into same df (Title,Date,Full Text,URL)
fxstreet_news = pd.read_csv('_2_llm_paper/datasets/news_articles/fxstreet_articles.csv')
fxstreet_news = filter_df(fxstreet_news)

# Convert time to datetime
fxstreet_news['Date'] = pd.to_datetime(fxstreet_news['Date'], utc=True)

fxstreet_news['Date'] = (
    fxstreet_news['Date']                        
    .dt.tz_convert('America/New_York')     # Convert to NY time like paper
    .dt.tz_localize(None)                  # Remove timezone info to allow for merging later
    .dt.normalize()                        # Normalise to midnight
)

# Sort by date
fxstreet_news = fxstreet_news.sort_values(by='Date')

print(f"Loaded {len(fxstreet_news)} articles")

# %%
# Load investing.com news articles dataset into same df (Title,Full Text,URL,Date,Author)
investing_news = pd.read_csv('_2_llm_paper/datasets/news_articles/investingcom_finaldata_2011-062024.csv')
investing_news = filter_df(investing_news)

# Remove "Published" from date column
investing_news['Date'] = investing_news['Date'].str.split(' ', n=1).str[-1].str.strip()

# Convert the 'Date' column in investing_news to datetime
investing_news['Date'] = pd.to_datetime(investing_news['Date'], format='%m/%d/%Y, %I:%M %p')

investing_news['Date'] = (
    investing_news['Date']
    .dt.tz_localize(None)    # Remove timezone info
    .dt.normalize()          # Noramlise to midnight
)

# Sort by date
investing_news = investing_news.sort_values(by='Date')

print(f"Loaded {len(investing_news)} articles")

# %%
# Combine all datasets together
df_news = pd.concat(
    [
        dailyfx_news[['Title', 'Date', 'Full Text', 'URL']],
        fxstreet_news[['Title', 'Date', 'Full Text', 'URL']],
        investing_news[['Title', 'Date', 'Full Text', 'URL']]
    ],
    ignore_index=True
)

df_news = df_news.reset_index(drop=True)

print(f"Total size: {len(df_news)}")

df_news

# %% [markdown]
# ### 1.2. Creating mentioned_currency column
# - Use regex to capture all the currencies used in an article
# - Make use of common synomyms
# - Filter articles that don't mention any of the G10 currencies 

# %%
# Dictionary mapping ISO codes to the regex patterns (synonyms) from Figure A.1
currency_synonyms = {
    "EUR": [r"EUR", r"Euro"],
    "USD": [r"USD", r"Dollar", r"Dollars", r"US Dollar", r"US-Dollar", r"U\.S\. Dollar", 
            r"US Dollars", r"US-Dollars", r"U\.S\. Dollars", r"Greenback"],
    "JPY": [r"JPY", r"Yen", r"Japanese Yen"],
    "GBP": [r"GBP", r"Pound", r"Pounds", r"Sterling", r"British Pound", r"British Pounds"],
    "AUD": [r"AUD", r"Australian Dollar", r"Australian Dollars", r"Aussie"],
    "CAD": [r"CAD", r"Canadian Dollar", r"Canadian Dollars"],
    "CHF": [r"CHF", r"Swiss Franc", r"Swiss Francs", r"Swissie"],
    "NZD": [r"NZD", r"New Zealand Dollar", r"New Zealand Dollars", r"Kiwi"],
    "NOK": [r"NOK", r"Norwegian Krone", r"Norwegian Kroner"],
    "SEK": [r"SEK", r"Swedish Krona", r"Swedish Kronor"]
}

# Get list of mentioned currencies from text
def get_mentioned_currencies(text):
    mentioned_currencies = list()

    for currency, patterns in currency_synonyms.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                mentioned_currencies.append(currency)
                break

    return mentioned_currencies

df_news['mentioned_currencies'] = df_news['Full Text'].apply(get_mentioned_currencies)

# Filter articles to keep only those where 'mentioned_currencies' is non empty
df_news = df_news[df_news['mentioned_currencies'].apply(lambda x: len(x) > 0)]

# So that it is incrimenting by 1 properly due to dropped values from before
df_news = df_news.reset_index(drop=True)

print("mentioned currencies column created")

# %% [markdown]
# ### 1.3. Getting historical prices
# Using nominal narrow effective exchange rate (daily) for each country.
# 
# Narrow effective exchange rate is a good proxy for the tradable currency index that the authors used.
# 
# Allows us to put a number to the currency rather than using a pair as then the currency can be effected by the other in the pair. 
# 
# It is done by taking the geometric mean from the exchange rate of various other currencies (narrow means only a small number of industrialised countries so that the average isn't skewed by some other non industrialised country going down).

# %%
# All links to get data from for effective exchage rate
urls = {
    "USD": "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_EER/1.0/D.N.N.US?startPeriod=2011-01-01&endPeriod=2024-06-01&format=csv",
    "EUR": "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_EER/1.0/D.N.N.XM?startPeriod=2011-01-01&endPeriod=2024-06-01&format=csv",
    "JPY": "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_EER/1.0/D.N.N.JP?startPeriod=2011-01-01&endPeriod=2024-06-01&format=csv",
    "GBP": "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_EER/1.0/D.N.N.GB?startPeriod=2011-01-01&endPeriod=2024-06-01&format=csv",
    "CAD": "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_EER/1.0/D.N.N.CA?startPeriod=2011-01-01&endPeriod=2024-06-01&format=csv",
    "AUD": "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_EER/1.0/D.N.N.AU?startPeriod=2011-01-01&endPeriod=2024-06-01&format=csv",
    "CHF": "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_EER/1.0/D.N.N.CH?startPeriod=2011-01-01&endPeriod=2024-06-01&format=csv", 
    "SEK": "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_EER/1.0/D.N.N.SE?startPeriod=2011-01-01&endPeriod=2024-06-01&format=csv",
    "NOK": "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_EER/1.0/D.N.N.NO?startPeriod=2011-01-01&endPeriod=2024-06-01&format=csv",
    "NZD": "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_EER/1.0/D.N.N.NZ?startPeriod=2011-01-01&endPeriod=2024-06-01&format=csv"
}

# Initialise an empty DataFrame (EER = effective exchange rate)
df_EER = pd.DataFrame()

for code, url in urls.items():
    # Read only the required columns from the CSV
    df_temp = pd.read_csv(url, usecols=lambda c: c in ["TIME_PERIOD", "OBS_VALUE"])
    
    # Convert OBS_VALUE to float for log calculations later
    df_temp["OBS_VALUE"] = pd.to_numeric(df_temp["OBS_VALUE"], errors="coerce")
    
    # Rename "OBS_VALUE" to currency code
    df_temp = df_temp.rename(columns={
        "OBS_VALUE": code,
        "TIME_PERIOD": "date"
    })
    
    # If the main df is empty, set it to this df
    if df_EER.empty:
        df_EER = df_temp
    else:
        # Join on "date", keep all records (outer join)
        df_EER = pd.merge(df_EER, df_temp, on=["date"], how='outer')


df_EER['date'] = pd.to_datetime(df_EER['date'])
df_EER = df_EER.set_index('date')

# drop all NaNs in the data
df_EER.dropna(inplace=True)

df_EER.head()

# %% [markdown]
# ### 1.4 Calculate log returns

# %%
# Calculate daily log returns
df_log_returns = np.log(df_EER / df_EER.shift(1))

df_log_returns.dropna(inplace=True)  # created in the shifting

df_log_returns.to_pickle("_2_llm_paper/cache/df_log_returns.pkl")

# %% [markdown]
# ### 1.5 Calculate cumulative 5 day windows

# %%
# Future returns
# At index t, we want the sum of t+1, t+2, t+3, t+4, t+5 returns
df_future_returns = df_log_returns.rolling(window=5, min_periods=5).sum().shift(-5)
df_future_returns.dropna(inplace=True)

# Past returns
# At index t, we want the sum of t-1, t-2, t-3, t-4, t-5 returns
df_past_returns = df_log_returns.rolling(window=5, min_periods=5).sum().shift(1)
df_past_returns.dropna(inplace=True)


# Merge future and past returns DataFrames into df_log_returns, aligning on date index.
df_log_returns = df_log_returns.join(df_future_returns.add_suffix('_future'), how='inner')
df_log_returns = df_log_returns.join(df_past_returns.add_suffix('_past'), how='inner')

# %% [markdown]
# ### 1.6 Get sentiment labels
# 
# Based of future returns:
# 
# For each timestep:
# - Top 3 (30%) -> "appreciation"
# - Middle 4 (40%) -> "unchanged"
# - Bottom 3 (30%) -> "depreciation"

# %%
# Get list of currency codes (G10 currencies)
currency_codes = ['USD', 'EUR', 'JPY', 'GBP', 'CAD', 'AUD', 'CHF', 'SEK', 'NOK', 'NZD']

# Initialize label columns for each currency
for currency in currency_codes:
    df_log_returns[f'{currency}_label'] = None

# For each date (row), rank currencies by their future returns and assign labels
for date in df_log_returns.index:
    # Get future returns for this date
    future_returns = {}
    for currency in currency_codes:
        value = df_log_returns.loc[date, f'{currency}_future']
        if pd.notna(value):
            future_returns[currency] = value

    # Get past returns for this date
    past_returns = {}
    for currency in currency_codes:
        value = df_log_returns.loc[date, f'{currency}_past']
        if pd.notna(value):
            past_returns[currency] = value
    
    # Rank currencies by future returns (highest to lowest)
    sorted_currencies_future = sorted(future_returns.items(), key=lambda x: x[1], reverse=True)

    # Rank currencies by past returns (highest to lowest)
    sorted_currencies_past = sorted(past_returns.items(), key=lambda x: x[1], reverse=True)
    
    # Assign labels based on ranking
    # Top 3 (30%) -> "appreciation"
    # Middle 4 (40%) -> "unchanged"
    # Bottom 3 (30%) -> "depreciation"
    for i, (currency, _) in enumerate(sorted_currencies_future):
        if i < 3:  # Top 3 (0, 1, 2)
            df_log_returns.loc[date, f'{currency}_future_label'] = 'appreciation'
        elif i >= 7:  # Bottom 3 (7, 8, 9)
            df_log_returns.loc[date, f'{currency}_future_label'] = 'depreciation'
        else:  # Middle 4 (3, 4, 5, 6)
            df_log_returns.loc[date, f'{currency}_future_label'] = 'unchanged'
    
    for i, (currency, _) in enumerate(sorted_currencies_past):
        if i < 3:  # Top 3 (0, 1, 2)
            df_log_returns.loc[date, f'{currency}_past_label'] = 'appreciation'
        elif i >= 7:  # Bottom 3 (7, 8, 9)
            df_log_returns.loc[date, f'{currency}_past_label'] = 'depreciation'
        else:  # Middle 4 (3, 4, 5, 6)
            df_log_returns.loc[date, f'{currency}_past_label'] = 'unchanged'

# Only keep labels
df_labels = df_log_returns[
    [f'{currency}_future_label' for currency in currency_codes] + 
    [f'{currency}_past_label' for currency in currency_codes] 
]


# %%
# Get rid of time info
df_labels.index = df_labels.index.normalize()

# Have date as a column
df_labels = df_labels.reset_index()

df_labels.columns

# %% [markdown]
# ### 1.7 Assign labels to news articles
# - df_labels - labels for sentiment for each valid day
# - df_news_future and past - mapping between all real dates and valid dates based on the way we backfill
# - df_news - now will contain these new sentiment labels joined using the dates from the news_future/past

# %%
# maps news date to future trading date
df_news_future = pd.merge_asof(
    df_news[['Date']],
    df_labels[['date']],
    left_on='Date',
    right_on='date',
    direction='forward'
).rename(columns={'Date': 'news_date', 'date': 'trading_date_future'})


# maps news date to past trading date
df_news_past = pd.merge_asof(
    df_news[['Date']],
    df_labels[['date']],
    left_on='Date',
    right_on='date',
    direction='backward'
).rename(columns={'Date': 'news_date', 'date': 'trading_date_past'})

# Perform a concat of df_news and df_news_future and df_news_past
# Example: Only concat 'news_date' from df_news_future, and 'trading_date_past' from df_news_past
df_news = pd.concat([
    df_news,
    df_news_future[['trading_date_future']],
    df_news_past[['trading_date_past']]
], axis=1)


# Merge the future labels into the news dataframe
future_cols = ['date'] + [col for col in df_labels.columns if col.endswith('_future_label')]
df_labels_future = df_labels[future_cols]

df_news = df_news.merge(df_labels_future, left_on='trading_date_future', right_on='date', how='left')
df_news = df_news.drop(columns=['date'])  # Drop the date column as we don't need it

# Merge the past labels into the news dataframe
past_cols = ['date'] + [col for col in df_labels.columns if col.endswith('_past_label')]
df_labels_past = df_labels[past_cols]

df_news = df_news.merge(df_labels_past, left_on='trading_date_past', right_on='date', how='left')
df_news = df_news.drop(columns=['date'])  # Drop the date column as we already have trading_date

# %% [markdown]
# ### 1.8 Cleaning final DataFrame
# - drop rows with nulls
# - Removing unnecessary columns

# %%
df_news = df_news.rename(columns={'trading_date_future': 'Trading Date'})  # Rename trading_date_future to trading_date

# Only keep title, full text and all labels
df_news = df_news[['Title', 'Date', 'Full Text', 'mentioned_currencies', 'Trading Date', *future_cols[1:], *past_cols[1:]]]

df_news = df_news.dropna()

df_news.to_pickle("_2_llm_paper/cache/df_news.pkl")

print("created df_news")

df_news

# %% [markdown]
# ## 2 Loading, Training and Evaluation of Model
# 
# - meta-llama/Llama-3.1-8B-Instruct
# - meta-llama/Llama-3.1-8B
# - meta-llama/Llama-3.3-70B

# %% [markdown]
# ### Train Test Split
# - 200 examples for final eval
# - Otherwise 80/20 train test split

# %%
df_news = pd.read_pickle("_2_llm_paper/cache/df_news.pkl")

df_news_before_2020 = df_news[df_news['Date'] < pd.to_datetime('2020-01-01', utc=True)]     # we train the model on this for now
# df_news_after_2020 = df_news[df_news['Date'] >= pd.to_datetime('2020-01-01', utc=True)]   # we use this for the trading strategy

# Randomly sample 30,000 articles
df_news_before_2020 = df_news_before_2020.sample(n=30000, random_state=42)

df_rest, df_eval = train_test_split(df_news_before_2020, test_size=200, random_state=42)
df_train, df_test = train_test_split(df_rest, test_size=0.2, random_state=42)


print(f"Size of all news: {len(df_news)}")
print("Size of train set: ", len(df_train))
print("Size of test set: ", len(df_test))
print("Size of eval set: ", len(df_eval))

# %% [markdown]
# ### Finetuned Llama

# %%
# import _2_llm_paper.models_code.finetuned_llama as finetuned_llama

# model, tokenizer, peft_config = finetuned_llama.setup(MODEL_ID)
# finetuned_llama.finetune(model, tokenizer, peft_config, df_train_, df_test, MODEL_SAVE_DIR)
# finetuned_llama.evaulation(model, tokenizer, df_eval)

# %% [markdown]
# ### Base Llama

# %%
import _2_llm_paper.models_code.finetuned_llama as base_llama

print("setup model")
model, tokenizer = base_llama.setup(MODEL_ID)

print("evalulate model")
base_llama.evaluation(model, tokenizer, df_eval)

# %% [markdown]
# ## 3 Downstream Application
# Using model trained on pre 2020 data to backtest on 2020-2024 market

# %% [markdown]
# ### 3.1 Load Model

# %%
# import _2_llm_paper.models_code.finetuned_llama as finetuned_llama

# finetuned_llama.load(MODEL_ID, MODEL_LOAD_DIR)

# %%
import _2_llm_paper.models_code.base_llama as base_llama

model, tokenizer = base_llama.setup(MODEL_ID)

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
    sentiment = base_llama.get_sentiment(row, model, tokenizer)
    sentiment_predictions.append(sentiment)
    df_news.at[idx, 'sentiment_predictions'] = sentiment

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


