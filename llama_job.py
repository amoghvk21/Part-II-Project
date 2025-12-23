# %% [markdown]
# # Implementing "FX sentiment analysis with large language models" (Ballinari et al.)
# This paper can be found at 

# %% [markdown]
# ## Imports

# %%
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
dailyfx_news = pd.read_csv('datasets/news_articles/dailyfx_articles_012011-062024.csv')
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
fxstreet_news = pd.read_csv('datasets/news_articles/fxstreet_articles.csv')
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
investing_news = pd.read_csv('datasets/news_articles/investingcom_finaldata_2011-062024.csv')
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

df_log_returns.to_pickle("df_log_returns.pkl")

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

df_news.to_pickle("df_news.pkl")

df_news

# %% [markdown]
# ### 1.9 Train Test Split
# - 200 examples for final eval
# - Otherwise 80/20 train test split

# %%
df_news = pd.read_pickle("df_news.pkl")

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
# ## 2 Generate Prompt

# %%
def generate_prompt(row):
    title = row.get('Title', '')
    text = row.get('Full Text', '')
    currencies = row.get('mentioned_currencies')

    target_currencies = ''
    for c in currencies:
        target_currencies += f'{c}_past: "appreciation, depreciation, or unchanged",\n'
        target_currencies += f'{c}_future: "appreciation, depreciation, or unchanged",\n'
    target_currencies = target_currencies.strip().rstrip(",") # Remove last comma

    # Same structure as per paper
    return (
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
        "- **SEK**: SEK, Swedish Krona, Swedish Kronor\n"
        "Answer below in the given format:"
    )

# %% [markdown]
# ## 3 Model setup

# %%
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

# 'meta-llama/Llama-3.1-8B-Instruct' - real
# "meta-llama/Llama-3.2-1B-Instruct" - for local testing as smallest possible model
model_id =  "meta-llama/Llama-3.2-1B-Instruct"

# quntisation config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,      # "Double Quantization"
    bnb_4bit_quant_type="nf4",           # 4-bit NormalFloat data type
    bnb_4bit_compute_dtype=torch.bfloat16 # Compute in bfloat16 for stability
)

# load tokeniser
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.model_max_length = 8192
tokenizer.pad_token = tokenizer.eos_token # Llama has no default pad token
tokenizer.padding_side = "right"  # TODO Check this

# load model 
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto", # Automatically puts model on GPU
    dtype=torch.bfloat16
)

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    model = model.to(device)
    print(f"Model explicitly loaded onto: {device}")
else:
    device = torch.device("cpu")
    model = model.to(device)
    print("CUDA not available. Model loaded onto CPU.")

model.config.use_cache = False
model.config.pretraining_tp = 1

# Prepare for training 
model = prepare_model_for_kbit_training(model)

# LoRA config 
# Params from Table A.1
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",           # TODO Check this
    task_type="CAUSAL_LM", # TODO Check this
    
    # inject low-rank adaptation matrices into all linear layers TODO check this
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", # Attention projections
        "gate_proj", "up_proj", "down_proj"     # MLP projections
    ]
)

# %% [markdown]
# ## 4 LLM Fine Tuning
# - Stopping criterion is used
#     - Optimisises for least loss in the validation stage rather than most traning epochs
#     - So if the model with best validation loss is in epoch 1 or 2, then the weights in epoch 3 will be discarded
#     - Used to prevent overfitting due to this being a small dataset
#     - Stops traning if the validation loss stagnates due to overfitting

# %%
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,      # TODO Check this
    per_device_eval_batch_size=1,       # TODO Check this
    gradient_accumulation_steps=32, 
    optim="paged_adamw_32bit",          # 
    save_steps=50,                      # TODO get better number
    learning_rate=1e-5,                 #  Note: significantly lower than standard
    weight_decay=0.1,                   #  High weight decay
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,                  # TODO apparenly this is the best for lora??? - not said in the paper
    warmup_ratio=0.0,                   # 
    lr_scheduler_type="cosine",         #                  
    save_strategy="steps",              # for early stopping   (could be epoch)
    eval_strategy="steps",              # for early stopping   (could be epoch)
    load_best_model_at_end=True,         # for early stopping
    metric_for_best_model="eval_loss",   # for early stopping
    greater_is_better=False,     # less loss is better
    logging_steps=10,                   # TODO get a better number
    group_by_length=True,
    report_to="none"                    # Disable wandb unless needed
)


df_train = Dataset.from_pandas(df_train)
df_test = Dataset.from_pandas(df_test)

trainer = SFTTrainer(
    model=model,
    train_dataset=df_train, # Ensure this is loaded
    eval_dataset=df_test,
    peft_config=peft_config,
    formatting_func=generate_prompt,
    processing_class=tokenizer,
    args=training_args,
    # packing=False,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)] # to stop after epoch 1 if validaiton loss gets worse
)


trainer.train()

trainer.model.save_pretrained('models_2/llama_finetuned')
print("Model saved.")

# %% [markdown]
# ## 5 Evaluate

# %% [markdown]
# ### 5.1 Predict sentiment
# - Gets the sentiment for a single article
# - Used for evaulation

# %%
def get_sentiment(row, model, tokenizer):
    prompt = generate_prompt(row)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,  # to avoid crashing model due to very large article
        max_length=8192
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,     # only needs to generate enough for sentiment
            temperature=0.1,        # incase there was sampling
            do_sample=False,        # no sampling - so no randomness
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()    # skips over prompt

    # Parse response to get labels into a dict
    sentiment = {}
    for line in response.split('\n'):
        if line.strip():
            currency, label = line.split(':')
            currency = currency.strip()
            label = label.strip()
            sentiment[currency] = label

    return sentiment

# %% [markdown]
# ### 5.2 Get evaulation statistics

# %%
currency_codes = ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD', 'NOK', 'SEK']

all_actual = []
all_predictions = []

for i, row in df_eval.iterrows():
    sentiment = get_sentiment(row, model, tokenizer)
    for c in currency_codes:
        for t in ['past', 'future']:
            all_actual.append(row[f'{c}_{t}_label'])
            all_predictions.append(sentiment.get(f'{c}_{t}', 'unchanged'))

    
    
accuracy = accuracy_score(all_actual, all_predictions)
f1 = f1_score(all_actual, all_predictions, average='macro')
precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(all_actual, all_predictions, labels=['appreciation', 'depreciation', 'unchanged'])

stats = {
    'accuracy': accuracy,
    'f1': f1,
    'precision_per_class': dict(zip(['appreciation', 'depreciation', 'unchanged'], precision_per_class)),
    'recall_per_class': dict(zip(['appreciation', 'depreciation', 'unchanged'], recall_per_class)),
    'f1_per_class': dict(zip(['appreciation', 'depreciation', 'unchanged'], f1_per_class)),
    'support_per_class': dict(zip(['appreciation', 'depreciation', 'unchanged'], support_per_class))
}

report = classification_report(all_actual, all_predictions)

print(stats)

print()
print()

print(report)

# %% [markdown]
# ## 6 Downstream Application
# Using model trained on pre 2020 data to backtest on 2020-2024 market

# %% [markdown]
# ### 6.1 Load data

# %%
import pandas as pd

df_news = pd.read_pickle("df_news.pkl")

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
# ### 6.2 Load LLM

# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# 1. Define paths
base_model_id = "meta-llama/Llama-3.2-8B-Instruct"
adapter_dir = "models_2/llama_finetuned"

# 2. Quantization (Recommended to match your training environment)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 3. Load the Tokenizer (Load from base model, not adapter dir, unless you explicitly saved it there)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# 4. Load the Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 5. Load and attach the Fine-Tuned Adapters
model = PeftModel.from_pretrained(base_model, adapter_dir)

# 6. Set mode for inference
model.eval()

print("Tokenizer, Base Model, and Adapters loaded successfully.")

# %% [markdown]
# ### 6.3 Make inferences for all articles

# %%
print("Getting sentiment predictions for all articles...")

sentiment_predictions = []

for idx, row in tqdm(df_news.iterrows(), total=len(df_news), desc="Processing articles"):
    sentiment = get_sentiment(row, model, tokenizer)
    sentiment_predictions.append(sentiment)
    df_news.at[idx, 'sentiment_predictions'] = sentiment

# %%
# Make fake predictions for testing

# df_news = df_news.head(5000)

import random
random.seed(42)

currency_codes = ['USD', 'EUR', 'JPY', 'GBP', 'CAD', 'AUD', 'CHF', 'SEK', 'NOK', 'NZD']

def generate_fake_sentiment():
    labels = ['appreciation', 'depreciation', 'unchanged']
    # Future and past key naming as in further code
    fake_dict = {}
    for c in currency_codes:
        fake_dict[f"{c}_future"] = random.choice(labels)
        fake_dict[f"{c}_past"] = random.choice(labels)
    return fake_dict

df_news['sentiment_predictions'] = [generate_fake_sentiment() for _ in range(len(df_news))]

df_news

# %% [markdown]
# ### 6.4 Daily Sentiment Score Generation
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
# ### 6.5 Get weights
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
# ### 6.6 Get the market data

# %%
df_log_returns = pd.read_pickle("df_log_returns.pkl")

df_log_returns = df_log_returns[df_log_returns.index.isin(unique_dates_array)]

print(df_log_returns.shape)
print(len(unique_dates_array))

# %% [markdown]
# ### 6.7 Execute strategy

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
# ### 6.8 Evaluate results
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


