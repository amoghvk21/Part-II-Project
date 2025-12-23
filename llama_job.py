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
# ### 1.9 Train Test Split
# - 200 examples for final eval
# - Otherwise 80/20 train test split

# %%
df_news = pd.read_pickle("df_news.pkl")

df_news_before_2020 = df_news[df_news['Date'] < pd.to_datetime('2020-01-01')]     # we train the model on this for now
# df_news_after_2020 = df_news[df_news['Date'] >= pd.to_datetime('2020-01-01')]   # we use this for the trading strategy

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
model_id =  "meta-llama/Llama-3.1-8B-Instruct"

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
base_model_id = "meta-llama/Llama-3.1-8B-Instruct"
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


