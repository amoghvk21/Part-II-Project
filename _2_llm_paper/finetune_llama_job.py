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

    'c': "meta-llama/Meta-Llama-3.1-8B",
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

#MODEL = 'c'
#MODEL_LOAD_DIR = 'finetuned_llama_31_8b'
#MODEL_ID = get_model_id[MODEL]
#MODEL_SAVE_DIR = 'finetuned_llama_31_8b'

MODEL = 'c'
MODEL_LOAD_DIR = 'finetuned_llama_8b'
MODEL_ID = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
MODEL_SAVE_DIR = 'finetuned_llama_8b'


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

# print("Creating train, test and eval")

# df_news = pd.read_pickle("_2_llm_paper/cache/df_news.pkl")

# df_news_before_2020 = df_news[df_news['Date'] < pd.to_datetime('2020-01-01')]     # we train the model on this for now
# # df_news_after_2020 = df_news[df_news['Date'] >= pd.to_datetime('2020-01-01', utc=True)]   # we use this for the trading strategy

# # Randomly sample 30,000 articles
# df_news_before_2020 = df_news_before_2020.sample(n=30000, random_state=42)

# df_rest, df_eval = train_test_split(df_news_before_2020, test_size=200, random_state=42)
# df_train, df_test = train_test_split(df_rest, test_size=0.2, random_state=42)


# # Save df_train, df_test, df_eval
# df_train.to_pickle("_2_llm_paper/cache/df_train.pkl")
# df_test.to_pickle("_2_llm_paper/cache/df_test.pkl")
# df_eval.to_pickle("_2_llm_paper/cache/df_eval.pkl")


# print(f"Size of all news: {len(df_news)}")
# print("Size of train set: ", len(df_train))
# print("Size of test set: ", len(df_test))
# print("Size of eval set: ", len(df_eval))


print("Reading from cache")
df_train = pd.read_pickle("_2_llm_paper/cache/df_train.pkl")
df_test = pd.read_pickle("_2_llm_paper/cache/df_test.pkl")
df_eval = pd.read_pickle("_2_llm_paper/cache/df_eval.pkl")

# 8192 * 3.9 = 31,948.8 ~ 32,000 characters max allowed by the model
# estimate 2000 characters for prompt instructions and meta data
# 32,000 - 2,000 = 30,000 characters max allowed for the article
print("Truncating articles to 30,000 characters")
MAX_ARTICLE_CHARS = 30000
for df in [df_train, df_test, df_eval]:
    df['Full Text'] = df['Full Text'].str[:MAX_ARTICLE_CHARS]

print("Size of train set: ", len(df_train))
print("Size of test set: ", len(df_test))
print("Size of eval set: ", len(df_eval))

# %% [markdown]
# ### Finetuned Llama

# %%
import models_code.finetuned_llama as finetuned_llama


model, tokenizer, peft_config = finetuned_llama.setup(MODEL_ID)
print("setup complete")

# Optional: Print GPU usage info
if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

print("finetuning model")
finetuned_llama.finetune(model, tokenizer, peft_config, df_train, df_test, MODEL_SAVE_DIR)
print("finetune completed")

print("evaluating model")
finetuned_llama.evaluation(model, tokenizer, df_eval)
print("evaluation completed")

# %% [markdown]
# ### Base Llama

# %%
# import models_code.base_llama as base_llama

# print("setup model")
# model, tokenizer = base_llama.setup(MODEL_ID)

# print("evalulate model")
# base_llama.evaluation(model, tokenizer, df_eval)
