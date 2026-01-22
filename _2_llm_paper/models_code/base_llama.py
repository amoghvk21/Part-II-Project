# %% [markdown]
# # Imports

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
import import_ipynb

# %% [markdown]
# # Prompt Generation

# %%
def generate_prompt(row, is_training=False, instruct_model=False):
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
    
    if is_training:
        # Exptected output for currencies mentioned in the article
        expected_output = ""
        for c in currencies:
            past_label = row.get(f'{c}_past_label', 'unchanged')
            future_label = row.get(f'{c}_future_label', 'unchanged')
            
            expected_output += f'{c}_past: "{past_label}"\n'
            expected_output += f'{c}_future: "{future_label}"\n'
        
        return prompt + expected_output
    else:
        return prompt

# %% [markdown]
# # Llama LLM Setup

# %%
def setup(model_id):

    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))
    
    # Setup Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = 8192
    
    if tokenizer.pad_token is None:
        # Check for Llama 3 specific reserved token first
        if '<|reserved_special_token_0|>' in tokenizer.get_vocab():
            tokenizer.pad_token = '<|reserved_special_token_0|>'
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|reserved_special_token_0|>')
            print("Set pad_token to Llama 3 reserved token (<|reserved_special_token_0|>)")
        else:
            raise Exception("Can't find padding token")

    else:
        print(f"Padding token is already set to: {tokenizer.pad_token}")
    
    tokenizer.padding_side = "left"   # for inference

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
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

    model.config.use_cache = True
    model.eval()

    print(f"Base model '{model_id}' loaded successfully (No Fine-Tuning).")

    return model, tokenizer

# %% [markdown]
# # Load

# %%


# %% [markdown]
# # Evaulation

# %% [markdown]
# ## 5.1 Predict sentiment
# - Gets the sentiment for a single article
# - Used for evaulation

# %%
def get_sentiment(row, model, tokenizer):

    tokenizer.padding_side = "left"   # for inference
    
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

    # Validate response is not empty
    if not response:
        return {}

    # Parse response to get labels into a dict
    sentiment = {}
    for line in response.split('\n'):
        try:
            if line.strip():
                currency, label = line.split(':')
                currency = currency.strip()
                label = label.strip()
                sentiment[currency] = label
        except ValueError:
            print(f"Error in response: {response} on line: {line}")
            return {}

    return sentiment

# %% [markdown]
# ## 5.2 Get evaulation statistics

# %%
def evaluation(model, tokenizer, df_eval):
    currency_codes = ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD', 'NOK', 'SEK']

    all_actual = []
    all_predictions = []

    tokenizer.padding_side = "left"   # for inference

    skipped_rows = 0
    for i, row in df_eval.iterrows():
        sentiment = get_sentiment(row, model, tokenizer)
        
        # Skip this row if LLM response was invalid
        if sentiment == {}:
            skipped_rows += 1
            print(f"Skipping row {i} due to invalid LLM response format")
            continue
            
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

    if skipped_rows > 0:
        print(f"\nWarning: Skipped {skipped_rows} row(s) out of {len(df_eval)} total due to invalid LLM response")
    
    print()
    print()

    print(stats)

    print()
    print()

    print(report)


