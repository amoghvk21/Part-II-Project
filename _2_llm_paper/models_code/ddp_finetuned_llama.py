# %% [markdown]
# # Imports

# %%
import numpy as np
import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os
from huggingface_hub import login
from trl import SFTTrainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import import_ipynb
from transformers.trainer_utils import get_last_checkpoint
from functools import partial

# %% [markdown]
# # Hyperparamers

# %%
MAX_SEQ_LENGTH = 8192


# LoRA/PEFT parameters
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_BIAS = "none"
LORA_TASK_TYPE = "CAUSAL_LM"
LORA_TARGET_MODULES = [   # Injecting into all linear layers as per paper
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention projections
    "gate_proj", "up_proj", "down_proj"      # MLP projections
]


# Llama parameters
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
SAVE_STEPS = 373
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 0.3
LOGGING_STEPS = 10
EVAL_STEPS = 373

# effective batch size = batch size per device * gradient accumulation steps * number of gpus 
# paper used effective batch size of 32

print("new hyperparams")

# %% [markdown]
# # Prompt Generation

# %%
def generate_prompt(row, tokenizer, is_training=True):
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
        {"role": "user", "content": prompt} # 'prompt' is your string from before
    ]
    
    if is_training:
        # Exptected output for currencies mentioned in the article
        expected_output = ""
        for c in currencies:
            past_label = row.get(f'{c}_past_label', 'unchanged')
            future_label = row.get(f'{c}_future_label', 'unchanged')
            
            expected_output += f'{c}_past: "{past_label}"\n'
            expected_output += f'{c}_future: "{future_label}"\n'
            
        messages.append({"role": "assistant", "content": expected_output})
        
        # Apply template to the WHOLE conversation
        return tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# %% [markdown]
# # Finetuned Llama Model Setup

# %%
def setup(model_id):

    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print(f"DDP enabled | world_size={world_size}")

    print(
        f"PID={os.getpid()} "
        f"RANK={os.environ['RANK']} "
        f"LOCAL_RANK={os.environ['LOCAL_RANK']} "
    )

    # quntisation config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # load tokeniser
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = MAX_SEQ_LENGTH
    
    # Force padding token to <|reserved_special_token_0|>
    tokenizer.pad_token = '<|reserved_special_token_0|>'
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|reserved_special_token_0|>')
    if local_rank == 0:  # Only print on main process
        print("Set pad_token to Llama 3 reserved token (<|reserved_special_token_0|>)")

    tokenizer.padding_side = "right"    # Use right for finetuning

    # Load on specific GPU for this process
    device_map = {"": local_rank}  

    print(f"DEBUG: Rank {local_rank} is attempting to load model.")
    print(f"DEBUG: Rank {local_rank} device_map is STRICTLY: {device_map}")    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        dtype=torch.bfloat16
    )

    model = model.to(device)
    print(f"[Rank {local_rank}] Model loaded on device: {next(model.parameters()).device}")

    model.config.use_cache = False

    # Prepare for training 
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)   # do in need gradient checkpointing

    # LoRA config
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        task_type=LORA_TASK_TYPE,
        target_modules=LORA_TARGET_MODULES
    )


    return model, tokenizer, peft_config

# %% [markdown]
# ### 4 LLM Fine Tuning

# %% [markdown]
# - Stopping criterion is used
#     - Optimisises for least loss in the validation stage rather than most traning epochs
#     - So if the model with best validation loss is in epoch 1 or 2, then the weights in epoch 3 will be discarded
#     - Used to prevent overfitting due to this being a small dataset
#     - Stops traning if the validation loss stagnates due to overfitting

# %%
def finetune(model, tokenizer, peft_config, df_train, df_test, save_name):

    tokenizer.padding_side = "right"   # for finetuning

    # DDP environment
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    if local_rank == 0:  # Only print on main process
        print(f"Training with DDP | world_size={world_size}")

    training_args = TrainingArguments(
        output_dir=f"_2_llm_paper/models/{save_name}/checkpoints",
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,      # TODO Check this
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,       # TODO Check this
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, 
        optim="paged_adamw_32bit",          # 
        save_steps=SAVE_STEPS,                      # TODO get better number
        learning_rate=LEARNING_RATE,                 #  Note: significantly lower than standard
        weight_decay=WEIGHT_DECAY,                   #  High weight decay
        fp16=False,
        bf16=True,
        tf32=True,                                     # Use TF32 on Ampere+ GPUs for faster matmuls
        max_grad_norm=MAX_GRAD_NORM,                  # TODO apparenly this is the best for lora??? - not said in the paper
        warmup_ratio=0.0,                   # 
        lr_scheduler_type="cosine",         #                  
        save_strategy="steps",              # for early stopping   (could be epoch)
        eval_strategy="steps",              # for early stopping   (could be epoch)
        load_best_model_at_end=True,         # for early stopping
        metric_for_best_model="eval_loss",   # for early stopping
        greater_is_better=False,     # less loss is better
        logging_steps=LOGGING_STEPS,                   # TODO get a better number
        save_total_limit=3,
        group_by_length=True,
        report_to="none",                    # Disable wandb unless needed
        ddp_find_unused_parameters=False,    # Important for DDP efficiency
        dataloader_pin_memory=True,          # Can help with multi-GPU performance
        dataloader_num_workers=8,
        torch_compile=True,                  # Compile model for optimized execution
        eval_steps=EVAL_STEPS,
    )


    df_train = Dataset.from_pandas(df_train)
    df_test = Dataset.from_pandas(df_test)

    # Create partial function with tokenizer bound
    formatting_func = partial(generate_prompt, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        train_dataset=df_train, # Ensure this is loaded
        eval_dataset=df_test,
        peft_config=peft_config,
        formatting_func=formatting_func,
        processing_class=tokenizer,
        args=training_args,
        # packing=False,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)] # to stop after epoch 1 if validaiton loss gets worse
    )

    last_checkpoint = get_last_checkpoint(training_args.output_dir)

    if last_checkpoint is not None:
        if local_rank == 0:  # Only print on main process
            print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        if local_rank == 0:  # Only print on main process
            print("No checkpoint found. Starting training from scratch...")
        trainer.train()

    # Only save on main process (rank 0) to avoid conflicts
    if local_rank == 0:
        trainer.model.save_pretrained(f"_2_llm_paper/models/{save_name}/model")
        print("Model saved.")
    else:
        print(f"Skipping save on non-main process (rank {local_rank})")

# %% [markdown]
# # Evaulation

# %% [markdown]
# ## 5.1 Predict sentiment
# - Gets the sentiment for a single article
# - Used for evaulation

# %%
def get_sentiment(row, model, tokenizer):

    tokenizer.padding_side = "left"   # for inference

    prompt = generate_prompt(row, tokenizer, is_training=False)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,  # to avoid crashing model due to very large article
        max_length=8192
    )
    
    # For multi-GPU models, get device from first parameter
    # The model will automatically handle device placement
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,     # only needs to generate enough for sentiment
            temperature=0.1,        # incase there was sampling
            do_sample=False,        # no sampling - so no randomness
            pad_token_id=tokenizer.pad_token_id   # Use same padding token as training
        )
    
    input_len = inputs['input_ids'].shape[1] # le of input tokens
    response_tokens = outputs[0][input_len:] # remove to get only the response
    response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

    # Validate response is not empty
    if not response:
        print("")
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
        sentiment = get_sentiment(row, model=model, tokenizer=tokenizer)
        
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
    
    print(stats)

    print()
    print()

    print(report)

# %% [markdown]
# # Loading Model for Downstream Application

# %%
def load(base_model_id, adapter_dir):
    # Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    tokenizer.pad_token = '<|reserved_special_token_0|>'
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|reserved_special_token_0|>')
    print("Set pad_token to Llama 3 reserved token (<|reserved_special_token_0|>)")

    tokenizer.padding_side = "left"    # for inference

    # Load the Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16
    )

    # Load and attach the Fine-Tuned Adapters
    model = PeftModel.from_pretrained(base_model, f"_2_llm_paper/models/{adapter_dir}/model")

    # Set mode for inference
    model.eval()

    print("Tokenizer, Base Model, and Adapters loaded successfully.")

    return model, tokenizer