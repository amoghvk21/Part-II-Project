#! /bin/bash

set -e

python3.12 -m venv venv 
source venv/bin/activate

echo "Installing dependencies"
pip install -r requirements_linux.txt

echo "Running ddp finetuned llama job"
python3.12 _2_llm_paper/ddp_finetune_llama_job.py

echo "done"