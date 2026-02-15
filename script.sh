#! /bin/bash

set -eo pipefail

# Always shut down the instance when the script exits (success or failure)
trap 'sudo shutdown -h now' EXIT

mkdir -p logs

echo "Running ddp finetuned llama job - lr=5e-6"
torchrun --nproc_per_node=4 --master_port=29500 "_2_llm_paper/ddp_finetune_llama_job.py" --learning_rate 5e-6 --save_dir "finetuned_llama_8b_lr5e6" 2>&1 | tee logs/lr5e6.log
echo "--------------------------------"

# echo "Running ddp finetuned llama job - lr=1e-5"  # DONE ALEADY
# torchrun --nproc_per_node=4 --master_port=29501 "_2_llm_paper/ddp_finetune_llama_job.py" --learning_rate 1e-5 --save_dir "finetuned_llama_8b_lr1e5" 2>&1 | tee logs/lr1e5.log
# echo "--------------------------------"

echo "Running ddp finetuned llama job - lr=2e-5"
torchrun --nproc_per_node=4 --master_port=29502 "_2_llm_paper/ddp_finetune_llama_job.py" --learning_rate 2e-5 --save_dir "finetuned_llama_8b_lr2e5" 2>&1 | tee logs/lr2e5.log
echo "--------------------------------"

echo "Running ddp finetuned llama job - lr=5e-5"
torchrun --nproc_per_node=4 --master_port=29503 "_2_llm_paper/ddp_finetune_llama_job.py" --learning_rate 5e-5 --save_dir "finetuned_llama_8b_lr5e5" 2>&1 | tee logs/lr5e5.log
echo "--------------------------------"

echo "done"