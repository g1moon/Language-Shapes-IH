#!/bin/bash
#SBATCH --job-name persona-judge
#SBATCH --time 4-00:00:00
#SBATCH -c 4
#SBATCH --mem 100G

# Judge model configuration
JUDGE_MODEL=gpt-5-mini-2025-08-07
BATCH_DIR=batch_jobs_judge

# Change this to modify which models to evaluate
# model_list=(llama3.1-8b llama3.1-70b llama3.2-3b qwen3-4b qwen3-30b ministral3-8b ministral3-14b mistral-small-24b gpt-5 gpt-5-mini gpt-5-nano claude-haiku claude-sonnet)
model_list=(llama3.1-70b)

settings=(reference conflict)
hierarchy=(sys-user user-tool sys-tool)

high_langs=(en de hi zh es fr)
low_langs=(en de hi zh es fr)

# high_langs=(en)
# low_langs=(en)


python src/model/run_persona_judge_batch.py \
    --judge_model ${JUDGE_MODEL} \
    --models ${model_list[@]} \
    --settings ${settings[@]} \
    --hierarchy ${hierarchy[@]} \
    --high_langs ${high_langs[@]} \
    --low_langs ${low_langs[@]} \
    --check_interval 60 \
    --batch_dir ${BATCH_DIR}

echo -e "\n\e[32m=============== All persona judge tasks completed ===============\e[0m"
