#!/bin/bash
#SBATCH --job-name safety
#SBATCH --time 4-00:00:00
#SBATCH -c 4
#SBATCH --mem 40G
#SBATCH --gres=gpu:1



domains=(task-execution rule-following persona safety)



settings=(reference conflict)


hierarchy=(sys-user user-tool sys-tool)


high_langs=(en de hi zh es fr)



low_langs=(en de hi zh es fr)


# (only API models: gpt-5, gpt-5-mini, gpt-5-nano, claude-haiku, claude-sonnet)
# model_list=(gpt-5 gpt-5-mini gpt-5-nano claude-haiku claude-sonnet)

# Run batch processing
python src/model/run_model_batch.py \
    --domains ${domains[@]} \
    --settings ${settings[@]} \
    --hierarchy ${hierarchy[@]} \
    --high_langs ${high_langs[@]} \
    --low_langs ${low_langs[@]} \
    --models ${model_list[@]} \
    --check_interval 60 \
    --batch_dir batch_jobs

echo "Batch jobs completed!"
