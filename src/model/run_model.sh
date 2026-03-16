#!/bin/bash
#SBATCH --job-name test
#SBATCH --time 4-00:00:00
#SBATCH -c 8
#SBATCH --mem 200G
#SBATCH --gres=gpu:2


if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    TP_SIZE=$SLURM_GPUS_ON_NODE
else

    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        TP_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    else
        TP_SIZE=1
    fi
fi

echo -e "\e[33mDetected GPU Count: ${TP_SIZE}. Setting tensor_parallel to ${TP_SIZE}.\e[0m"
# =================================================================


# Change this to specify which domain to run
domains=(rule-following safety task-execution persona)  # run all domains





# Change this to specify which settings to run
# settings=(reference conflict)  # run all settings

# Change this to specify which hierarchy to run
hierarchy=(sys-user user-tool sys-tool)  # run all hierarchies

# Change this to specify which high-level languages to run
high_langs=(en es de hi zh fr)  # run all high-level languages


# Change this to specify which low-level languages to run
low_langs=(en es de hi zh fr)  # run all low-level languages


# Change this to modify which model to run
# model_list=(llama3.1-8b llama3.1-70b llama3.2-3b qwen3-4b qwen3-30b ministral3-8b ministral3-14b mistral-small-24b gpt-5 gpt-5-mini gpt-5-nano claude-haiku claude-sonnet)



# vLLM server configuration
VLLM_PORT=8000  # Edit this to the port of the vLLM server
VLLM_SERVER_STARTED=0
CURRENT_VLLM_MODEL=""

# Domain to run script mapping
declare -A domain_run_script
domain_run_script["rule-following"]="run_rule_following.py"
domain_run_script["safety"]="run_safety.py"
domain_run_script["task-execution"]="run_task_execution.py"
domain_run_script["persona"]="run_persona.py"

# Function to start vLLM server
start_vllm_server() {
    local model_path=$1
    local model_family=$2
    local tensor_parallel=$3
    local max_model_len=$4
    
    echo -e "\n\e[32m=============== Starting vLLM Server ===============\e[0m"
    echo "Model: ${model_path}"
    echo "Family: ${model_family}"
    echo "Port: ${VLLM_PORT}"
    echo -e "\e[32m===================================================\e[0m\n"
    
    python src/utils/vllm_server_manager.py start \
        -model ${model_path} \
        -model_family ${model_family} \
        -port ${VLLM_PORT} \
        -tensor_parallel ${tensor_parallel} \
        -max_model_len ${max_model_len} \
        -gpu_memory_utilization 0.90
    
    if [ $? -eq 0 ]; then
        VLLM_SERVER_STARTED=1
        CURRENT_VLLM_MODEL=${model_path}
        echo -e "\e[32mvLLM server started successfully\e[0m"
        return 0
    else
        echo -e "\e[31mFailed to start vLLM server\e[0m"
        return 1
    fi
}

# Function to stop vLLM server
stop_vllm_server() {
    if [ ${VLLM_SERVER_STARTED} -eq 1 ]; then
        echo -e "\n\e[32m=============== Stopping vLLM Server ===============\e[0m"
        python src/utils/vllm_server_manager.py stop -port ${VLLM_PORT}
        VLLM_SERVER_STARTED=0
        CURRENT_VLLM_MODEL=""
        echo -e "\e[32m====================================================\e[0m\n"
    fi
}

# Trap to ensure server is stopped on exit
trap stop_vllm_server EXIT INT TERM

for domain in ${domains[@]}; do
    # Check if domain is valid
    if [[ ! -v domain_run_script[$domain] ]]; then
        echo "Error: Unknown domain: $domain"
        echo "Valid domains are: rule-following, safety, task-execution, tool-use"
        exit 1
    fi

    run_script=${domain_run_script[$domain]}


    for model in ${model_list[@]}; do
        
        if [ "$model" = "llama3.1-8b" ]; then
            model_family=llama
            model_path=meta-llama/Llama-3.1-8B-Instruct
            script_type=vllm
        
        elif [ "$model" = "llama3.1-70b" ]; then
            model_family=llama
            model_path=meta-llama/Llama-3.1-70B-Instruct
            script_type=vllm
        
        elif [ "$model" = "llama3.2-3b" ]; then
            model_family=llama
            model_path=meta-llama/Llama-3.2-3B-Instruct
            script_type=vllm

        elif [ "$model" = "qwen3-4b" ]; then
            model_family=qwen
            model_path=Qwen/Qwen3-4B-Instruct-2507
            script_type=vllm
        elif [ "$model" = "qwen3-30b" ]; then
            model_family=qwen
            model_path=Qwen/Qwen3-30B-A3B-Instruct-2507
            script_type=vllm
        elif [ "$model" = "ministral3-8b" ]; then
            model_family=mistral
            model_path=mistralai/Ministral-3-8B-Instruct-2512
            script_type=vllm
        elif [ "$model" = "ministral3-14b" ]; then
            model_family=mistral
            model_path=mistralai/Ministral-3-14B-Instruct-2512
            script_type=vllm
        elif [ "$model" = "mistral-small-24b" ]; then
            model_family=mistral
            model_path=mistralai/Mistral-Small-3.2-24B-Instruct-2506
            script_type=vllm

        elif [ "$model" = "gpt-5" ]; then
            model_family=gpt
            model_path=gpt-5-2025-08-07
            script_type=api
        elif [ "$model" = "gpt-5-mini" ]; then
            model_family=gpt
            model_path=gpt-5-mini-2025-08-07
            script_type=api
        elif [ "$model" = "gpt-5-nano" ]; then
            model_family=gpt
            model_path=gpt-5-nano-2025-08-07
            script_type=api

        elif [ "$model" = "claude-haiku" ]; then
            model_family=claude
            model_path=claude-haiku-4-5-20251001
            script_type=api
        elif [ "$model" = "claude-sonnet" ]; then
            model_family=claude
            model_path=claude-sonnet-4-5-20250929
            script_type=api
        else
            echo "Model not found"
            exit 1
        fi

        # Start vLLM server if needed (for vllm backend)
        if [ "$script_type" = "vllm" ]; then
            # Check if we need to start a new server (different model or not started)
            if [ "${CURRENT_VLLM_MODEL}" != "${model_path}" ]; then
                # Stop existing server if running
                stop_vllm_server
                
                # Start new server
                start_vllm_server ${model_path} ${model_family} ${TP_SIZE} 4096
                if [ $? -ne 0 ]; then
                    echo "Failed to start vLLM server, skipping model ${model}"
                    continue
                fi
                
                # Wait a bit for server to be fully ready
                sleep 5
            fi
        fi

        # Iterate over settings, hierarchies, and language combinations
        for setting in ${settings[@]}; do
            for hier in ${hierarchy[@]}; do
                for high_lang in ${high_langs[@]}; do
                    for low_lang in ${low_langs[@]}; do
                        
                        # Construct input file path based on hierarchy type
                        if [ "$hier" = "sys-user" ]; then
                            input_file="benchmark/${domain}/${setting}/${hier}/input-sys_${high_lang}-user_${low_lang}.json"
                        elif [ "$hier" = "sys-tool" ]; then
                            input_file="benchmark/${domain}/${setting}/${hier}/input-sys_${high_lang}-tool_${low_lang}.json"
                        elif [ "$hier" = "user-tool" ]; then
                            input_file="benchmark/${domain}/${setting}/${hier}/input-user_${high_lang}-tool_${low_lang}.json"
                        else
                            echo "Error: Unknown hierarchy: $hier"
                            echo "Valid hierarchies are: sys-user, user-tool, sys-tool"
                            continue
                        fi

                        # Check if input file exists
                        if [ ! -f "$input_file" ]; then
                            echo "Warning: Input file not found: $input_file, skipping..."
                            continue
                        fi

                        # Construct output directory path
                        # Extract filename without extension for output directory
                        input_filename=$(basename "$input_file" .json)
                        OUTPUT_DIR="results/${domain}/${setting}/${hier}/${model_family}/${model}/${input_filename}"

                        echo -e "\n\e[31m********************** ${domain}/${setting}/${hier}/${input_filename} **********************\e[0m"

                        # Run inference using domain-specific script
                        if [ "$script_type" = "vllm" ]; then
                            python src/model/${run_script} \
                                -model ${model_path} \
                                -input ${input_file} \
                                -request_file ${OUTPUT_DIR}/input_request.json \
                                -response_file ${OUTPUT_DIR}/input_response.json \
                                -eval_output_dir ${OUTPUT_DIR} \
                                -max_tokens 2048 \
                                -temperature 0.0 \
                                -top_p 1.0 \
                                -top_k 1 \
                                -precision auto \
                                -backend ${script_type} \
                                -model_family ${model_family} \
                                -vllm_port ${VLLM_PORT} \
                                -max_retries 10 \
                                -max_threads 16 \
                                -sleep 5
                        else
                            python src/model/${run_script} \
                                -model ${model_path} \
                                -input ${input_file} \
                                -request_file ${OUTPUT_DIR}/input_request.json \
                                -response_file ${OUTPUT_DIR}/input_response.json \
                                -eval_output_dir ${OUTPUT_DIR} \
                                -max_tokens 4096 \
                                -backend ${script_type} \
                                -max_retries 10 \
                                -max_threads 16 \
                                -sleep 5
                        fi

                    done
                done
            done
        done
        
        # Stop vLLM server after processing all tasks for this model
        if [ "$script_type" = "vllm" ]; then
            stop_vllm_server
        fi
    done
done

echo -e "\n\e[32m=============== All tasks completed ===============\e[0m"
