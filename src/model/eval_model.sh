#!/bin/bash
#SBATCH --job-name eval
#SBATCH --time 4-00:00:00
#SBATCH -c 4
#SBATCH --mem 100G
#SBATCH --gres=gpu:1

# Change this to specify which domain to evaluate
domains=(rule-following task-execution safety persona)  # evaluate all domains


# Change this to modify which model to evaluate
# model_list=(llama3.1-8b llama3.1-70b llama3.2-3b qwen3-4b qwen3-30b ministral3-8b ministral3-14b mistral-small-24b gpt-5 gpt-5-mini gpt-5-nano claude-haiku claude-sonnet)

# model_list=(llama3.1-70b)

# Domain to eval script mapping
declare -A domain_eval_script
domain_eval_script["rule-following"]="eval_rule_following.py"
domain_eval_script["task-execution"]="eval_task_execution.py"
domain_eval_script["safety"]="eval_safety.py"
domain_eval_script["persona"]="eval_persona.py"

# Make sure model-scores directory exists
mkdir -p model-scores/overall

for domain in ${domains[@]}; do
    # Check if domain is valid
    if [[ ! -v domain_eval_script[$domain] ]]; then
        echo "Error: Unknown domain: $domain"
        echo "Valid domains are: rule-following, task-execution, safety, persona"
        exit 1
    fi

    eval_script=${domain_eval_script[$domain]}

    # All domains (rule-following, task-execution, safety) use multilingual hierarchy structure
    if [ "$domain" = "rule-following" ] || [ "$domain" = "task-execution" ] || [ "$domain" = "safety" ] || [ "$domain" = "persona" ]; then
        # Find all evaluation types (reference, conflict, etc.)
        eval_types=($(find benchmark/${domain} -mindepth 1 -maxdepth 1 -type d | xargs -n 1 basename))
        
        for eval_type in ${eval_types[@]}; do
            # Find all hierarchy types (sys-tool, sys-user, user-tool)
            hierarchy_types=($(find benchmark/${domain}/${eval_type} -mindepth 1 -maxdepth 1 -type d | xargs -n 1 basename))
            
            for hierarchy_type in ${hierarchy_types[@]}; do
                # Find all language combination files
                lang_files=($(find benchmark/${domain}/${eval_type}/${hierarchy_type} -maxdepth 1 -name "input-*.json" -type f | xargs -n 1 basename))
                
                for lang_file in ${lang_files[@]}; do
                    # Extract language combination name (e.g., input-sys_en-tool_en.json -> input-sys_en-tool_en)
                    # Keep "input-" prefix as aggregate_matrix_scores.py expects it
                    lang_combo="${lang_file%.json}"
                    
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

                        echo -e "\n\e[31m********************** ${domain}/${eval_type}/${hierarchy_type}/${lang_combo} **********************\e[0m"

                        # Use results directory for all model types
                        OUTPUT_DIR=results/${domain}/${eval_type}/${hierarchy_type}/${model_family}/${model}/${lang_combo}

                        # Run evaluation using domain-specific script
                        python src/model/${eval_script} \
                            -model ${model_path} \
                            -input benchmark/${domain}/${eval_type}/${hierarchy_type}/${lang_file} \
                            -response_file ${OUTPUT_DIR}/input_response.json \
                            -eval_output_dir ${OUTPUT_DIR} \
                            -backend ${script_type}

                        echo -e "\e[32m--------------------- Process Scores ---------------------\e[0m"

                        # Check if evaluation results exist
                        if [ "$domain" = "safety" ] || [ "$domain" = "task-execution" ] || [ "$domain" = "persona" ]; then
                            if [ -f "${OUTPUT_DIR}/eval_results.json" ]; then
                                echo -e "\e[32mSuccessfully generated evaluation results for ${lang_combo}\e[0m"
                            else
                                echo -e "\e[33mWarning: eval_results.json not found at ${OUTPUT_DIR}\e[0m"
                            fi
                        else
                            if [ -f "${OUTPUT_DIR}/eval_results_strict.json" ] && [ -f "${OUTPUT_DIR}/eval_results_loose.json" ]; then
                                echo -e "\e[32mSuccessfully generated evaluation results for ${lang_combo}\e[0m"
                            else
                                echo -e "\e[33mWarning: eval_results_strict.json or eval_results_loose.json not found at ${OUTPUT_DIR}\e[0m"
                            fi
                        fi

                    done
                done
            done
        done
    fi
    
    # After processing all models in the domain, handle post-processing
    for model in ${model_list[@]}; do
        # Determine model family
        if [[ "$model" =~ ^llama ]]; then
            model_family="llama"
        elif [[ "$model" =~ ^qwen ]]; then
            model_family="qwen"
        elif [[ "$model" =~ ^ministral ]] || [[ "$model" =~ ^mistral-small ]]; then
            model_family="mistral"
        elif [[ "$model" =~ ^gpt ]]; then
            model_family="gpt"
        elif [[ "$model" =~ ^claude ]]; then
            model_family="claude"
        fi
        
        # Generate 6x6 language matrices for all domains
        if [[ " ${domains[@]} " =~ " rule-following " ]] || [[ " ${domains[@]} " =~ " task-execution " ]] || [[ " ${domains[@]} " =~ " safety " ]] || [[ " ${domains[@]} " =~ " persona " ]]; then
            echo -e "\e[32m--------------------- Aggregating Matrix Scores ---------------------\e[0m"
            python src/model/aggregate_matrix_scores.py \
                -results_dir results \
                -model_family ${model_family} \
                -model_name ${model} \
                -output_dir model-scores \
                -domain ${domain}
            
            # Calculate hierarchy averages for this model and domain
            echo -e "\e[32m--------------------- Calculating Hierarchy Averages ---------------------\e[0m"
            python src/model/calculate_hierarchy_avg.py \
                -model_family ${model_family} \
                -model_name ${model} \
                -domain ${domain} \
                -output_dir model-scores
        fi
    done
    
    # After processing all models, generate family averages
    if [[ " ${domains[@]} " =~ " rule-following " ]] || [[ " ${domains[@]} " =~ " task-execution " ]] || [[ " ${domains[@]} " =~ " safety " ]] || [[ " ${domains[@]} " =~ " persona " ]]; then
        # Get unique model families from the model list
        declare -A families_to_process
        for model in ${model_list[@]}; do
            # Determine model family
            if [[ "$model" =~ ^llama ]]; then
                families_to_process["llama"]=1
            elif [[ "$model" =~ ^qwen ]]; then
                families_to_process["qwen"]=1
            elif [[ "$model" =~ ^ministral ]] || [[ "$model" =~ ^mistral-small ]]; then
                families_to_process["mistral"]=1
            elif [[ "$model" =~ ^gpt ]]; then
                families_to_process["gpt"]=1
            elif [[ "$model" =~ ^claude ]]; then
                families_to_process["claude"]=1
            fi
        done
        
        # Process family averages
        for family in "${!families_to_process[@]}"; do
            echo -e "\e[32m--------------------- Aggregating Family Scores: ${family} ---------------------\e[0m"
            python src/model/aggregate_family_scores.py \
                -model_family ${family} \
                -output_dir model-scores \
                -domain ${domain}
        done
    fi

done
