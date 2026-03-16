"""
Common utility functions for running models
"""


def extract_output(args, example):
    """Extract output from model response based on backend type"""
    if args.backend == "api":
        if "gpt" in args.model or "claude" in args.model:
            if example['output'] is not None:
                return example["output"].strip()
            else:
                return ""
        else:
            raise NotImplementedError
    elif args.backend == "vllm":
        # vllm also returns output in the same format
        if example['output'] is not None:
            return example["output"].strip()
        else:
            return ""
    else:
        if example['output'] is not None:
            return example["output"].strip()
        else:
            return ""


def get_model_config(model):
    """Get model configuration based on model name"""
    # Llama models
    if model == "llama3.1-8b":
        return {
            "model_family": "llama",
            "model_path": "meta-llama/Llama-3.1-8B-Instruct",
            "script_type": "vllm"
        }
    elif model == "llama3.1-70b":
        return {
            "model_family": "llama",
            "model_path": "meta-llama/Llama-3.1-70B-Instruct",
            "script_type": "vllm"
        }
    elif model == "llama3.2-3b":
        return {
            "model_family": "llama",
            "model_path": "meta-llama/Llama-3.2-3B-Instruct",
            "script_type": "vllm"
        }

    # Qwen models
    elif model == "qwen3-4b":
        return {
            "model_family": "qwen",
            "model_path": "Qwen/Qwen3-4B-Instruct-2507",
            "script_type": "vllm"
        }
    elif model == "qwen3-30b":
        return {
            "model_family": "qwen",
            "model_path": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "script_type": "vllm"
        }
    # Ministral models
    elif model == "ministral3-8b":
        return {
            "model_family": "mistral",
            "model_path": "mistralai/Ministral-3-8B-Instruct-2512",
            "script_type": "vllm"
        }
    elif model == "ministral3-14b":
        return {
            "model_family": "mistral",
            "model_path": "mistralai/Ministral-3-14B-Instruct-2512",
            "script_type": "vllm"
        }
    elif model == "mistral-small-24b":
        return {
            "model_family": "mistral",
            "model_path": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            "script_type": "vllm"
        }

    # GPT models
    elif model == "gpt-5":
        return {
            "model_family": "gpt",
            "model_path": "gpt-5-2025-08-07",
            "script_type": "api"
        }
    elif model == "gpt-5-mini":
        return {
            "model_family": "gpt",
            "model_path": "gpt-5-mini-2025-08-07",
            "script_type": "api"
        }
    elif model == "gpt-5-nano":
        return {
            "model_family": "gpt",
            "model_path": "gpt-5-nano-2025-08-07",
            "script_type": "api"
        }
    # Claude models
    elif model == "claude-haiku":
        return {
            "model_family": "claude",
            "model_path": "claude-haiku-4-5-20251001",
            "script_type": "api"
        }
    elif model == "claude-sonnet":
        return {
            "model_family": "claude",
            "model_path": "claude-sonnet-4-5-20250929",
            "script_type": "api"
        }
    else:
        raise ValueError(f"Model not found: {model}")
