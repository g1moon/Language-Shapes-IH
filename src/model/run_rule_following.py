"""
Run inference for rule-following domain
"""

import os
import sys
import json
import argparse

sys.path.append(".")

import colorama
from termcolor import colored
colorama.init()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model", type=str, required=True, help='Huggingface model name or local path'
    )
    parser.add_argument(
        "-input",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-request_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-response_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-eval_output_dir", type=str, required=True
    )
    parser.add_argument("-max_tokens", type=int, default=2048)
    parser.add_argument("-top_k", type=int, default=250)
    parser.add_argument("-top_p", type=float, default=1.0)
    parser.add_argument("-temperature", type=float, default=0.0)
    parser.add_argument("-precision", type=str, default="auto")
    parser.add_argument(
        "-backend", type=str, choices=["vllm", "api"], required=True
    )
    parser.add_argument("-model_family", type=str, default="qwen", choices=["qwen", "mistral", "llama"],
                        help="Model family for vllm backend")
    parser.add_argument("-vllm_port", type=int, default=8000,
                        help="Port for vLLM API server")

    # vllm arguments
    parser.add_argument("-tensor_parallel", type=int, default=1)
    parser.add_argument("-gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("-max_model_len", type=int, default=8192)

    # API call arguments
    parser.add_argument("-max_loop", type=int, default=2)
    parser.add_argument("-max_retries", type=int, default=10)
    parser.add_argument("-max_threads", type=int, default=8)
    parser.add_argument("-sleep", type=int, default=2)
    args = parser.parse_args()

    data = json.load(open(args.input, "r", encoding="utf-8"))
    print(f"Loaded {len(data)} data instances from {args.input}")

    # prepare model inputs
    requests = []
    id2answer = {}
    for example in data:
        id_ = example["id"]
        messages = []

        # # For multi-turn settings, there should be a "conversation_history" field in the example
        # if "conversation_history" in example:
        #     messages.extend([
        #         {"role": "user", "content": msg} if i % 2 == 0 else {"role": "assistant", "content": msg}
        #         for i, msg in enumerate(example["conversation_history"])
        #     ])

        messages.append({"role": "user", "content": example["user"]})

        request_example = {"id": id_}

        if "system" in example and example['system'] is not None:
            if args.backend == "api":
                request_example["system"] = example["system"]
            elif args.backend == "vllm":
                messages.insert(0, {"role": "system", "content": example["system"]})

        request_example["messages"] = messages

        if "tool" in example:
            request_example["tool"] = example["tool"]

        requests.append(request_example)
        id2answer[id_] = example["answer"]

    # create directory if not exists
    os.makedirs(os.path.dirname(args.request_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.response_file), exist_ok=True)
    json.dump(
        requests,
        open(args.request_file, "w", encoding="utf-8"),
        indent=4,
        ensure_ascii=False,
    )

    # run inference command
    if args.backend == "vllm":
        inference_script = "src/utils/run_vllm_api_model.py"
    elif args.backend == "api":
        inference_script = "src/utils/call_api.py"
    else:
        raise NotImplementedError

    inference_command = (
        f"python {inference_script}"
        f" -model {args.model}"
        f" -input {args.request_file}"
        f" -output {args.response_file}"
        f" -max_new_tokens {args.max_tokens}"
    )

    if args.backend == "api":
        inference_command += (
            f" -max_loop {args.max_loop}"
            f" -max_retries {args.max_retries}"
            f" -max_threads {args.max_threads}"
            f" -sleep {args.sleep}"
            f" -benchmark_file {args.input}"
        )
    elif args.backend == "vllm":
        inference_command += (
            f" -top_k {args.top_k}"
            f" -top_p {args.top_p}"
            f" -temperature {args.temperature}"
            f" -model_family {args.model_family}"
            f" -port {args.vllm_port}"
            f" -benchmark_file {args.input}"
        )

    # Run inference (if inference was done before, the script will skip running the model)
    print(colored("-" * 20 + "Inference" + "-" * 20, "green"))
    print(inference_command)
    os.system(inference_command)

    print(colored("Inference completed!", "green"))


if __name__ == "__main__":
    main()
