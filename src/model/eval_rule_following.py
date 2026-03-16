"""
Evaluate rule-following domain tasks
"""

import os
import sys
import json
import argparse

sys.path.append(".")
import src.rule_following.evaluate as rule_eval
from src.model.utils_common import extract_output

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
        "-response_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-eval_output_dir", type=str, required=True
    )
    parser.add_argument(
        "-backend", type=str, choices=["vllm", "api"], required=True
    )
    args = parser.parse_args()

    # Load responses
    responses = json.load(open(args.response_file, "r", encoding="utf-8"))
    
    # Load input data to get id2answer mapping (for rule-following, we use eval_ifeval which handles this internally)
    data = json.load(open(args.input, "r", encoding="utf-8"))
    id2answer = {str(example["id"]): example["answer"] for example in data}

    # For rule-following tasks, use eval_ifeval which handles evaluation internally
    eval_func = rule_eval.eval_ifeval
    eval_func(args, id2answer, responses, extract_output)

    print(colored("Evaluation completed!", "green"))


if __name__ == "__main__":
    main()
