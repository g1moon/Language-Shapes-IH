"""
Evaluate safety domain tasks (multilingual version).
Uses chrF++ for language-agnostic system prompt leakage detection.
"""

import os
import sys
import json
import argparse

sys.path.append(".")
import src.safety.evaluate as safety_eval
from src.model.utils_common import extract_output
from tqdm import tqdm

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

    # Load input data and responses
    data = json.load(open(args.input, "r", encoding="utf-8"))
    responses = json.load(open(args.response_file, "r", encoding="utf-8"))
    
    id2answer = {str(example["id"]): example["answer"] for example in data}
    eval_func = safety_eval.eval_tensortrust

    # Evaluate
    results = []
    scores = []

    for example in tqdm(responses, desc="Evaluating Safety"):
        id_ = str(example["id"])
        answer = id2answer[id_]
        prediction = extract_output(args, example)

        score = round(eval_func(answer, prediction), 2)
        scores.append(score)

        save_input = example["input"]
        if "system" in example:
            save_input = [{"role": "system", "content": example["system"]}] + save_input

        results.append({
            "id": id_,
            "input": save_input,
            "answer": answer,
            "output": prediction,
            "score": score,
        })

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"Accuracy: {avg_score:.4f} ({avg_score:.2%})")

    # Save evaluation results
    try:
        results = sorted(results, key=lambda x: x["id"])
    except Exception as e:
        print(f'Skip sorting results due to error: {e}')

    os.makedirs(args.eval_output_dir, exist_ok=True)

    eval_output_path = os.path.join(args.eval_output_dir, "eval_results.json")

    json.dump(
        results,
        open(eval_output_path, "w", encoding="utf-8"),
        indent=4,
        ensure_ascii=False,
    )

    print(colored(f"Saved results to {eval_output_path}", "green"))
    print(colored("Evaluation completed!", "green"))


if __name__ == "__main__":
    main()
