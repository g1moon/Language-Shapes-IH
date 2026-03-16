"""
Evaluate task-execution domain tasks using chrF++ recall metric.
Uses threshold-based binary scoring: chrF++ recall >= threshold -> 1 (translated), < threshold -> 0 (not translated).
chrF++ recall measures how much of the reference's n-gram content appears in the model output,
directly capturing whether translation was performed.
"""

import os
import sys
import json
import argparse

sys.path.append(".")
import src.task_execution.evaluate as task_eval
from src.model.utils_common import extract_output
from tqdm import tqdm

import colorama
from termcolor import colored
colorama.init()

# Default threshold for chrF++ recall binary scoring.
# Determined via threshold analysis on 194,400 records across 9 models.
# At 0.15, F1=0.9435 for capable models (excluding llama3.2-1b/3b).
CHRF_RECALL_THRESHOLD = 0.15


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

    threshold = CHRF_RECALL_THRESHOLD

    # Load input data and responses
    data = json.load(open(args.input, "r", encoding="utf-8"))
    responses = json.load(open(args.response_file, "r", encoding="utf-8"))

    id2answer = {str(example["id"]): example["answer"] for example in data}

    # Use chrF++ recall for translation detection
    eval_func = task_eval.eval_chrf_recall

    # Evaluate
    results = []
    scores = []
    recall_values = []

    for example in tqdm(responses, desc="Evaluating with chrF++ recall"):
        id_ = str(example["id"])
        answer = id2answer[id_]
        prediction = extract_output(args, example)

        recall = round(eval_func(answer, prediction), 4)
        binary = 1 if recall >= threshold else 0
        scores.append(binary)
        recall_values.append(recall)

        save_input = example["input"]
        if "system" in example:
            save_input = [{"role": "system", "content": example["system"]}] + save_input

        results.append({
            "id": id_,
            "input": save_input,
            "answer": answer,
            "output": prediction,
            "chrf_recall": recall,
            "score": binary,
        })

    avg_score = sum(scores) / len(scores) if scores else 0
    avg_recall = sum(recall_values) / len(recall_values) if recall_values else 0

    print(f"Threshold: {threshold}")
    print(f"chrF++ recall: {avg_recall:.4f}, Accuracy: {avg_score:.4f} ({avg_score:.2%})")

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
