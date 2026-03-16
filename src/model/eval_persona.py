"""
Evaluate persona domain tasks.
Reads judge_response.json, extracts verdicts, computes scores.
Outputs eval_results.json.
"""

import os
import sys
import json
import argparse

sys.path.append(".")
import src.persona.evaluate as persona_eval
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

    # Load benchmark input data (for labels)
    data = json.load(open(args.input, "r", encoding="utf-8"))
    id2answer = {str(example["id"]): example["label"] for example in data}

    # Load judge responses
    judge_response_path = os.path.join(args.eval_output_dir, "judge_response.json")
    if not os.path.exists(judge_response_path):
        print(colored(f"Error: judge_response.json not found at {judge_response_path}", "red"))
        print(colored("Run eval_persona_judge.sh first to generate judge responses.", "red"))
        sys.exit(1)

    judge_responses = json.load(open(judge_response_path, "r", encoding="utf-8"))

    # Evaluate
    results = []
    scores = []
    null_count = 0

    for example in tqdm(judge_responses, desc="Evaluating Persona"):
        id_ = str(example["id"])
        label = id2answer[id_]
        judge_output = example.get("judge_output", "")
        response = example.get("response", "")

        # Extract verdict
        verdict = persona_eval.extract_verdict(judge_output)

        if verdict is None:
            null_count += 1
            results.append({
                "id": id_,
                "input": example.get("persona_a", ""),
                "answer": label,
                "output": response,
                "judge_output": judge_output,
                "verdict": None,
                "score": None,
            })
            continue

        # Compute score
        score = persona_eval.eval_persona(verdict, label)
        scores.append(score)

        results.append({
            "id": id_,
            "input": example.get("persona_a", ""),
            "answer": label,
            "output": response,
            "judge_output": judge_output,
            "verdict": verdict,
            "score": score,
        })

    # Print stats
    valid_count = len(scores)
    total_count = len(judge_responses)
    accuracy = sum(scores) / valid_count if valid_count > 0 else 0

    print(f"Total: {total_count}, Valid verdicts: {valid_count}, Null verdicts: {null_count}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")

    # Sort results
    try:
        results = sorted(results, key=lambda x: int(x["id"]))
    except Exception as e:
        print(f"Skip sorting results due to error: {e}")

    # Save evaluation results
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
