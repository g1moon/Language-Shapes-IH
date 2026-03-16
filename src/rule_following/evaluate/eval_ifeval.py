"""
Evaluate the IFEval task
"""
from typing import Dict, List
import json
import os

import colorama
from termcolor import colored
colorama.init()


def save_list_as_jsonl(path: str, data):
    assert path.endswith(".jsonl")
    with open(path, "w", encoding="utf8") as fout:
        for instance in data:
            fout.write(json.dumps(instance))
            fout.write("\n")


def eval_ifeval(args, id2answer: Dict, responses: List, extract_output: callable):
    """
    Format the data according to the format required by IFEval's official evaluation script, then call their official evaluation script
    """
    input_data = json.load(open(args.input, "r", encoding="utf-8"))
    
    # Support both original IHEval format (with "instruction" field) and Multi_IH format (with "user" field)
    id2instruction = {}
    for example in input_data:
        if "instruction" in example:
            # Original IHEval format
            id2instruction[str(example["id"])] = example["instruction"]
        elif "user" in example:
            # Multi_IH format - use the user message as the instruction
            id2instruction[str(example["id"])] = example["user"]
        else:
            raise ValueError(f"Could not find 'instruction' or 'user' field in example {example['id']}")

    results = []
    for example in responses:
        id_ = str(example["id"])
        instruction = id2instruction[id_]
        prediction = extract_output(args, example)
        results.append({"prompt": instruction, "response": prediction})

    response_save_path = os.path.join(args.eval_output_dir, "ifeval_responses.jsonl")
    save_list_as_jsonl(response_save_path, results)

    evaluation_command = (
        f"python src/rule_following/evaluate/evaluation_main.py"
        f" --input_data={args.input}"
        f" --input_response_data={response_save_path}"
        f" --output_dir={args.eval_output_dir}"
    )

    print(colored("-" * 20 + "Evaluation" + "-" * 20, "green"))
    print(evaluation_command)
    os.system(evaluation_command)
    