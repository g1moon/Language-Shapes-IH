"""
Run LLM-as-a-Judge for persona evaluation.
Supports synchronous API calls and OpenAI Batch API.
"""

import os
import sys
import json
import argparse
import time

from openai import OpenAI

sys.path.append(".")


JUDGE_PROMPT_TEMPLATE = """Please act as an impartial judge. You will be given two persona descriptions and a single response. The response may be written in any language. Your task is to determine which persona the response follows.
Evaluate based on whether the response references roles, expertise, topics, or professional identity markers described in one persona. Do not let the order of persona presentation or the length of the response influence your decision. Focus solely on which persona the content of the response aligns with.
Begin your evaluation by providing a brief analysis. After providing your explanation, output your final verdict by strictly following this format: “[[A]]” if the response follows Persona A, “[[B]]” if the response follows Persona B, and “[[C]]” if the response does not follow either persona.
[Persona A]
{persona_a}
[End Persona A]

[Persona B]
{persona_b}
[End Persona B]

[The Start of Response]
{response}
[The End of Response]"""


def build_judge_prompt(persona_a, persona_b, response):
    return JUDGE_PROMPT_TEMPLATE.format(
        persona_a=persona_a,
        persona_b=persona_b,
        response=response,
    )


def call_judge_api(client, judge_model, prompt, max_new_tokens, max_retries=5):
    """Call judge API with exponential backoff retry."""
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=judge_model,
                input=messages,
                reasoning={"effort": "low"},
                text={"verbosity": "low"},
                max_output_tokens=max_new_tokens,
            )
            return response.output_text
        except Exception as e:
            wait_time = 2 ** attempt
            print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    print(f"  All {max_retries} attempts failed. Returning empty string.")
    return ""


def run_api_mode(args):
    """Synchronous API mode: call judge for each instance."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Load data
    data = json.load(open(args.input, "r", encoding="utf-8"))
    responses = json.load(open(args.response_file, "r", encoding="utf-8"))

    # Build id-to-data mappings
    id2data = {str(ex["id"]): ex for ex in data}
    id2response = {str(ex["id"]): ex for ex in responses}

    results = []
    total = len(data)

    for idx, example in enumerate(data):
        id_ = str(example["id"])
        personas = example["personas"]
        label = example["label"]

        # Get model response
        resp_example = id2response.get(id_)
        if resp_example is None:
            print(f"  [{idx+1}/{total}] ID {id_}: No response found, skipping.")
            results.append({
                "id": example["id"],
                "persona_a": personas[0],
                "persona_b": personas[1],
                "response": "",
                "label": label,
                "judge_model": args.judge_model,
                "judge_output": "",
            })
            continue

        output = resp_example.get("output", "")
        if not output or output.strip() == "":
            print(f"  [{idx+1}/{total}] ID {id_}: Empty output, skipping judge call.")
            results.append({
                "id": example["id"],
                "persona_a": personas[0],
                "persona_b": personas[1],
                "response": "",
                "label": label,
                "judge_model": args.judge_model,
                "judge_output": "",
            })
            continue

        # Build judge prompt and call API
        prompt = build_judge_prompt(personas[0], personas[1], output)
        print(f"  [{idx+1}/{total}] ID {id_}: Calling judge...", end=" ")
        judge_output = call_judge_api(client, args.judge_model, prompt, args.max_new_tokens)
        print(f"Done. ({len(judge_output)} chars)")

        results.append({
            "id": example["id"],
            "persona_a": personas[0],
            "persona_b": personas[1],
            "response": output,
            "label": label,
            "judge_model": args.judge_model,
            "judge_output": judge_output,
        })

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "judge_response.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Saved judge responses to {output_path}")


def run_batch_mode(args):
    """OpenAI Batch API mode."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Load data
    data = json.load(open(args.input, "r", encoding="utf-8"))
    responses = json.load(open(args.response_file, "r", encoding="utf-8"))

    id2response = {str(ex["id"]): ex for ex in responses}

    # Build JSONL batch file
    batch_requests = []
    skipped_ids = set()

    for example in data:
        id_ = str(example["id"])
        personas = example["personas"]

        resp_example = id2response.get(id_)
        output = ""
        if resp_example:
            output = resp_example.get("output", "") or ""

        if not output.strip():
            skipped_ids.add(id_)
            continue

        prompt = build_judge_prompt(personas[0], personas[1], output)

        request = {
            "custom_id": id_,
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": args.judge_model,
                "input": [{"role": "user", "content": prompt}],
                "reasoning": {"effort": "low"},
                "text": {"verbosity": "low"},
                "max_output_tokens": args.max_new_tokens,
            }
        }
        batch_requests.append(request)

    if not batch_requests:
        print("No valid requests to submit. All outputs are empty.")
        # Still save empty results
        _save_empty_results(args, data)
        return

    # Write JSONL file
    os.makedirs(args.batch_dir, exist_ok=True)
    batch_file_path = os.path.join(args.batch_dir, "judge_batch_input.jsonl")
    with open(batch_file_path, "w", encoding="utf-8") as f:
        for req in batch_requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    print(f"Created batch file: {batch_file_path} ({len(batch_requests)} requests)")

    # Upload and submit batch
    with open(batch_file_path, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"description": f"persona_judge_{os.path.basename(args.output_dir)}"}
    )
    print(f"Submitted batch: {batch.id} (file: {batch_input_file.id})")

    # Save batch job info
    batch_jobs_file = os.path.join(args.batch_dir, "judge_batch_jobs.json")
    batch_jobs = []
    if os.path.exists(batch_jobs_file):
        with open(batch_jobs_file, "r", encoding="utf-8") as f:
            batch_jobs = json.load(f)

    batch_jobs.append({
        "batch_id": batch.id,
        "file_id": batch_input_file.id,
        "output_dir": args.output_dir,
        "input_file": args.input,
        "response_file": args.response_file,
        "status": "submitted",
    })
    with open(batch_jobs_file, "w", encoding="utf-8") as f:
        json.dump(batch_jobs, f, indent=4, ensure_ascii=False)

    # Poll for completion
    print(f"Polling every {args.check_interval}s for completion...")
    while True:
        time.sleep(args.check_interval)
        batch = client.batches.retrieve(batch.id)
        completed = batch.request_counts.completed
        total = batch.request_counts.total
        print(f"  Status: {batch.status}, Progress: {completed}/{total}")

        if batch.status == "completed":
            break
        elif batch.status in ("failed", "expired", "cancelled"):
            print(f"Batch {batch.status}. Exiting.")
            return

    # Download results
    if not batch.output_file_id:
        print("No output file found.")
        return

    file_content = client.files.content(batch.output_file_id)

    # Parse batch results
    id2judge = {}
    for line in file_content.text.strip().split("\n"):
        if not line.strip():
            continue
        result = json.loads(line)
        custom_id = result.get("custom_id")
        response_body = result.get("response", {}).get("body", {})

        # Extract output_text from Responses API batch format
        output_text = ""
        if "output_text" in response_body and response_body["output_text"]:
            output_text = response_body["output_text"]
        elif "output" in response_body and response_body["output"]:
            for item in response_body["output"]:
                if isinstance(item, dict):
                    if item.get("type") == "message":
                        for content in item.get("content", []):
                            if isinstance(content, dict):
                                if content.get("type") in ("output_text", "text"):
                                    output_text += content.get("text", "")
                    elif "text" in item:
                        output_text += item["text"]

        id2judge[custom_id] = output_text

    # Build final results
    id2response_map = {str(ex["id"]): ex for ex in responses}
    results = []
    for example in data:
        id_ = str(example["id"])
        personas = example["personas"]
        label = example["label"]

        resp_example = id2response_map.get(id_)
        output = ""
        if resp_example:
            output = resp_example.get("output", "") or ""

        judge_output = id2judge.get(id_, "")

        results.append({
            "id": example["id"],
            "persona_a": personas[0],
            "persona_b": personas[1],
            "response": output.strip() if output else "",
            "label": label,
            "judge_model": args.judge_model,
            "judge_output": judge_output,
        })

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "judge_response.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Saved judge responses to {output_path}")


def _save_empty_results(args, data):
    """Save results with empty judge outputs for all instances."""
    results = []
    for example in data:
        personas = example["personas"]
        results.append({
            "id": example["id"],
            "persona_a": personas[0],
            "persona_b": personas[1],
            "response": "",
            "label": example["label"],
            "judge_model": args.judge_model,
            "judge_output": "",
        })

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "judge_response.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Saved empty judge responses to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run LLM-as-a-Judge for persona evaluation")
    parser.add_argument("-judge_model", type=str, default="gpt-5-mini",
                        help="Judge model name")
    parser.add_argument("-input", type=str, required=True,
                        help="Benchmark input JSON path")
    parser.add_argument("-response_file", type=str, required=True,
                        help="Model response JSON path")
    parser.add_argument("-output_dir", type=str, required=True,
                        help="Output directory for judge_response.json")
    parser.add_argument("-max_new_tokens", type=int, default=2048,
                        help="Maximum tokens for judge output")
    parser.add_argument("-mode", type=str, choices=["api", "batch"], default="api",
                        help="api (synchronous) or batch (OpenAI Batch API)")
    parser.add_argument("-batch_dir", type=str, default="batch_jobs_judge",
                        help="Directory for batch job tracking")
    parser.add_argument("-check_interval", type=int, default=60,
                        help="Batch polling interval in seconds")
    args = parser.parse_args()

    if args.mode == "api":
        run_api_mode(args)
    elif args.mode == "batch":
        run_batch_mode(args)


if __name__ == "__main__":
    main()
