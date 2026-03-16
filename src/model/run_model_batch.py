#!/usr/bin/env python3
"""
Batch processing for GPT and Claude models.
Creates batches for all domain/setting/hierarchy/language combinations,
then polls status every minute until all batches complete.
"""

import os
import sys
import json
import argparse
import time
import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from openai import OpenAI
import itertools

sys.path.append(".")

# Claude API support
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic package not installed. Claude batch API will not be available.")


# Domain to run script mapping
DOMAIN_RUN_SCRIPT = {
    "rule-following": "run_rule_following.py",
    "safety": "run_safety.py",
    "task-execution": "run_task_execution.py",
    "persona": "run_persona.py"
}

# Model configurations
MODEL_CONFIGS = {
    "gpt-5": {"model_family": "gpt", "model_path": "gpt-5-2025-08-07", "script_type": "api"},
    "gpt-5-mini": {"model_family": "gpt", "model_path": "gpt-5-mini-2025-08-07", "script_type": "api"},
    "gpt-5-nano": {"model_family": "gpt", "model_path": "gpt-5-nano-2025-08-07", "script_type": "api"},
    "claude-haiku": {"model_family": "claude", "model_path": "claude-haiku-4-5-20251001", "script_type": "api"},
    "claude-sonnet": {"model_family": "claude", "model_path": "claude-sonnet-4-5-20250929", "script_type": "api"},
}


def tool_call_openai(tool: Dict):
    """Convert tool call to OpenAI format"""
    raw_definition = tool["definition"]
    raw_tool_call = tool["call"]
    raw_tool_return = tool["return"]

    definition = [{
        "type": "function",
        "name": raw_definition['name'],
        "description": raw_definition['description'],
        "parameters": {
            **raw_definition["parameters"],
            "additionalProperties": False
        },
        "strict": True
    }]

    reasoning_item = {
        "summary": [],
        "type": 'reasoning'
    }

    tool_call = {
        "type": "function_call",
        'name': raw_tool_call['name'],
        "call_id": raw_tool_call['id'],
        "arguments": json.dumps(raw_tool_call['arguments']),
        "status": "completed"
    }

    tool_return = {
        "type": "function_call_output",
        "call_id": raw_tool_return['id'],
        "output": json.dumps({"content": raw_tool_return['content']})
    }

    return definition, reasoning_item, tool_call, tool_return


def tool_call_claude(tool: Dict):
    """Convert tool call to Claude format"""
    raw_definition = tool["definition"]
    raw_tool_call = tool["call"]
    raw_tool_return = tool["return"]

    definition = [{
        "name": raw_definition['name'],
        "description": raw_definition['description'],
        "input_schema": raw_definition['parameters']
    }]

    tool_call = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": raw_tool_call['id'],
                "name": raw_tool_call['name'],
                "input": raw_tool_call['arguments']
            }
        ]
    }

    tool_output = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": raw_tool_return['id'],
                "content": raw_tool_return['content']
            }
        ]
    }

    return definition, tool_call, tool_output


def prepare_batch_requests(input_file: str, model_family: str, model_path: str, request_file: str,
                           max_tokens: int = 2048) -> Tuple[List[Dict], List[Dict]]:
    """Prepare batch requests from input file and write request file"""
    from copy import deepcopy

    data = json.load(open(input_file, "r", encoding="utf-8"))
    batch_requests = []
    request_data = []

    for example in data:
        id_ = example["id"]

        request_example = {
            "id": id_,
            "messages": [{"role": "user", "content": example["user"]}]
        }
        
        if "system" in example and example['system'] is not None:
            request_example["system"] = example["system"]
        
        if "tool" in example:
            request_example["tool"] = example["tool"]
        
        request_data.append(request_example)

        if model_family == "gpt":
            batch_input = [{"role": "user", "content": example["user"]}]

            if "system" in example and example['system'] is not None:
                batch_input.insert(0, {"role": "developer", "content": example["system"]})

            if "tool" in example:
                tool_definition, reasoning_item, tool_call, tool_return = tool_call_openai(example["tool"])
                batch_input.extend([reasoning_item, tool_call, tool_return])

            request = {
                "custom_id": str(id_),
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model_path,
                    "input": batch_input,
                    "reasoning": {"effort": "low"},
                    "text": {"verbosity": "low"},
                    "max_output_tokens": max_tokens,
                }
            }

        elif model_family == "claude":
            batch_messages = [{"role": "user", "content": example["user"]}]

            if "tool" in example:
                tool_definition, tool_call_msg, tool_output = tool_call_claude(example["tool"])
                batch_messages.append(tool_call_msg)
                batch_messages.append(tool_output)

            request = {
                "custom_id": str(id_),
                "params": {
                    "model": model_path,
                    "max_tokens": max_tokens,
                    "messages": batch_messages
                }
            }

            if "system" in example and example['system'] is not None:
                request["params"]["system"] = example["system"]
            

        batch_requests.append(request)
    
    os.makedirs(os.path.dirname(request_file), exist_ok=True)
    with open(request_file, "w", encoding="utf-8") as f:
        json.dump(request_data, f, indent=4, ensure_ascii=False)
    print(f"✓ Request file created: {request_file}")

    return batch_requests, request_data


def create_batch_file(batch_requests: List[Dict], output_path: str):
    """Save batch requests to a JSONL file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for request in batch_requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    print(f"✓ Batch file created: {output_path} ({len(batch_requests)} requests)")


def submit_openai_batch(batch_file_path: str, description: str) -> Tuple[str, str]:
    """Submit an OpenAI batch job"""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with open(batch_file_path, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"description": description}
    )

    print(f"✓ OpenAI batch submitted: {batch.id} (file: {batch_input_file.id})")
    return batch.id, batch_input_file.id


def submit_claude_batch(batch_file_path: str, description: str) -> str:
    """Submit a Claude batch job"""
    if not ANTHROPIC_AVAILABLE:
        raise ImportError("anthropic package is required for Claude batch API")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    with open(batch_file_path, "r", encoding="utf-8") as f:
        requests = [json.loads(line) for line in f if line.strip()]

    batch = client.messages.batches.create(requests=requests)

    print(f"✓ Claude batch submitted: {batch.id}")
    return batch.id


def check_openai_batch_status(batch_id: str) -> Tuple[str, int, int]:
    """Check OpenAI batch status"""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batch = client.batches.retrieve(batch_id)
    
    total = batch.request_counts.total
    completed = batch.request_counts.completed
    failed = batch.request_counts.failed
    
    return batch.status, completed, total


def check_claude_batch_status(batch_id: str) -> Tuple[str, int, int]:
    """Check Claude batch status"""
    if not ANTHROPIC_AVAILABLE:
        raise ImportError("anthropic package is required for Claude batch API")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    batch = client.messages.batches.retrieve(batch_id)

    # Possible statuses: in_progress, canceling, ended
    status = batch.processing_status
    completed = batch.request_counts.succeeded + batch.request_counts.errored
    total = batch.request_counts.processing + batch.request_counts.succeeded + batch.request_counts.errored
    
    return status, completed, total


def download_openai_batch_results(batch_id: str, output_path: str, request_file: str) -> bool:
    """Download OpenAI batch results"""
    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        batch = client.batches.retrieve(batch_id)

        if batch.status != "completed":
            return False

        if not batch.output_file_id:
            print(f"⚠ Batch {batch_id} has no output file.")
            return False

        # Load input data from request file to reconstruct input field
        input_data = {}
        if os.path.exists(request_file):
            with open(request_file, "r", encoding="utf-8") as f:
                requests = json.load(f)
                for req in requests:
                    from copy import deepcopy
                    messages = deepcopy(req["messages"])

                    if "tool" in req:
                        tool_definition, reasoning_item, tool_call_item, tool_return = tool_call_openai(req["tool"])
                        messages.extend([reasoning_item, tool_call_item, tool_return])

                    if "system" in req and req["system"] is not None:
                        messages.insert(0, {"role": "developer", "content": req["system"]})

                    input_data[str(req["id"])] = messages

        file_content = client.files.content(batch.output_file_id)

        results = []
        for line in file_content.text.strip().split("\n"):
            if line.strip():
                result = json.loads(line)
                custom_id = result.get("custom_id")
                response = result.get("response", {})
                body = response.get("body", {})
                
                output_text = ""

                # Extract output_text from OpenAI Responses API batch response
                if "output_text" in body and body["output_text"]:
                    output_text = body["output_text"]

                elif "output" in body and body["output"]:
                    for item in body["output"]:
                        if isinstance(item, dict):
                            if item.get("type") == "message":
                                for content in item.get("content", []):
                                    if isinstance(content, dict):
                                        if content.get("type") == "output_text":
                                            output_text += content.get("text", "")
                                        elif content.get("type") == "text":
                                            output_text += content.get("text", "")
                            elif "text" in item:
                                output_text += item["text"]
                            elif "content" in item and isinstance(item["content"], list):
                                for content in item["content"]:
                                    if isinstance(content, dict) and "text" in content:
                                        output_text += content["text"]
                
                # Fallback: Chat Completions format
                if not output_text and "choices" in body:
                    for choice in body["choices"]:
                        if "message" in choice and "content" in choice["message"]:
                            output_text += choice["message"]["content"] or ""
                
                results.append({
                    "id": custom_id,
                    "input": input_data.get(custom_id, []),
                    "output": output_text
                })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"✓ Results downloaded: {output_path}")
        return True

    except Exception as e:
        print(f"✗ Download failed ({batch_id}): {e}")
        import traceback
        traceback.print_exc()
        return False


def download_claude_batch_results(batch_id: str, output_path: str, request_file: str) -> bool:
    """Download Claude batch results"""
    try:
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is required for Claude batch API")

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        batch = client.messages.batches.retrieve(batch_id)

        if batch.processing_status != "ended":
            return False

        # Load input data from request file to reconstruct input field
        input_data = {}
        if os.path.exists(request_file):
            with open(request_file, "r", encoding="utf-8") as f:
                requests = json.load(f)
                for req in requests:
                    from copy import deepcopy
                    messages = deepcopy(req["messages"])

                    if "tool" in req:
                        tool_definition, tool_call_msg, tool_output = tool_call_claude(req["tool"])
                        messages.append(tool_call_msg)
                        messages.append(tool_output)

                    input_data[str(req["id"])] = messages

        results = []
        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id

            output_text = ""
            if result.result.type == "succeeded":
                message = result.result.message
                for content in message.content:
                    if hasattr(content, 'text'):
                        output_text += content.text

            results.append({
                "id": custom_id,
                "input": input_data.get(custom_id, []),
                "output": output_text
            })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"✓ Results downloaded: {output_path}")
        return True

    except Exception as e:
        print(f"✗ Download failed ({batch_id}): {e}")
        import traceback
        traceback.print_exc()
        return False


def get_input_file_path(domain: str, setting: str, hier: str, high_lang: str, low_lang: str) -> str:
    """Build input file path for given domain/setting/hierarchy/language combination"""
    if hier == "sys-user":
        return f"benchmark/{domain}/{setting}/{hier}/input-sys_{high_lang}-user_{low_lang}.json"
    elif hier == "sys-tool":
        return f"benchmark/{domain}/{setting}/{hier}/input-sys_{high_lang}-tool_{low_lang}.json"
    elif hier == "user-tool":
        return f"benchmark/{domain}/{setting}/{hier}/input-user_{high_lang}-tool_{low_lang}.json"
    else:
        raise ValueError(f"Unknown hierarchy: {hier}")


def get_output_dir(domain: str, setting: str, hier: str, model_family: str, model: str,
                   high_lang: str, low_lang: str, hier_type: str) -> str:
    """Build output directory path"""
    if hier_type == "sys-user":
        input_filename = f"input-sys_{high_lang}-user_{low_lang}"
    elif hier_type == "sys-tool":
        input_filename = f"input-sys_{high_lang}-tool_{low_lang}"
    elif hier_type == "user-tool":
        input_filename = f"input-user_{high_lang}-tool_{low_lang}"
    
    return f"results/{domain}/{setting}/{hier}/{model_family}/{model}/{input_filename}"


def redownload_batch_results(batch_dir: str):
    """Re-download completed batch results from an existing batch_jobs.json"""
    batch_jobs_file = f"{batch_dir}/batch_jobs.json"

    if not os.path.exists(batch_jobs_file):
        print(f"✗ batch_jobs.json not found: {batch_jobs_file}")
        return

    with open(batch_jobs_file, "r", encoding="utf-8") as f:
        batch_jobs = json.load(f)

    print("=" * 80)
    print(f"Re-downloading results for {len(batch_jobs)} batch jobs")
    print("=" * 80)
    
    success_count = 0
    fail_count = 0
    
    for i, job in enumerate(batch_jobs):
        batch_id = job["batch_id"]
        model_family = job["model_family"]
        description = job["description"]
        response_file = job["response_file"]
        request_file = job["request_file"]
        
        print(f"\n[{i+1}/{len(batch_jobs)}] {description}")
        
        try:
            if model_family == "gpt":
                success = download_openai_batch_results(batch_id, response_file, request_file)
            elif model_family == "claude":
                success = download_claude_batch_results(batch_id, response_file, request_file)
            else:
                print(f"  ⚠ Unknown model family: {model_family}")
                fail_count += 1
                continue
            
            if success:
                success_count += 1
                print(f"  ✓ Done!")
            else:
                fail_count += 1
                print(f"  ⚠ Not yet complete or download failed")

        except Exception as e:
            fail_count += 1
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 80)
    print(f"Re-download complete: {success_count} succeeded, {fail_count} failed")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Batch processing for GPT and Claude models")
    
    parser.add_argument("--domains", nargs="+", default=["rule-following"],
                        help="Domains to run (rule-following, safety, task-execution, persona)")
    parser.add_argument("--settings", nargs="+", default=["reference"],
                        help="Settings to run (reference, conflict)")
    parser.add_argument("--hierarchy", nargs="+", default=["sys-user", "user-tool", "sys-tool"],
                        help="Hierarchy types to run")
    parser.add_argument("--high_langs", nargs="+", default=["en", "de", "hi", "zh", "es", "fr"],
                        help="High-level languages")
    parser.add_argument("--low_langs", nargs="+", default=["en", "de", "hi", "zh", "es", "fr"],
                        help="Low-level languages")
    parser.add_argument("--models", nargs="+", default=["gpt-5", "gpt-5-mini"],
                        help="Models to run")
    
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Maximum output tokens")
    parser.add_argument("--check_interval", type=int, default=60,
                        help="Batch status polling interval (seconds)")
    parser.add_argument("--batch_dir", type=str, default="batch_jobs",
                        help="Directory to store batch job metadata")
    parser.add_argument("--redownload", action="store_true",
                        help="Re-download completed results from existing batch_jobs.json")
    
    args = parser.parse_args()
    
    if args.redownload:
        redownload_batch_results(args.batch_dir)
        return
    
    batch_jobs = []

    print("=" * 80)
    print("Creating batch jobs")
    print("=" * 80)

    for domain in args.domains:
        if domain not in DOMAIN_RUN_SCRIPT:
            print(f"✗ Unknown domain: {domain}")
            continue
        
        for model in args.models:
            if model not in MODEL_CONFIGS:
                print(f"✗ Unknown model: {model}")
                continue
            
            config = MODEL_CONFIGS[model]
            model_family = config["model_family"]
            model_path = config["model_path"]
            
            for setting, hier, high_lang, low_lang in itertools.product(
                args.settings, args.hierarchy, args.high_langs, args.low_langs
            ):
                input_file = get_input_file_path(domain, setting, hier, high_lang, low_lang)

                if not os.path.exists(input_file):
                    print(f"⚠ Input file not found: {input_file}")
                    continue

                output_dir = get_output_dir(domain, setting, hier, model_family, model,
                                            high_lang, low_lang, hier)

                request_file = f"{output_dir}/input_request.json"
                response_file = f"{output_dir}/input_response.json"

                # Skip if output already complete
                if os.path.exists(response_file):
                    try:
                        input_data = json.load(open(input_file, "r", encoding="utf-8"))
                        expected_count = len(input_data)
                        existing_results = json.load(open(response_file, "r", encoding="utf-8"))
                        actual_count = len(existing_results)

                        if actual_count == expected_count:
                            print(f"✓ Already complete ({actual_count}/{expected_count}): {domain}/{setting}/{hier}/{model}/{high_lang}_{low_lang}")
                            continue
                        else:
                            print(f"⚠ Incomplete results ({actual_count}/{expected_count}), re-running: {domain}/{setting}/{hier}/{model}/{high_lang}_{low_lang}")
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"⚠ Corrupted result file, re-running: {domain}/{setting}/{hier}/{model}/{high_lang}_{low_lang}")

                batch_requests, request_data = prepare_batch_requests(
                    input_file, model_family, model_path, request_file,
                    max_tokens=args.max_tokens
                )
                
                if not batch_requests:
                    print(f"⚠ No requests: {input_file}")
                    continue

                batch_file_path = f"{args.batch_dir}/{domain}/{setting}/{hier}/{model}/{high_lang}_{low_lang}.jsonl"
                create_batch_file(batch_requests, batch_file_path)

                description = f"{domain}/{setting}/{hier}/{model}/{high_lang}_{low_lang}"
                
                try:
                    if model_family == "gpt":
                        batch_id, file_id = submit_openai_batch(batch_file_path, description)
                    elif model_family == "claude":
                        batch_id = submit_claude_batch(batch_file_path, description)
                        file_id = None
                    
                    batch_jobs.append({
                        "batch_id": batch_id,
                        "file_id": file_id,
                        "model_family": model_family,
                        "description": description,
                        "output_dir": output_dir,
                        "request_file": request_file,
                        "response_file": f"{output_dir}/input_response.json",
                        "status": "submitted"
                    })
                    
                except Exception as e:
                    print(f"✗ Batch submission failed ({description}): {e}")
                    continue
    
    os.makedirs(args.batch_dir, exist_ok=True)
    batch_jobs_file = f"{args.batch_dir}/batch_jobs.json"
    with open(batch_jobs_file, "w", encoding="utf-8") as f:
        json.dump(batch_jobs, f, indent=4, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print(f"Submitted {len(batch_jobs)} batch jobs")
    print("=" * 80)

    print("\nMonitoring batch status (Ctrl+C to stop)")
    print(f"Poll interval: {args.check_interval}s\n")
    
    completed_jobs = set()
    
    try:
        while len(completed_jobs) < len(batch_jobs):
            print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking batch status...")

            for i, job in enumerate(batch_jobs):
                if i in completed_jobs:
                    continue

                batch_id = job["batch_id"]
                model_family = job["model_family"]
                description = job["description"]

                try:
                    if model_family == "gpt":
                        status, completed, total = check_openai_batch_status(batch_id)
                    elif model_family == "claude":
                        status, completed, total = check_claude_batch_status(batch_id)

                    print(f"  [{i+1}/{len(batch_jobs)}] {description}")
                    print(f"      status: {status}, progress: {completed}/{total}")

                    if status in ["completed", "ended"]:
                        if model_family == "gpt":
                            success = download_openai_batch_results(batch_id, job["response_file"], job["request_file"])
                        elif model_family == "claude":
                            success = download_claude_batch_results(batch_id, job["response_file"], job["request_file"])

                        if success:
                            completed_jobs.add(i)
                            job["status"] = "completed"
                            print(f"      ✓ Done!")

                except Exception as e:
                    print(f"  [{i+1}/{len(batch_jobs)}] {description}")
                    print(f"      ✗ Status check failed: {e}")

            with open(batch_jobs_file, "w", encoding="utf-8") as f:
                json.dump(batch_jobs, f, indent=4, ensure_ascii=False)

            print(f"\nProgress: {len(completed_jobs)}/{len(batch_jobs)} complete")

            if len(completed_jobs) < len(batch_jobs):
                print(f"Checking again in {args.check_interval}s...")
                time.sleep(args.check_interval)

    except KeyboardInterrupt:
        print("\n\nBatch monitoring interrupted.")
        print(f"Progress saved to {batch_jobs_file}.")
        print("Re-run to continue monitoring.")
        sys.exit(0)
    
    print("\n" + "=" * 80)
    print("All batch jobs complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
