import json
import argparse
import os
import shutil
import datetime
import time
import pdb
from typing import Dict
from copy import deepcopy
from tqdm import tqdm
import concurrent.futures
from openai import OpenAI
from openai import BadRequestError
from transformers import AutoTokenizer

# Claude API support
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic package not installed. Claude API will not be available.")



def tool_call_openai(tool: Dict):
    # Edit for gpt-5
    """
    Convert the tool call to OpenAI format.
    """
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

    reasoning_item={
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
    """
    Convert the tool call to Claude format.
    """
    raw_definition = tool["definition"]
    raw_tool_call = tool["call"]
    raw_tool_return = tool["return"]

    # Claude tool definition format
    definition = [{
        "name": raw_definition['name'],
        "description": raw_definition['description'],
        "input_schema": raw_definition['parameters']
    }]

    # Claude tool call format
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

    # Claude tool output format
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


def convert_jsonl_to_json(file_path) -> None:
    """Convert a jsonl file to a json file."""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist. Creating empty file.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump([], f, indent=4, ensure_ascii=False)
        return
    
    if os.path.getsize(file_path) == 0:
        print(f"Warning: {file_path} is empty. Creating empty JSON file.")
        with open(file_path, "w") as f:
            json.dump([], f, indent=4, ensure_ascii=False)
        return
    
    with open(file_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == '[':
            try:
                data = json.load(f)
                with open(file_path, "w", encoding="utf-8") as f_out:
                    json.dump(data, f_out, indent=4, ensure_ascii=False)
                return
            except json.decoder.JSONDecodeError:
                pass
    
    shutil.copy(file_path, file_path + ".bak")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            if not lines:
                data = []
            else:
                data = [json.loads(line) for line in lines]
    except Exception as e:
        print(f"Error reading JSONL file {file_path}: {e}")
        if os.path.exists(file_path + ".bak"):
            shutil.copy(file_path + ".bak", file_path)
        data = []
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    if os.path.exists(file_path + ".bak"):
        os.remove(file_path + ".bak")


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")

    
def save_list_as_jsonl(path: str, data):
    with open(path, "w", encoding="utf8") as fout:
        for instance in data:
            fout.write(json.dumps(instance))
            fout.write("\n")
    print(f"Saved {len(data)} data to {path}")


def call_openai(l_data):

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = deepcopy(l_data["messages"])
    if "tool" in l_data.keys():
        tool_definition, reasoning_item, tool_call, tool_return = tool_call_openai(l_data["tool"])
        messages.extend([reasoning_item, tool_call, tool_return])

    if "system" in l_data.keys() and l_data["system"] is not None:
        messages.insert(0, {"role": "developer", "content": l_data["system"]})

    retries = 0
    while retries < args.max_retries:

        try:
            response = client.responses.create(
                model=args.model,
                input=messages,
                reasoning={"effort": "low"},
                text={
                    "verbosity": "low"
                },
                max_output_tokens=args.max_new_tokens,
            )
            response_body = response.output_text
            append_to_jsonl(
                {
                    "id": l_data["id"],
                    "input": messages,
                    "output": response_body,
                },
                args.output,
            )
            p_bar.update(1)
            break

        except Exception as e:

            # For safety tasks, sometimes openai rejects the request due to repetitive patterns in the attack
            # we should identify such as a successful defense by outputting an empty string (which would be considered as a success)
            if isinstance(e, BadRequestError) and "repetitive patterns in your prompt" in e.body['message']:
                append_to_jsonl(
                    {
                        "id": l_data["id"],
                        "input": messages,
                        "output": "",
                    },
                    args.output,
                )
                p_bar.update(1)
                break

            retries += 1
            print(
                f"Error on call attempt {retries}: {e}"
            )  # If we've reached the maximum number of retries, record an error message (or handle as desired)
            if retries == args.max_retries:
                print(f"Retry exceed the max_retries {retries} times.")
                break
            time.sleep(args.sleep)


def call_claude(l_data):
    """Call Claude API for a single data instance"""
    if not ANTHROPIC_AVAILABLE:
        raise ImportError("anthropic package is required for Claude API")
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    messages = deepcopy(l_data["messages"])
    system_prompt = None
    tools = None
    
    if "system" in l_data.keys() and l_data["system"] is not None:
        system_prompt = l_data["system"]

    if "tool" in l_data.keys():
        tool_definition, tool_call, tool_output = tool_call_claude(l_data["tool"])
        tools = tool_definition
        messages.append(tool_call)
        messages.append(tool_output)
    
    retries = 0
    while retries < args.max_retries:
        try:
                response = client.messages.create(
                model=args.model,
                system=system_prompt,
                messages=messages,
                max_tokens=args.max_new_tokens,
            )
            
            response_body = ""
            for content in response.content:
                if hasattr(content, 'text'):
                    response_body += content.text
            
            append_to_jsonl(
                {
                    "id": l_data["id"],
                    "input": messages,
                    "output": response_body,
                },
                args.output,
            )
            p_bar.update(1)
            break
            
        except Exception as e:
            retries += 1
            print(f"Error on Claude API call attempt {retries}: {e}")
            
            if retries == args.max_retries:
                print(f"Retry exceed the max_retries {retries} times.")
                break
            time.sleep(args.sleep)

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-model", default="")
    parser.add_argument("-input", default=None)
    parser.add_argument("-output", default=None)
    parser.add_argument("-benchmark_file", type=str, default=None, help="Original benchmark file to compare count")
    parser.add_argument("-max_new_tokens", type=int, default=2048)
    parser.add_argument("-max_loop", type=int, default=1)
    parser.add_argument("-max_retries", type=int, default=20)
    parser.add_argument("-max_threads", type=int, default=8)
    parser.add_argument("-sleep", type=int, default=2)
    args = parser.parse_args()

    is_claude = "claude" in args.model.lower()
    is_gpt = "gpt" in args.model.lower()
    
    if is_claude and not ANTHROPIC_AVAILABLE:
        raise ImportError("anthropic package is required for Claude models. Install it with: pip install anthropic")
    
    if is_claude:
        api_call_func = call_claude
    elif is_gpt:
        api_call_func = call_openai
    else:
        raise ValueError(f"Unknown model type: {args.model}. Supported models: GPT, Claude")
    
    print(f"Using API call function for: {'Claude' if is_claude else 'GPT'}")

    # Input is a JSON file
    requests = json.load(open(args.input, "r", encoding="utf-8"))
    
    # Determine expected count from benchmark file if provided
    if args.benchmark_file:
        try:
            benchmark_data = json.load(open(args.benchmark_file, "r", encoding="utf-8"))
            expected_total = len(benchmark_data)
            print(f"Benchmark file has {expected_total} examples")
        except Exception as e:
            print(f"Warning: Could not load benchmark file {args.benchmark_file}: {e}")
            expected_total = len(requests)
    else:
        expected_total = len(requests)
    
    tstart = datetime.datetime.now()
    for _ in range(args.max_loop):

        try:
            exist_ids = set()
            if os.path.isfile(args.output):
                with open(args.output) as f:
                    for line in f.readlines():
                        l_data = json.loads(line)
                        exist_ids.add(l_data["id"])
        # If inference has finished and the output has been converted to JSON, this will cause an error
        except json.decoder.JSONDecodeError:
            existing_data = json.load(open(args.output, "r", encoding="utf-8"))
            exist_ids = {l_data["id"] for l_data in existing_data}
            save_list_as_jsonl(args.output, existing_data)  # convert back to JSONL, otherwise there will be errors

        dataset = []
        # Each data instance should have "id", "messages", and optionally "system"
        for l_data in requests:
            if l_data["id"] not in exist_ids:
                dataset.append(l_data)
        print("EXISTING DATA #: ", len(exist_ids))
        print("API Call REQUEST #: ", len(dataset))
        print(f"Total expected: {expected_total} (existing: {len(exist_ids)}, remaining: {len(dataset)})")
        
        # Check if we have the expected number of results
        if len(exist_ids) >= expected_total:
            print(f"All data instances have been processed! ({len(exist_ids)}/{expected_total}) Skipping...")
            break
        
        if len(dataset) == 0:
            print(f"Warning: No remaining data in request file, but only {len(exist_ids)}/{expected_total} completed")
            break

        p_bar = tqdm(range(len(dataset)))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_threads
        ) as executor:
            executor.map(api_call_func, dataset)

    tend = datetime.datetime.now()
    ttime = tend - tstart
    print("Total time for {} calls: {}".format(len(dataset), ttime))
    # Convert the final output to JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    convert_jsonl_to_json(args.output)
