"""
Generate text using open-source model with vLLM API server (OpenAI-compatible)
"""

import os
import sys
import json
import time
import argparse
from typing import Dict
from openai import OpenAI


def prepare_tool_call_qwen(tool: Dict):
    """Prepare tool call for Qwen models
    
    Args:
        tool: Tool dictionary with definition, call, and return
    """
    tool_definition = {
        "type": "function",
        "function": {
            "name": tool['definition']['name'],
            "description": tool['definition']['description'],
            "parameters": tool['definition']['parameters']
        }
    }
    
    tool_call = {
        'content': None,
        'refusal': None,
        'role': 'assistant',
        'annotations': None,
        'audio': None,
        'function_call': None,
        'tool_calls': [
            {
                'id': tool['call']['id'],
                'function': {
                    'arguments': json.dumps(tool['call']['arguments'], ensure_ascii=False),
                    'name': tool['call']['name']
                },
                'type': 'function'
            }
        ],
        'reasoning': None,
        'reasoning_content': None
    }
    
    
    tool_output = {
        'role': 'tool',
        'tool_call_id': tool['call']['id'],
        'content': json.dumps({"content": tool['return']['content']}, ensure_ascii=False)
    }
    
    return tool_definition, tool_call, tool_output


def prepare_tool_call_mistral(tool: Dict):
    """Prepare tool call for Mistral models"""
    tool_definition = {
        "type": "function",
        "function": {
            "name": tool['definition']['name'],
            "description": tool['definition']['description'],
            "parameters": tool['definition']['parameters']
        }
    }
    
    tool_call = {
        'role': 'assistant',
        'tool_calls': [
            {
                'id': tool['call']['id'],
                'function': {
                    'arguments': json.dumps(tool['call']['arguments'], ensure_ascii=False),
                    'name': tool['call']['name']
                },
                'type': 'function'
            }
        ],
    }
    
    tool_output = {
        'role': 'tool',
        'tool_call_id': tool['return']['id'],
        'name': tool['return']['name'],
        'content': tool['return']['content']
    }
    
    return tool_definition, tool_call, tool_output


def prepare_tool_call_llama(tool: Dict):
    """Prepare tool call for Mistral models"""
    tool_definition = {
        "type": "function",
        "function": {
            "name": tool['definition']['name'],
            "description": tool['definition']['description'],
            "parameters": tool['definition']['parameters']
        }
    }
    
    tool_call = {
        'role': 'assistant',
        'tool_calls': [
            {
                'id': tool['call']['id'],
                'function': {
                    'arguments': json.dumps(tool['call']['arguments'], ensure_ascii=False),
                    'name': tool['call']['name']
                },
                'type': 'function'
            }
        ],
    }
    
    tool_output = {
        'role': 'tool',
        'tool_call_id': tool['return']['id'],
        'name': tool['return']['name'],
        'content': tool['return']['content']
    }
    
    return tool_definition, tool_call, tool_output



def main(args):
    # Load input data (request file)
    data = json.load(open(args.input, "r", encoding="utf-8"))
    print(f"Loaded {len(data)} data instances from {args.input}")
    
    # Determine expected count from benchmark file if provided
    if hasattr(args, 'benchmark_file') and args.benchmark_file:
        try:
            benchmark_data = json.load(open(args.benchmark_file, "r", encoding="utf-8"))
            expected_count = len(benchmark_data)
            print(f"Benchmark file has {expected_count} examples")
        except Exception as e:
            print(f"Warning: Could not load benchmark file {args.benchmark_file}: {e}")
            expected_count = len(data)
    else:
        expected_count = len(data)
    
    # Check if output file exists and is complete
    if os.path.exists(args.output):
        try:
            existing_results = json.load(open(args.output, "r", encoding="utf-8"))
            actual_count = len(existing_results)
            
            if actual_count == expected_count:
                print(f"Output file {args.output} already exists with {actual_count}/{expected_count} examples, skipping inference...")
                return
            else:
                print(f"Output file {args.output} exists but incomplete ({actual_count}/{expected_count} examples), re-running inference...")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Output file {args.output} exists but is corrupted ({e}), re-running inference...")



    # Determine model family for tool call preparation
    model_family = args.model_family.lower() if args.model_family else "qwen"
    print(f"Using model family: {model_family}")
    
    # Select appropriate tool call preparation function
    if model_family == "mistral":
        prepare_tool_call = prepare_tool_call_mistral
    elif model_family == "llama":
        prepare_tool_call = prepare_tool_call_llama
    else:  # default to qwen
        prepare_tool_call = prepare_tool_call_qwen

    # Initialize OpenAI client for vLLM API server
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{args.port}/v1"
    )

    print(f"Using vLLM API server at http://localhost:{args.port}/v1")

    results = []
    
    for idx, example in enumerate(data):
        messages = example['messages'].copy()
        tools = None
        
        # Prepare tool if exists
        if "tool" in example.keys():

            tool_definition, tool_call, tool_output = prepare_tool_call(example["tool"])
            tools = [tool_definition]
            messages.append(tool_call)
            messages.append(tool_output)

        # Show example input for first instance
        if idx == 0:
            print("*" * 15 + " Example input " + "*" * 15)
            print(json.dumps(messages, indent=2, ensure_ascii=False))
            if tools:
                print("Tools:")
                print(json.dumps(tools, indent=2, ensure_ascii=False))
            print("*" * 45)

        # Call API
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                tools=tools,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                tool_choice="none"
            )
            
            output_text = response.choices[0].message.content
            
            # Construct input prompt (for compatibility with original format)
            # We'll try to reconstruct what the tokenizer would have produced
            input_prompt = json.dumps(messages, ensure_ascii=False)
            
        except Exception as e:
            print(f"Error processing example {example['id']}: {e}")
            output_text = f"ERROR: {str(e)}"
            input_prompt = json.dumps(messages, ensure_ascii=False)

        results.append({
            "id": example["id"],
            "input": input_prompt,
            "generator": args.model,
            "output": output_text,
        })

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(data)} examples")

    # Sort results by id if possible
    try:
        results = sorted(results, key=lambda x: x["id"])
    except Exception as e:
        print(f'Skip sorting results due to error: {e}')

    # Save results
    json.dump(
        results,
        open(args.output, "w", encoding="utf8"),
        indent=4,
        ensure_ascii=False,
    )
    print(f"Saved {len(results)} examples to {args.output}.")

    print("*" * 15 + " Example output " + "*" * 15)
    print(results[0])
    print("*" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, required=True)
    parser.add_argument("-model_family", type=str, default="qwen", choices=["qwen", "mistral", "llama"])
    parser.add_argument("-input", type=str, required=True)
    parser.add_argument("-output", type=str, required=True)
    parser.add_argument("-benchmark_file", type=str, default=None, help="Original benchmark file to compare count")
    parser.add_argument("-max_new_tokens", type=int, default=100)
    parser.add_argument("-temperature", type=float, default=0.0)
    parser.add_argument("-top_k", type=int, default=250)
    parser.add_argument("-top_p", type=float, default=1.0)
    parser.add_argument("-port", type=int, default=8000, help="vLLM server port")
    args = parser.parse_args()

    main(args)
