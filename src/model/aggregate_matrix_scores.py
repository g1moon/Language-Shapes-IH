"""
Aggregate evaluation results into 6x6 language matrices.
Generates reference.csv, conflict.csv, and hcr.csv for each hierarchy type.
HCR (Hierarchy Compliance Rate) = conflict / reference.
Supports both rule-following and task-execution domains.
"""

import os
import json
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np


def get_file_hash(filepath):
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def record_rule_following_single_file(data):
    """Calculate prompt and instruction accuracy from rule-following eval results."""
    prompt_total, instruction_total = 0, 0
    prompt_correct, instruction_correct = 0, 0

    for example in data:
        follow_instruction_list = example["follow_instruction_list"]
        instruction_id_list = example["instruction_id_list"]

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)

    prompt_accuracy = prompt_correct / prompt_total if prompt_total > 0 else 0
    instruction_accuracy = instruction_correct / instruction_total if instruction_total > 0 else 0

    # Return average of prompt and instruction accuracy
    return (prompt_accuracy + instruction_accuracy) / 2


def record_task_execution_single_file(data):
    """Calculate average SacreBLEU score from task-execution eval results."""
    scores = []
    for example in data:
        if "score" in example:
            scores.append(example["score"])
    
    return sum(scores) / len(scores) if scores else 0


def record_safety_single_file(data):
    """Calculate average safety accuracy from safety eval results."""
    scores = []
    for example in data:
        if "score" in example:
            scores.append(example["score"])

    return sum(scores) / len(scores) if scores else 0


def record_persona_single_file(data):
    """Calculate average persona accuracy from persona eval results.
    Excludes instances with null scores (failed verdict extraction) from denominator."""
    scores = [ex["score"] for ex in data if ex.get("score") is not None]
    return sum(scores) / len(scores) if scores else 0


def aggregate_hierarchy_scores(results_dir, model_family, model_name, hierarchy_type, eval_type, domain='rule-following', strictness='both'):
    """
    Aggregate scores for a specific hierarchy type (sys-tool, sys-user, user-tool).
    
    Args:
        results_dir: Base results directory
        model_family: Model family (e.g., 'qwen')
        model_name: Model name (e.g., 'qwen3-4b')
        hierarchy_type: One of 'sys-tool', 'sys-user', 'user-tool'
        eval_type: 'reference' or 'conflict'
        domain: 'rule-following' or 'task-execution'
        strictness: 'strict', 'loose', or 'both' (returns both matrices separately)
    
    Returns:
        If strictness='both': (strict_matrix, loose_matrix, processed_files)
        Otherwise: (matrix, processed_files)
    """
    languages = ['en', 'de', 'hi', 'zh', 'es', 'fr']
    strict_matrix = np.zeros((6, 6))
    loose_matrix = np.zeros((6, 6))
    processed_files = []
    
    # Determine hierarchy order (higher vs lower)
    if hierarchy_type == 'sys-tool':
        high_key, low_key = 'sys', 'tool'
    elif hierarchy_type == 'sys-user':
        high_key, low_key = 'sys', 'user'
    elif hierarchy_type == 'user-tool':
        high_key, low_key = 'user', 'tool'
    else:
        raise ValueError(f"Unknown hierarchy type: {hierarchy_type}")
    
    # Construct base path
    base_path = Path(results_dir) / domain / eval_type / hierarchy_type / model_family / model_name
    
    # Select the appropriate score calculation function
    if domain == 'rule-following':
        score_func = record_rule_following_single_file
    elif domain == 'task-execution':
        score_func = record_task_execution_single_file
    elif domain == 'safety':
        score_func = record_safety_single_file
    elif domain == 'persona':
        score_func = record_persona_single_file
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    # Domains that use a single eval_results.json (no strict/loose split)
    single_file_domains = {'safety', 'task-execution', 'persona'}

    # Iterate through all language combinations
    for i, high_lang in enumerate(languages):  # x-axis (higher hierarchy)
        for j, low_lang in enumerate(languages):  # y-axis (lower hierarchy)

            # Both reference and conflict have language combinations
            folder_name = f"input-{high_key}_{high_lang}-{low_key}_{low_lang}"
            eval_dir = base_path / folder_name

            if domain in single_file_domains:
                # Single eval_results.json for safety and task-execution
                eval_file = eval_dir / 'eval_results.json'

                if eval_file.exists():
                    try:
                        with open(eval_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        score = score_func(data)
                        strict_matrix[j, i] = score
                        loose_matrix[j, i] = score

                        processed_files.append({
                            'path': str(eval_file),
                            'hash': get_file_hash(eval_file)
                        })
                    except Exception as e:
                        print(f"Warning: Error processing {eval_dir}: {e}")
                        strict_matrix[j, i] = 0.0
                        loose_matrix[j, i] = 0.0
                else:
                    print(f"Warning: File not found at {eval_file}")
                    strict_matrix[j, i] = 0.0
                    loose_matrix[j, i] = 0.0
            else:
                # Strict/loose split for rule-following and persona
                strict_file = eval_dir / 'eval_results_strict.json'
                loose_file = eval_dir / 'eval_results_loose.json'

                if strict_file.exists() and loose_file.exists():
                    try:
                        with open(strict_file, 'r', encoding='utf-8') as f:
                            strict_data = json.load(f)

                        with open(loose_file, 'r', encoding='utf-8') as f:
                            loose_data = json.load(f)

                        strict_score = score_func(strict_data)
                        loose_score = score_func(loose_data)

                        strict_matrix[j, i] = strict_score
                        loose_matrix[j, i] = loose_score

                        processed_files.append({
                            'path': str(strict_file),
                            'hash': get_file_hash(strict_file)
                        })
                        processed_files.append({
                            'path': str(loose_file),
                            'hash': get_file_hash(loose_file)
                        })
                    except Exception as e:
                        print(f"Warning: Error processing {eval_dir}: {e}")
                        strict_matrix[j, i] = 0.0
                        loose_matrix[j, i] = 0.0
                else:
                    print(f"Warning: Files not found at {eval_dir}")
                    strict_matrix[j, i] = 0.0
                    loose_matrix[j, i] = 0.0

    if strictness == 'both':
        return strict_matrix, loose_matrix, processed_files
    elif strictness == 'strict':
        return strict_matrix, processed_files
    elif strictness == 'loose':
        return loose_matrix, processed_files
    else:
        raise ValueError(f"Unknown strictness: {strictness}")


def save_matrix_csv(matrix, languages, output_path, hierarchy_type=None):
    """Save matrix as CSV with language labels."""
    df = pd.DataFrame(matrix, index=languages, columns=languages)
    
    # Set index name based on hierarchy type
    if hierarchy_type == 'sys-tool':
        df.index.name = 'tool\\sys'
    elif hierarchy_type == 'sys-user':
        df.index.name = 'user\\sys'
    elif hierarchy_type == 'user-tool':
        df.index.name = 'tool\\user'
    else:
        df.index.name = 'low\\high'  # fallback for unknown types
    
    df.to_csv(output_path, float_format='%.4f')
    print(f"Saved matrix to {output_path}")


def save_manifest(processed_files, output_path):
    """Save manifest with processed file information."""
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'total_files': len(processed_files),
        'files': processed_files
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved manifest to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate evaluation results into language matrices')
    parser.add_argument('-results_dir', type=str, default='results', help='Base results directory')
    parser.add_argument('-model_family', type=str, required=True, help='Model family (e.g., qwen)')
    parser.add_argument('-model_name', type=str, required=True, help='Model name (e.g., qwen3-4b)')
    parser.add_argument('-output_dir', type=str, default='model-scores', help='Output directory for scores')
    parser.add_argument('-domain', type=str, default='rule-following', choices=['rule-following', 'task-execution', 'safety', 'persona'],
                        help='Domain to process (rule-following, task-execution, safety, or persona)')
    args = parser.parse_args()
    
    languages = ['en', 'de', 'hi', 'zh', 'es', 'fr']
    hierarchy_types = ['sys-tool', 'sys-user', 'user-tool']
    
    # Create output directory structure
    model_output_dir = Path(args.output_dir) / args.model_family / args.model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    all_processed_files = []
    
    # Process each hierarchy type
    for hierarchy_type in hierarchy_types:
        print(f"\n{'='*60}")
        print(f"Processing hierarchy: {hierarchy_type} (domain: {args.domain})")
        print(f"{'='*60}")
        
        # Create subdirectory for this hierarchy
        hierarchy_output_dir = model_output_dir / 'results' / args.domain / hierarchy_type
        hierarchy_output_dir.mkdir(parents=True, exist_ok=True)

        # Domains without strict/loose distinction
        single_file_domains = {'safety', 'task-execution', 'persona'}
        has_strict_loose = args.domain not in single_file_domains

        if has_strict_loose:
            # Create strict and loose subdirectories
            strict_output_dir = hierarchy_output_dir / 'strict'
            loose_output_dir = hierarchy_output_dir / 'loose'
            strict_output_dir.mkdir(parents=True, exist_ok=True)
            loose_output_dir.mkdir(parents=True, exist_ok=True)

            # Aggregate reference scores (both strict and loose)
            print(f"\nAggregating reference scores...")
            ref_strict_matrix, ref_loose_matrix, ref_files = aggregate_hierarchy_scores(
                args.results_dir, args.model_family, args.model_name,
                hierarchy_type, 'reference', domain=args.domain, strictness='both'
            )

            # Aggregate conflict scores (both strict and loose)
            print(f"\nAggregating conflict scores...")
            conflict_strict_matrix, conflict_loose_matrix, conflict_files = aggregate_hierarchy_scores(
                args.results_dir, args.model_family, args.model_name,
                hierarchy_type, 'conflict', domain=args.domain, strictness='both'
            )

            # Calculate HCR (Hierarchy Compliance Rate) for strict and loose
            hcr_strict_matrix = np.where(ref_strict_matrix > 0, conflict_strict_matrix / ref_strict_matrix, 0.0)
            hcr_loose_matrix = np.where(ref_loose_matrix > 0, conflict_loose_matrix / ref_loose_matrix, 0.0)

            # Save strict matrices
            print(f"\nSaving strict matrices...")
            save_matrix_csv(ref_strict_matrix, languages, strict_output_dir / 'reference.csv', hierarchy_type)
            save_matrix_csv(conflict_strict_matrix, languages, strict_output_dir / 'conflict.csv', hierarchy_type)
            save_matrix_csv(hcr_strict_matrix, languages, strict_output_dir / 'hcr.csv', hierarchy_type)

            # Save loose matrices
            print(f"\nSaving loose matrices...")
            save_matrix_csv(ref_loose_matrix, languages, loose_output_dir / 'reference.csv', hierarchy_type)
            save_matrix_csv(conflict_loose_matrix, languages, loose_output_dir / 'conflict.csv', hierarchy_type)
            save_matrix_csv(hcr_loose_matrix, languages, loose_output_dir / 'hcr.csv', hierarchy_type)

            # Calculate and save average matrices (strict + loose) / 2
            print(f"\nSaving average matrices...")
            ref_avg_matrix = (ref_strict_matrix + ref_loose_matrix) / 2
            conflict_avg_matrix = (conflict_strict_matrix + conflict_loose_matrix) / 2
            hcr_avg_matrix = np.where(ref_avg_matrix > 0, conflict_avg_matrix / ref_avg_matrix, 0.0)

            save_matrix_csv(ref_avg_matrix, languages, hierarchy_output_dir / 'reference.csv', hierarchy_type)
            save_matrix_csv(conflict_avg_matrix, languages, hierarchy_output_dir / 'conflict.csv', hierarchy_type)
            save_matrix_csv(hcr_avg_matrix, languages, hierarchy_output_dir / 'hcr.csv', hierarchy_type)
        else:
            # Single-file domains: no strict/loose split
            print(f"\nAggregating reference scores...")
            ref_matrix, ref_files = aggregate_hierarchy_scores(
                args.results_dir, args.model_family, args.model_name,
                hierarchy_type, 'reference', domain=args.domain, strictness='strict'
            )

            print(f"\nAggregating conflict scores...")
            conflict_matrix, conflict_files = aggregate_hierarchy_scores(
                args.results_dir, args.model_family, args.model_name,
                hierarchy_type, 'conflict', domain=args.domain, strictness='strict'
            )

            if args.domain == 'safety':
                # For safety, reference only depends on the upper (system) language.
                # The lower-hierarchy language has no effect on the reference score,
                # so we broadcast the diagonal (same-language) reference to all rows.
                ref_diag = np.diag(ref_matrix)
                ref_for_hcr = np.tile(ref_diag, (len(languages), 1))
            else:
                ref_for_hcr = ref_matrix

            hcr_matrix = np.where(ref_for_hcr > 0, conflict_matrix / ref_for_hcr, 0.0)

            print(f"\nSaving matrices...")
            save_matrix_csv(ref_matrix, languages, hierarchy_output_dir / 'reference.csv', hierarchy_type)
            save_matrix_csv(conflict_matrix, languages, hierarchy_output_dir / 'conflict.csv', hierarchy_type)
            save_matrix_csv(hcr_matrix, languages, hierarchy_output_dir / 'hcr.csv', hierarchy_type)
        
        # Collect all processed files
        all_processed_files.extend(ref_files)
        all_processed_files.extend(conflict_files)
    
    # Save manifest for the model
    save_manifest(all_processed_files, model_output_dir / 'manifest.json')
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully processed all hierarchies for {args.model_name} ({args.domain})")
    print(f"✓ Output directory: {model_output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
