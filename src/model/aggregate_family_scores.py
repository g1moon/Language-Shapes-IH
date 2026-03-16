"""
Aggregate scores across all models in a family.
Calculates average matrices for family_results.
HCR (Hierarchy Compliance Rate) = conflict / reference.
Supports both rule-following and task-execution domains.
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


def load_model_matrix(csv_path):
    """Load a matrix CSV file."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, index_col=0)
    return df.values


def save_manifest(model_list, output_path):
    """Save manifest with list of models included."""
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'models': model_list,
        'total_models': len(model_list)
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved family manifest to {output_path}")


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
    print(f"Saved family matrix to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate scores across all models in a family')
    parser.add_argument('-model_family', type=str, required=True, help='Model family (e.g., qwen)')
    parser.add_argument('-output_dir', type=str, default='model-scores', help='Base output directory')
    parser.add_argument('-domain', type=str, default='rule-following', choices=['rule-following', 'task-execution', 'safety', 'persona'],
                        help='Domain to process (rule-following, task-execution, safety, or persona)')
    args = parser.parse_args()
    
    languages = ['en', 'de', 'hi', 'zh', 'es', 'fr']
    hierarchy_types = ['sys-tool', 'sys-user', 'user-tool']
    
    family_dir = Path(args.output_dir) / args.model_family
    
    if not family_dir.exists():
        print(f"Error: Family directory not found: {family_dir}")
        return
    
    # Find all model subdirectories
    model_dirs = [d for d in family_dir.iterdir() if d.is_dir() and d.name != 'family_results']
    
    if not model_dirs:
        print(f"Error: No model directories found in {family_dir}")
        return
    
    model_names = [d.name for d in model_dirs]
    print(f"\nFound {len(model_names)} models in {args.model_family} family:")
    for name in model_names:
        print(f"  - {name}")
    
    # Create family_results directory
    family_results_dir = family_dir / 'family_results'
    family_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Domains without strict/loose distinction
    single_file_domains = {'safety', 'task-execution', 'persona'}
    has_strict_loose = args.domain not in single_file_domains

    # Process each hierarchy type
    for hierarchy_type in hierarchy_types:
        print(f"\n{'='*60}")
        print(f"Processing hierarchy: {hierarchy_type} (domain: {args.domain})")
        print(f"{'='*60}")

        # Create subdirectory for this hierarchy
        hierarchy_output_dir = family_results_dir / args.domain / hierarchy_type
        hierarchy_output_dir.mkdir(parents=True, exist_ok=True)

        if has_strict_loose:
            # Create strict and loose subdirectories
            strict_output_dir = hierarchy_output_dir / 'strict'
            loose_output_dir = hierarchy_output_dir / 'loose'
            strict_output_dir.mkdir(parents=True, exist_ok=True)
            loose_output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize matrices for averaging
            strict_ref_matrices = []
            strict_conflict_matrices = []
            strict_hcr_matrices = []

            loose_ref_matrices = []
            loose_conflict_matrices = []
            loose_hcr_matrices = []

            # Load matrices from all models
            for model_dir in model_dirs:
                hierarchy_dir = model_dir / 'results' / args.domain / hierarchy_type

                # Load strict matrices
                strict_ref_matrix = load_model_matrix(hierarchy_dir / 'strict' / 'reference.csv')
                strict_conflict_matrix = load_model_matrix(hierarchy_dir / 'strict' / 'conflict.csv')
                strict_hcr_matrix = load_model_matrix(hierarchy_dir / 'strict' / 'hcr.csv')

                if strict_ref_matrix is not None:
                    strict_ref_matrices.append(strict_ref_matrix)
                else:
                    print(f"Warning: Strict reference matrix not found for {model_dir.name}")

                if strict_conflict_matrix is not None:
                    strict_conflict_matrices.append(strict_conflict_matrix)
                else:
                    print(f"Warning: Strict conflict matrix not found for {model_dir.name}")

                if strict_hcr_matrix is not None:
                    strict_hcr_matrices.append(strict_hcr_matrix)
                else:
                    print(f"Warning: Strict hcr matrix not found for {model_dir.name}")

                # Load loose matrices
                loose_ref_matrix = load_model_matrix(hierarchy_dir / 'loose' / 'reference.csv')
                loose_conflict_matrix = load_model_matrix(hierarchy_dir / 'loose' / 'conflict.csv')
                loose_hcr_matrix = load_model_matrix(hierarchy_dir / 'loose' / 'hcr.csv')

                if loose_ref_matrix is not None:
                    loose_ref_matrices.append(loose_ref_matrix)
                else:
                    print(f"Warning: Loose reference matrix not found for {model_dir.name}")

                if loose_conflict_matrix is not None:
                    loose_conflict_matrices.append(loose_conflict_matrix)
                else:
                    print(f"Warning: Loose conflict matrix not found for {model_dir.name}")

                if loose_hcr_matrix is not None:
                    loose_hcr_matrices.append(loose_hcr_matrix)
                else:
                    print(f"Warning: Loose hcr matrix not found for {model_dir.name}")

            # Calculate and save strict averages
            print(f"\nSaving strict family averages...")
            if strict_ref_matrices:
                avg_strict_ref_matrix = np.mean(strict_ref_matrices, axis=0)
                save_matrix_csv(avg_strict_ref_matrix, languages, strict_output_dir / 'reference.csv', hierarchy_type)

            if strict_conflict_matrices:
                avg_strict_conflict_matrix = np.mean(strict_conflict_matrices, axis=0)
                save_matrix_csv(avg_strict_conflict_matrix, languages, strict_output_dir / 'conflict.csv', hierarchy_type)

            if strict_hcr_matrices:
                avg_strict_hcr_matrix = np.mean(strict_hcr_matrices, axis=0)
                save_matrix_csv(avg_strict_hcr_matrix, languages, strict_output_dir / 'hcr.csv', hierarchy_type)

            # Calculate and save loose averages
            print(f"\nSaving loose family averages...")
            if loose_ref_matrices:
                avg_loose_ref_matrix = np.mean(loose_ref_matrices, axis=0)
                save_matrix_csv(avg_loose_ref_matrix, languages, loose_output_dir / 'reference.csv', hierarchy_type)

            if loose_conflict_matrices:
                avg_loose_conflict_matrix = np.mean(loose_conflict_matrices, axis=0)
                save_matrix_csv(avg_loose_conflict_matrix, languages, loose_output_dir / 'conflict.csv', hierarchy_type)

            if loose_hcr_matrices:
                avg_loose_hcr_matrix = np.mean(loose_hcr_matrices, axis=0)
                save_matrix_csv(avg_loose_hcr_matrix, languages, loose_output_dir / 'hcr.csv', hierarchy_type)

            # Calculate and save overall averages (strict + loose) / 2
            print(f"\nSaving overall family averages...")
            if strict_ref_matrices and loose_ref_matrices:
                overall_ref_matrix = (avg_strict_ref_matrix + avg_loose_ref_matrix) / 2
                save_matrix_csv(overall_ref_matrix, languages, hierarchy_output_dir / 'reference.csv', hierarchy_type)

            if strict_conflict_matrices and loose_conflict_matrices:
                overall_conflict_matrix = (avg_strict_conflict_matrix + avg_loose_conflict_matrix) / 2
                save_matrix_csv(overall_conflict_matrix, languages, hierarchy_output_dir / 'conflict.csv', hierarchy_type)

            if strict_hcr_matrices and loose_hcr_matrices:
                overall_hcr_matrix = (avg_strict_hcr_matrix + avg_loose_hcr_matrix) / 2
                save_matrix_csv(overall_hcr_matrix, languages, hierarchy_output_dir / 'hcr.csv', hierarchy_type)

        else:
            # Single-file domains: no strict/loose split
            ref_matrices = []
            conflict_matrices = []
            hcr_matrices = []

            for model_dir in model_dirs:
                hierarchy_dir = model_dir / 'results' / args.domain / hierarchy_type

                ref_matrix = load_model_matrix(hierarchy_dir / 'reference.csv')
                conflict_matrix = load_model_matrix(hierarchy_dir / 'conflict.csv')
                hcr_matrix = load_model_matrix(hierarchy_dir / 'hcr.csv')

                if ref_matrix is not None:
                    ref_matrices.append(ref_matrix)
                else:
                    print(f"Warning: Reference matrix not found for {model_dir.name}")

                if conflict_matrix is not None:
                    conflict_matrices.append(conflict_matrix)
                else:
                    print(f"Warning: Conflict matrix not found for {model_dir.name}")

                if hcr_matrix is not None:
                    hcr_matrices.append(hcr_matrix)
                else:
                    print(f"Warning: HCR matrix not found for {model_dir.name}")

            print(f"\nSaving family averages...")
            if ref_matrices:
                save_matrix_csv(np.mean(ref_matrices, axis=0), languages, hierarchy_output_dir / 'reference.csv', hierarchy_type)

            if conflict_matrices:
                save_matrix_csv(np.mean(conflict_matrices, axis=0), languages, hierarchy_output_dir / 'conflict.csv', hierarchy_type)

            if hcr_matrices:
                save_matrix_csv(np.mean(hcr_matrices, axis=0), languages, hierarchy_output_dir / 'hcr.csv', hierarchy_type)
    
    # Save family manifest
    save_manifest(model_names, family_results_dir / 'manifest.json')
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully created family results for {args.model_family} ({args.domain})")
    print(f"✓ Output directory: {family_results_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
