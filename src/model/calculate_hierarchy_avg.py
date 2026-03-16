#!/usr/bin/env python3
"""
Calculate average scores across hierarchy types for a single model and domain.
This is called after aggregate_matrix_scores.py for each model.
"""

import os
import pandas as pd
from pathlib import Path
import argparse


def calculate_average_csv(csv_files, output_path):
    """
    Calculate average of multiple CSV files with the same structure.
    
    Args:
        csv_files: List of CSV file paths
        output_path: Path to save the averaged CSV
    """
    if not csv_files:
        print(f"No CSV files provided for {output_path}")
        return
    
    # Read all CSV files
    dfs = []
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, index_col=0)
            dfs.append(df)
        else:
            print(f"Warning: {csv_file} does not exist")
    
    if not dfs:
        print(f"No valid CSV files found for {output_path}")
        return
    
    # Calculate average
    avg_df = sum(dfs) / len(dfs)
    
    # Save to output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    avg_df.to_csv(output_path)
    print(f"Saved average to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Calculate average scores across hierarchy types for a single model')
    parser.add_argument('-model_family', type=str, required=True, help='Model family (e.g., llama)')
    parser.add_argument('-model_name', type=str, required=True, help='Model name (e.g., llama3.1-8b)')
    parser.add_argument('-domain', type=str, required=True, help='Domain (e.g., rule-following)')
    parser.add_argument('-output_dir', type=str, default='model-scores', help='Output directory')
    
    args = parser.parse_args()
    
    model_dir = Path(args.output_dir) / args.model_family / args.model_name / "results" / args.domain
    
    if not model_dir.exists():
        print(f"Model directory does not exist: {model_dir}")
        return
    
    # Hierarchy types to average
    hierarchy_types = ['sys-tool', 'sys-user', 'user-tool']
    
    # Find all CSV files in the hierarchy directories
    csv_files_map = {}
    
    for hierarchy_type in hierarchy_types:
        hierarchy_dir = model_dir / hierarchy_type
        if hierarchy_dir.exists():
            for csv_file in hierarchy_dir.glob("*.csv"):
                csv_name = csv_file.name
                if csv_name not in csv_files_map:
                    csv_files_map[csv_name] = []
                csv_files_map[csv_name].append(csv_file)
    
    # Calculate averages for each CSV file type
    for csv_name, csv_files in csv_files_map.items():
        output_path = model_dir / csv_name
        calculate_average_csv(csv_files, output_path)
    
    # Process subdirectories (loose, strict) only for domains that have them
    single_file_domains = {'safety', 'task-execution', 'persona'}
    if args.domain not in single_file_domains:
        subdirs = ['loose', 'strict']
        for subdir in subdirs:
            csv_files_map_sub = {}

            for hierarchy_type in hierarchy_types:
                hierarchy_subdir = model_dir / hierarchy_type / subdir
                if hierarchy_subdir.exists():
                    for csv_file in hierarchy_subdir.glob("*.csv"):
                        csv_name = csv_file.name
                        if csv_name not in csv_files_map_sub:
                            csv_files_map_sub[csv_name] = []
                        csv_files_map_sub[csv_name].append(csv_file)

            # Calculate averages for subdirectory CSV files
            for csv_name, csv_files in csv_files_map_sub.items():
                output_path = model_dir / subdir / csv_name
                calculate_average_csv(csv_files, output_path)

    print(f"✓ Calculated hierarchy averages for {args.model_name} ({args.domain})")


if __name__ == "__main__":
    main()
