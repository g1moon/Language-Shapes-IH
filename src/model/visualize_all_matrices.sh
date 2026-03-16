#!/bin/bash
# Visualize all language matrices for a model or family

if [ "$#" -lt 2 ]; then
    echo "Usage: bash visualize_all_matrices.sh <model_family> <model_name_or_family_results>"
    echo "Example (single model): bash visualize_all_matrices.sh qwen qwen3-4b"
    echo "Example (family avg):   bash visualize_all_matrices.sh qwen family_results"
    exit 1
fi

MODEL_FAMILY=$1
MODEL_NAME=$2

# Base paths
SCORES_DIR="model-scores/${MODEL_FAMILY}/${MODEL_NAME}/results/rule-following"
OUTPUT_DIR="figures/${MODEL_FAMILY}/${MODEL_NAME}"

# Check if input directory exists
if [ ! -d "$SCORES_DIR" ]; then
    echo "Error: Directory not found: $SCORES_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Hierarchy types
hierarchies=(sys-tool sys-user user-tool)

# Matrix types
declare -A matrix_configs
matrix_configs["reference"]="RdYlGn,0.0,1.0"
matrix_configs["conflict"]="RdYlGn,0.0,1.0"
matrix_configs["drop"]="RdYlGn_r,-0.5,0.5"

echo "Generating heatmaps for ${MODEL_FAMILY}/${MODEL_NAME}..."
echo ""

for hierarchy in ${hierarchies[@]}; do
    hierarchy_dir="${SCORES_DIR}/${hierarchy}"
    
    if [ ! -d "$hierarchy_dir" ]; then
        echo "Warning: Hierarchy directory not found: $hierarchy_dir"
        continue
    fi
    
    echo "Processing hierarchy: $hierarchy"
    
    for matrix_type in "${!matrix_configs[@]}"; do
        csv_file="${hierarchy_dir}/${matrix_type}.csv"
        
        if [ ! -f "$csv_file" ]; then
            echo "  Warning: CSV file not found: $csv_file"
            continue
        fi
        
        # Parse config
        IFS=',' read -r cmap vmin vmax <<< "${matrix_configs[$matrix_type]}"
        
        # Generate output filename
        output_file="${OUTPUT_DIR}/${hierarchy}_${matrix_type}.png"
        
        # Generate title
        title_hierarchy=$(echo "$hierarchy" | sed 's/-/ ↔ /g' | awk '{for(i=1;i<=NF;i++)sub(/./,toupper(substr($i,1,1)),$i)}1')
        title_type=$(echo "$matrix_type" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')
        
        if [ "$MODEL_NAME" = "family_results" ]; then
            title="${MODEL_FAMILY^} Family: ${title_hierarchy} ${title_type}"
        else
            title="${MODEL_NAME}: ${title_hierarchy} ${title_type}"
        fi
        
        # Run visualization
        python src/model/visualize_matrix.py \
            -input "$csv_file" \
            -output "$output_file" \
            -title "$title" \
            -cmap "$cmap" \
            -vmin "$vmin" \
            -vmax "$vmax"
        
        echo "  ✓ Generated: $output_file"
    done
    
    echo ""
done

echo "All heatmaps saved to: $OUTPUT_DIR"

