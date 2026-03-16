"""
Visualize language matrix as heatmap.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_heatmap(csv_path, output_path=None, title=None, vmin=None, vmax=None, cmap='RdYlGn'):
    """
    Plot heatmap from CSV matrix.
    
    Args:
        csv_path: Path to CSV file
        output_path: Path to save figure (optional)
        title: Plot title (optional)
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
        cmap: Colormap (default: RdYlGn for green=good, red=bad)
    """
    # Load CSV
    df = pd.read_csv(csv_path, index_col=0)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        df, 
        annot=True,           # Show values in cells
        fmt='.3f',            # Format with 3 decimal places
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Score'},
        linewidths=0.5,
        linecolor='gray'
    )
    
    # Set title
    if title is None:
        title = Path(csv_path).stem.replace('_', ' ').title()
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Set labels
    plt.xlabel('Higher Hierarchy Language', fontsize=12, fontweight='bold')
    plt.ylabel('Lower Hierarchy Language', fontsize=12, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize language matrix as heatmap')
    parser.add_argument('-input', type=str, required=True, help='Path to CSV file')
    parser.add_argument('-output', type=str, help='Path to save figure (optional)')
    parser.add_argument('-title', type=str, help='Plot title (optional)')
    parser.add_argument('-vmin', type=float, help='Minimum value for colorbar')
    parser.add_argument('-vmax', type=float, help='Maximum value for colorbar')
    parser.add_argument('-cmap', type=str, default='RdYlGn', 
                       help='Colormap (default: RdYlGn). Options: RdYlGn, viridis, coolwarm, etc.')
    args = parser.parse_args()
    
    # Plot heatmap
    plot_heatmap(
        args.input,
        output_path=args.output,
        title=args.title,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap
    )


if __name__ == '__main__':
    main()

