"""
Inference script for image matching.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse

from algorithm import SIFTMatcher, load_image_pair


def visualize_matches(img1, img2, pts1, pts2):
    """
    Visualize matches between two images (green lines with X markers).
    
    Args:
        img1: First image
        img2: Second image
        pts1: Matched points in image 1
        pts2: Matched points in image 2
        save_path: Path to save visualization
    """
    # Concatenate images side by side
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1+w2] = img2
    
    # Plot
    fig, ax = plt.subplots(figsize=(20, 10), dpi=150)
    ax.imshow(canvas)
    
    lime_green = '#00FF00'
    
    # Draw matches
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        x2_shifted = x2 + w1
        
        # Draw line
        ax.plot([x1, x2_shifted], [y1, y2], 
                color=lime_green, linewidth=1.2, alpha=0.7)
        
        # Draw X markers at endpoints
        ax.plot(x1, y1, 'x', color=lime_green, 
                markersize=8, markeredgewidth=2)
        ax.plot(x2_shifted, y2, 'x', color=lime_green, 
                markersize=8, markeredgewidth=2)
    
    ax.axis('off')
    ax.set_title(f'Feature Matches ({len(pts1)} matches)', 
                fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    plt.show()


def process_pair(pair_dir, output_dir=None):
    """
    Process a single image pair.
    
    Args:
        pair_dir: Path to pair directory
        output_dir: Directory to save results
    """
    print(f"Processing: {pair_dir}")
    
    # Load images
    img1, img2 = load_image_pair(pair_dir)
    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")
    
    # Load metadata
    metadata_path = Path(pair_dir) / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"\nPair info:")
        print(f"  Season 1: {metadata['img1'].get('season', 'unknown')}")
        print(f"  Season 2: {metadata['img2'].get('season', 'unknown')}")
    
    # Create matcher
    matcher = SIFTMatcher(
        ratio_threshold=0.8,
        ransac_threshold=10.0,
        min_matches=4,
        nfeatures=10000
    )
    
    # Match
    print(f"\nMatching...")
    results = matcher.match_image_pair(img1, img2)
    
    # Print results
    print(f"\nResults:")
    print(f"  Keypoints (img1): {len(results['keypoints1'])}")
    print(f"  Keypoints (img2): {len(results['keypoints2'])}")
    print(f"  Raw matches: {results['num_raw_matches']}")
    print(f"  Inlier matches: {results['num_inliers']}")
    print(f"  Inlier ratio: {results['inlier_ratio']:.2%}")
    
    # Visualize
    if results['num_inliers'] > 0:
        save_path = None
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            pair_name = Path(pair_dir).name
            save_path = str(output_path / f"{pair_name}_matches.png")
        
        visualize_matches(
            img1, img2,
            results['inlier_points1'],
            results['inlier_points2']
        )
    else:
        print("\nNo matches found!")


def main():
    parser = argparse.ArgumentParser(description='Image matching inference')
    parser.add_argument('--pair_dir', type=str,
                       help='Path to a single pair directory')
    parser.add_argument('--pairs_dir', type=str,
                       help='Path to directory containing multiple pairs')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    if args.pair_dir:
        # Process single pair
        process_pair(args.pair_dir, args.output_dir)
    
    elif args.pairs_dir:
        # Process all pairs
        pairs_path = Path(args.pairs_dir)
        pair_dirs = sorted(pairs_path.glob('pair_*'))
        
        print(f"Found {len(pair_dirs)} pairs\n")
        
        for pair_dir in pair_dirs:
            process_pair(str(pair_dir), args.output_dir)
    
    else:
        parser.error("Provide --pair_dir or --pairs_dir")


if __name__ == '__main__':
    main()