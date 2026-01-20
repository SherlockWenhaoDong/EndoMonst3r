#!/usr/bin/env python3
"""
Split SCARED dataset - every Nth frame for testing
Images in images/, depth in depth/, masks in masks/
"""

import os
import shutil
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse


def extract_frame_number(filename: str) -> int:
    """
    Extract frame number from SCARED dataset filename

    Args:
        filename: Filename like 'frame_data000036.png' or '1_1_frame_data000036.png'

    Returns:
        Frame number as integer
    """
    # SCARED dataset patterns
    patterns = [
        r'frame_data(\d{6})',  # frame_data000036
        r'(\d{6})\.(png|jpg|jpeg|bmp|tiff|npz)$',  # 000036.png
        r'image_(\d+)',  # image_36
        r'frame_(\d+)',  # frame_36
    ]

    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue

    # Try to find any 6-digit number in filename
    numbers = re.findall(r'\d{6}', filename)
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            pass

    # Try to find any number in filename
    numbers = re.findall(r'\d+', filename)
    if numbers:
        try:
            # Use the largest number (usually frame number)
            return int(max(numbers, key=len))
        except ValueError:
            pass

    return 0


def find_associated_file(base_dir: Path, rgb_filename: str, subdir: str,
                         possible_extensions: List[str]) -> Optional[Path]:
    """
    Find associated file (depth or mask) for a given RGB image

    Args:
        base_dir: Base directory containing all subdirectories
        rgb_filename: RGB image filename
        subdir: Subdirectory name (e.g., 'depth', 'masks')
        possible_extensions: List of possible file extensions

    Returns:
        Path to associated file if found, None otherwise
    """
    # Get frame number from RGB filename
    frame_num = extract_frame_number(rgb_filename)

    # Try different naming patterns
    rgb_stem = Path(rgb_filename).stem

    patterns_to_try = [
        # Pattern 1: Same filename as RGB
        f"{rgb_stem}.*",
        # Pattern 2: frame_dataXXXXXX pattern
        f"*{frame_num:06d}*",
        # Pattern 3: Depth specific patterns
        f"depth*{frame_num:06d}*",
        f"*depth*{frame_num:06d}*",
        # Pattern 4: Mask specific patterns
        f"mask*{frame_num:06d}*",
        f"*mask*{frame_num:06d}*",
    ]

    subdir_path = base_dir / subdir
    if not subdir_path.exists():
        return None

    # Try each pattern
    for pattern in patterns_to_try:
        for ext in possible_extensions:
            full_pattern = f"{pattern}{ext}"
            matches = list(subdir_path.glob(full_pattern))
            if matches:
                # Return first match
                return matches[0]

    return None


def get_all_rgb_images(images_dir: Path) -> List[Dict]:
    """
    Get all RGB images with their frame numbers

    Args:
        images_dir: Path to images directory

    Returns:
        List of dictionaries with image info
    """
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return []

    # Supported image extensions
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']

    rgb_images = []

    for ext in image_extensions:
        for img_path in images_dir.glob(f"*{ext}"):
            frame_num = extract_frame_number(img_path.name)
            rgb_images.append({
                'path': img_path,
                'frame_num': frame_num,
                'filename': img_path.name
            })

    # Sort by frame number
    rgb_images.sort(key=lambda x: x['frame_num'])

    return rgb_images


def split_scared_dataset(base_dir: str, output_dir: str, test_interval: int = 8,
                         copy: bool = True, quick: bool = False):
    """
    Split SCARED dataset - every Nth frame for testing

    Args:
        base_dir: Base directory containing images/, depth/, masks/ subdirectories
        output_dir: Output directory for split dataset
        test_interval: Use every Nth frame for testing (default: 8)
        copy: True to copy files, False to move files
        quick: Quick mode (skip prompts)
    """
    base_path = Path(base_dir)
    output_path = Path(output_dir)

    # Check if base directory exists
    if not base_path.exists():
        print(f"Error: Base directory does not exist: {base_dir}")
        return

    # Check for required subdirectories
    images_dir = base_path / "images"
    depth_dir = base_path / "depth"
    masks_dir = base_path / "masks"

    if not images_dir.exists():
        print(f"Error: 'images' directory not found in {base_dir}")
        return

    print(f"Base directory: {base_dir}")
    print(f"Found directories:")
    print(f"  images/: {'✓' if images_dir.exists() else '✗'}")
    print(f"  depth/: {'✓' if depth_dir.exists() else '✗'}")
    print(f"  masks/: {'✓' if masks_dir.exists() else '✗'}")

    # Get all RGB images
    rgb_images = get_all_rgb_images(images_dir)

    if not rgb_images:
        print(f"Error: No RGB images found in {images_dir}")
        return

    print(f"\nFound {len(rgb_images)} RGB images")

    if not quick:
        print(f"\nWill use every {test_interval}th frame for testing")
        print(f"Operation: {'COPY' if copy else 'MOVE'}")

        if not copy:
            confirm = input("\nWARNING: Files will be MOVED from source directory!\nContinue? (y/n): ")
            if confirm.lower() != 'y':
                print("Operation cancelled")
                return

    # Create output directories
    train_base = output_path / "train"
    test_base = output_path / "test"

    for base in [train_base, test_base]:
        (base / "images").mkdir(parents=True, exist_ok=True)
        (base / "depth").mkdir(parents=True, exist_ok=True)
        (base / "masks").mkdir(parents=True, exist_ok=True)

    train_files = []
    test_files = []
    train_count = 0
    test_count = 0

    print(f"\nProcessing images...")

    for i, rgb_info in enumerate(rgb_images):
        rgb_path = rgb_info['path']
        frame_num = rgb_info['frame_num']

        # Determine if this is a test frame
        # Use 1-based indexing: frame 8, 16, 24, etc. are test frames
        is_test = (frame_num % test_interval == 0) if test_interval > 0 else False

        if is_test:
            dest_base = test_base
            test_files.append(rgb_path.name)
            test_count += 1
            set_name = "TEST"
        else:
            dest_base = train_base
            train_files.append(rgb_path.name)
            train_count += 1
            set_name = "TRAIN"

        # Process RGB image
        rgb_dest = dest_base / "images" / rgb_path.name
        if copy:
            shutil.copy2(str(rgb_path), str(rgb_dest))
        else:
            shutil.move(str(rgb_path), str(rgb_dest))

        # Find and copy depth file
        depth_file = find_associated_file(base_path, rgb_path.name, "depth", ['.npz', '.npy', '.png', '.jpg', '.tiff'])
        if depth_file and depth_file.exists():
            depth_dest = dest_base / "depth" / depth_file.name
            if copy:
                shutil.copy2(str(depth_file), str(depth_dest))
            else:
                shutil.move(str(depth_file), str(depth_dest))

        # Find and copy mask file
        mask_file = find_associated_file(base_path, rgb_path.name, "masks", ['.png', '.jpg', '.bmp', '.tiff'])
        if mask_file and mask_file.exists():
            mask_dest = dest_base / "masks" / mask_file.name
            if copy:
                shutil.copy2(str(mask_file), str(mask_dest))
            else:
                shutil.move(str(mask_file), str(mask_dest))

        # Print progress
        if (i + 1) % 10 == 0 or (i + 1) == len(rgb_images):
            print(f"  Processed {i + 1}/{len(rgb_images)} images")

    # Create summary
    total_images = len(rgb_images)
    train_ratio = train_count / total_images if total_images > 0 else 0
    test_ratio = test_count / total_images if total_images > 0 else 0

    summary = {
        'base_directory': str(base_path),
        'output_directory': str(output_path),
        'test_interval': test_interval,
        'total_images': total_images,
        'train_images': train_count,
        'test_images': test_count,
        'train_ratio': f"{train_ratio:.2%}",
        'test_ratio': f"{test_ratio:.2%}",
        'operation': 'copy' if copy else 'move',
        'has_depth': depth_dir.exists(),
        'has_masks': masks_dir.exists(),
        'output_structure': {
            'train/': ['images/', 'depth/', 'masks/'],
            'test/': ['images/', 'depth/', 'masks/']
        }
    }

    # Save summary
    summary_path = output_path / "split_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("SCARED Dataset Split Summary\n")
        f.write("============================\n\n")

        for key, value in summary.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for subkey, subvalue in value.items():
                    f.write(f"  {subkey}: {subvalue}\n")
            else:
                f.write(f"{key}: {value}\n")

        f.write(f"\nTest images (every {test_interval}th frame):\n")
        for i, filename in enumerate(test_files[:20]):  # Show first 20
            f.write(f"  {filename}\n")
        if len(test_files) > 20:
            f.write(f"  ... and {len(test_files) - 20} more\n")

    # Create train.txt and test.txt files
    train_list_path = output_path / "train.txt"
    with open(train_list_path, 'w', encoding='utf-8') as f:
        for filename in train_files:
            # Write relative path from output directory
            rel_path = f"train/images/{filename}"
            f.write(f"{rel_path}\n")

    test_list_path = output_path / "test.txt"
    with open(test_list_path, 'w', encoding='utf-8') as f:
        for filename in test_files:
            rel_path = f"test/images/{filename}"
            f.write(f"{rel_path}\n")

    # Print final summary
    print(f"\n{'=' * 50}")
    print("SPLIT COMPLETE!")
    print(f"{'=' * 50}")
    print(f"Total images processed: {total_images}")
    print(f"Training set: {train_count} images ({train_ratio:.2%})")
    print(f"Test set: {test_count} images ({test_ratio:.2%})")
    print(f"Test interval: every {test_interval}th frame")
    print(f"\nOutput structure:")
    print(f"  {output_path}/")
    print(f"    train/")
    print(f"      images/  - {train_count} RGB images")
    print(f"      depth/   - Depth maps")
    print(f"      masks/   - Masks")
    print(f"    test/")
    print(f"      images/  - {test_count} RGB images")
    print(f"      depth/   - Depth maps")
    print(f"      masks/   - Masks")
    print(f"    train.txt  - Training file list")
    print(f"    test.txt   - Test file list")
    print(f"    split_summary.txt")
    print(f"\nSummary saved to: {summary_path}")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description='Split SCARED dataset - every Nth frame for testing'
    )

    parser.add_argument(
        'base_dir',
        help='Base directory containing images/, depth/, masks/ subdirectories'
    )

    parser.add_argument(
        'output_dir',
        help='Output directory for split dataset'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=8,
        help='Use every Nth frame for testing (default: 8)'
    )

    parser.add_argument(
        '--move',
        action='store_true',
        help='Move files instead of copying (default: copy)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode (skip prompts)'
    )

    parser.add_argument(
        '--frame-start',
        type=int,
        default=0,
        help='Starting frame number (default: 0)'
    )

    parser.add_argument(
        '--frame-end',
        type=int,
        default=None,
        help='Ending frame number (default: process all)'
    )

    args = parser.parse_args()

    # Validate interval
    if args.interval <= 0:
        print("Error: Interval must be positive integer")
        return

    if not args.quick:
        print(f"Base directory: {args.base_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Test interval: every {args.interval}th frame")
        print(f"Operation: {'MOVE' if args.move else 'COPY'}")
        print(f"Frame range: {args.frame_start} to {args.frame_end if args.frame_end else 'end'}")

    split_scared_dataset(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        test_interval=args.interval,
        copy=not args.move,
        quick=args.quick
    )


if __name__ == "__main__":
    main()