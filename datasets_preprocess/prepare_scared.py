#!/usr/bin/env python3
"""
SCARED dataset splitter - split data into training and test sets
Every Nth frame goes to test set, remaining frames to training set
"""

import os
import shutil
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
from dataclasses import dataclass


@dataclass
class FrameInfo:
    """Class to store frame information"""
    sequence_name: str  # Sequence name, e.g., "1_1"
    frame_number: int  # Frame number, e.g., 6
    rgb_path: Path  # Path to RGB image
    depth_path: Optional[Path] = None  # Path to depth map
    pose_path: Optional[Path] = None  # Path to pose file
    calib_path: Optional[Path] = None  # Path to calibration file


def parse_frame_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse SCARED dataset filename to extract sequence name and frame number

    Args:
        filename: Filename like '1_1_frame_data000006.png'

    Returns:
        Tuple of (sequence_name, frame_number) or (None, None) if parsing fails
    """
    # Pattern 1: RGB image pattern {sequence}_frame_data{frame:06d}.{ext}
    pattern1 = r'^(\d+_\d+)_frame_data(\d{6})\.(png|jpg|jpeg|bmp)$'

    # Pattern 2: Depth map pattern depth_{sequence}_frame_data{frame:06d}.npz
    pattern2 = r'^depth_(\d+_\d+)_frame_data(\d{6})\.npz$'

    # Pattern 3: Pose file pattern frame_data{frame:06d}.json
    pattern3 = r'^frame_data(\d{6})\.json$'

    # Try to match pattern 1 (RGB)
    match = re.match(pattern1, filename, re.IGNORECASE)
    if match:
        sequence_name = match.group(1)
        frame_number = int(match.group(2))
        return sequence_name, frame_number

    # Try to match pattern 2 (depth map)
    match = re.match(pattern2, filename, re.IGNORECASE)
    if match:
        sequence_name = match.group(1)
        frame_number = int(match.group(2))
        return sequence_name, frame_number

    # Try to match pattern 3 (pose file)
    match = re.match(pattern3, filename, re.IGNORECASE)
    if match:
        # Pose files don't contain sequence name, will be extracted from directory structure
        frame_number = int(match.group(1))
        return None, frame_number

    return None, None


def find_associated_depth(rgb_path: Path) -> Optional[Path]:
    """
    Find corresponding depth map for a given RGB image

    Args:
        rgb_path: Path to RGB image

    Returns:
        Path to depth map or None if not found
    """
    # Extract sequence name and frame number
    seq_name, frame_num = parse_frame_filename(rgb_path.name)
    if seq_name is None or frame_num is None:
        return None

    # Build depth map path
    rgb_parent = rgb_path.parent
    monodep_dir = rgb_parent / "monodep"

    if not monodep_dir.exists():
        # Try to find monodep folder in parent directory
        monodep_dir = rgb_parent.parent / "monodep"
        if not monodep_dir.exists():
            return None

    # Depth map filename pattern: depth_{seq_name}_frame_data{frame_num:06d}.npz
    depth_filename = f"depth_{seq_name}_frame_data{frame_num:06d}.npz"
    depth_path = monodep_dir / depth_filename

    if depth_path.exists():
        return depth_path

    # Try alternative naming patterns
    depth_patterns = [
        f"depth_{seq_name}_frame_data{frame_num:06d}.*",
        f"{seq_name}_frame_data{frame_num:06d}_depth.*",
        f"depth_{frame_num:06d}.*",
        f"{frame_num:06d}_depth.*"
    ]

    for pattern in depth_patterns:
        matches = list(monodep_dir.glob(pattern))
        if matches:
            return matches[0]

    return None


def find_associated_pose(rgb_path: Path) -> Optional[Path]:
    """
    Find corresponding pose file for a given RGB image

    Args:
        rgb_path: Path to RGB image

    Returns:
        Path to pose file or None if not found
    """
    # Extract sequence name and frame number
    seq_name, frame_num = parse_frame_filename(rgb_path.name)
    if seq_name is None or frame_num is None:
        return None

    # Build pose file path
    rgb_parent = rgb_path.parent
    poses_dir = rgb_parent / "poses"

    if not poses_dir.exists():
        # Try to find poses folder in parent directory
        poses_dir = rgb_parent.parent / "poses"
        if not poses_dir.exists():
            return None

    # Pose file might be in a subfolder named after the sequence
    seq_pose_dir = poses_dir / seq_name
    if seq_pose_dir.exists():
        pose_filename = f"frame_data{frame_num:06d}.json"
        pose_path = seq_pose_dir / pose_filename

        if pose_path.exists():
            return pose_path

    # Try to find directly in poses folder
    pose_filename = f"frame_data{frame_num:06d}.json"
    pose_path = poses_dir / pose_filename

    if pose_path.exists():
        return pose_path

    # Try alternative naming patterns
    pose_patterns = [
        f"frame_data{frame_num:06d}.*",
        f"{seq_name}_frame_data{frame_num:06d}.*",
        f"pose_{frame_num:06d}.*"
    ]

    for pattern in pose_patterns:
        if seq_pose_dir.exists():
            matches = list(seq_pose_dir.glob(pattern))
        else:
            matches = list(poses_dir.glob(pattern))

        if matches:
            return matches[0]

    return None


def collect_scared_sequence(seq_dir: Path) -> List[FrameInfo]:
    """
    Collect all frame information for a single sequence

    Args:
        seq_dir: Sequence folder path, e.g., /path/to/dataset/1_1

    Returns:
        List of frame information for this sequence
    """
    frame_infos = []

    # Check for inputs folder
    inputs_dir = seq_dir / "inputs"
    if not inputs_dir.exists():
        # If no inputs folder, search directly in sequence folder
        inputs_dir = seq_dir

    # Find all RGB images
    rgb_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    rgb_files = []

    for ext in rgb_extensions:
        rgb_files.extend(inputs_dir.glob(f"*{ext}"))

    if not rgb_files:
        return frame_infos

    # Parse each RGB file
    for rgb_path in rgb_files:
        seq_name, frame_num = parse_frame_filename(rgb_path.name)

        if seq_name is None or frame_num is None:
            print(f"Warning: Cannot parse filename {rgb_path.name}")
            continue

        # Find corresponding depth map
        depth_path = find_associated_depth(rgb_path)

        # Find corresponding pose file
        pose_path = find_associated_pose(rgb_path)

        frame_info = FrameInfo(
            sequence_name=seq_name,
            frame_number=frame_num,
            rgb_path=rgb_path,
            depth_path=depth_path,
            pose_path=pose_path
        )

        frame_infos.append(frame_info)

    # Sort by frame number
    frame_infos.sort(key=lambda x: x.frame_number)

    return frame_infos


def collect_all_sequences(base_dir: Path) -> Dict[str, List[FrameInfo]]:
    """
    Collect frame information for all sequences

    Args:
        base_dir: Root directory of the dataset

    Returns:
        Dictionary: {sequence_name: [list_of_frame_infos]}
    """
    all_sequences = {}

    print(f"Scanning SCARED dataset: {base_dir}")

    # Find all possible sequence folders
    for item in base_dir.iterdir():
        if item.is_dir():
            # Check if it's a sequence folder (format like "1_1", "1_2", etc.)
            if re.match(r'^\d+_\d+$', item.name):
                seq_name = item.name
                print(f"  Found sequence: {seq_name}")

                # Collect all frames for this sequence
                frame_infos = collect_scared_sequence(item)

                if frame_infos:
                    all_sequences[seq_name] = frame_infos
                    print(f"    Found {len(frame_infos)} frames")
                else:
                    print(f"    Warning: No valid frames found in sequence {seq_name}")

    return all_sequences


def split_frames_by_interval(frame_infos: List[FrameInfo], test_interval: int = 8) -> Tuple[
    List[FrameInfo], List[FrameInfo]]:
    """
    Split frames into training and test sets based on frame number interval

    Args:
        frame_infos: List of frame information
        test_interval: Test interval (every Nth frame goes to test set)

    Returns:
        Tuple of (train_frames, test_frames)
    """
    train_frames = []
    test_frames = []

    for frame_info in frame_infos:
        # Frame numbers start from 1, every Nth frame goes to test set
        if frame_info.frame_number % test_interval == 0:
            test_frames.append(frame_info)
        else:
            train_frames.append(frame_info)

    return train_frames, test_frames


def create_sequence_aware_filename(original_path: Path, sequence_name: str) -> str:
    """
    Create a filename that includes sequence information

    Args:
        original_path: Original file path
        sequence_name: Sequence name to include in filename

    Returns:
        New filename with sequence information
    """
    stem = original_path.stem
    suffix = original_path.suffix

    # Check if filename already contains sequence information
    if sequence_name in stem:
        # If already contains sequence info, return as is
        return original_path.name

    # Add sequence name to filename
    # For pose files (frame_data000006.json) -> 1_1_frame_data000006.json
    # For depth files (depth_1_1_frame_data000006.npz) -> already has sequence info
    # For RGB files (1_1_frame_data000006.png) -> already has sequence info

    # For pose files: frame_data000006.json -> {sequence}_frame_data{frame:06d}.json
    if stem.startswith('frame_data'):
        # Extract frame number from pose filename
        frame_match = re.search(r'frame_data(\d{6})', stem)
        if frame_match:
            frame_num = frame_match.group(1)
            return f"{sequence_name}_frame_data{frame_num}{suffix}"

    # For other files, add sequence name prefix
    return f"{sequence_name}_{stem}{suffix}"


def organize_output_files(frames: List[FrameInfo], output_base: Path, set_name: str, copy: bool = True):
    """
    Organize output files into specified directories

    Args:
        frames: List of frame information
        output_base: Base output directory
        set_name: Dataset name ('train' or 'test')
        copy: Whether to copy files (True) or move them (False)
    """
    output_dir = output_base / set_name

    # Create directory structure
    rgb_dir = output_dir / "images"
    depth_dir = output_dir / "depth"
    pose_dir = output_dir / "poses"

    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    operation = "Copying" if copy else "Moving"
    print(f"\n{operation} {set_name} set files...")

    for i, frame_info in enumerate(frames):
        seq_name = frame_info.sequence_name

        # Process RGB image - keep original name (already contains sequence info)
        rgb_output = rgb_dir / frame_info.rgb_path.name
        if copy:
            shutil.copy2(str(frame_info.rgb_path), str(rgb_output))
        else:
            shutil.move(str(frame_info.rgb_path), str(rgb_output))

        # Process depth map - keep original name (already contains sequence info)
        if frame_info.depth_path and frame_info.depth_path.exists():
            # Depth map filename already contains sequence info
            depth_output = depth_dir / frame_info.depth_path.name
            if copy:
                shutil.copy2(str(frame_info.depth_path), str(depth_output))
            else:
                shutil.move(str(frame_info.depth_path), str(depth_output))

        # Process pose file - add sequence name to filename
        if frame_info.pose_path and frame_info.pose_path.exists():
            # Create new filename with sequence information
            new_pose_filename = create_sequence_aware_filename(frame_info.pose_path, seq_name)
            pose_output = pose_dir / new_pose_filename

            # Copy or move the file with new name
            if copy:
                shutil.copy2(str(frame_info.pose_path), str(pose_output))
            else:
                shutil.move(str(frame_info.pose_path), str(pose_output))

        # Show progress
        if (i + 1) % 10 == 0 or (i + 1) == len(frames):
            print(f"  Processed {i + 1}/{len(frames)} frames")


def create_file_lists(train_frames: List[FrameInfo], test_frames: List[FrameInfo], output_dir: Path):
    """
    Create training and test set file lists

    Args:
        train_frames: Training set frame information
        test_frames: Test set frame information
        output_dir: Output directory
    """
    # Create training set file list
    train_list_path = output_dir / "train.txt"
    with open(train_list_path, 'w', encoding='utf-8') as f:
        for frame_info in train_frames:
            # Write relative path
            rel_path = f"train/images/{frame_info.rgb_path.name}"
            f.write(f"{rel_path}\n")

    # Create test set file list
    test_list_path = output_dir / "test.txt"
    with open(test_list_path, 'w', encoding='utf-8') as f:
        for frame_info in test_frames:
            rel_path = f"test/images/{frame_info.rgb_path.name}"
            f.write(f"{rel_path}\n")

    # Create detailed information file
    info = {
        "total_sequences": len({f.sequence_name for f in train_frames + test_frames}),
        "total_frames": len(train_frames) + len(test_frames),
        "train_frames": len(train_frames),
        "test_frames": len(test_frames),
        "train_ratio": f"{len(train_frames) / (len(train_frames) + len(test_frames)):.2%}",
        "test_ratio": f"{len(test_frames) / (len(train_frames) + len(test_frames)):.2%}",
        "sequences": sorted({f.sequence_name for f in train_frames + test_frames}),
        "train_sequences": sorted({f.sequence_name for f in train_frames}),
        "test_sequences": sorted({f.sequence_name for f in test_frames}),
    }

    info_path = output_dir / "split_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"File lists created:")
    print(f"  {train_list_path}")
    print(f"  {test_list_path}")
    print(f"  {info_path}")


def split_scared_dataset(input_dir: str, output_dir: str, test_interval: int = 8,
                         copy_files: bool = True, create_lists_only: bool = False):
    """
    Main function to split SCARED dataset

    Args:
        input_dir: Input dataset directory
        output_dir: Output directory
        test_interval: Test interval (every Nth frame for test set)
        copy_files: Whether to copy files (True) or move them (False)
        create_lists_only: Whether to only create file lists without copying/moving files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    print("=" * 60)
    print(f"SCARED Dataset Split")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Test interval: Every {test_interval}th frame for test set")
    print(f"Operation mode: {'Copy files' if copy_files else 'Move files'}")
    print(f"Create lists only: {'Yes' if create_lists_only else 'No'}")

    # Collect all sequences
    all_sequences = collect_all_sequences(input_path)

    if not all_sequences:
        print("Error: No valid sequences found!")
        return

    total_sequences = len(all_sequences)
    total_frames = sum(len(frames) for frames in all_sequences.values())

    print(f"\nDataset statistics:")
    print(f"  Total sequences: {total_sequences}")
    print(f"  Total frames: {total_frames}")

    # Split each sequence
    all_train_frames = []
    all_test_frames = []

    for seq_name, frame_infos in all_sequences.items():
        print(f"\nProcessing sequence {seq_name}: {len(frame_infos)} frames")

        train_frames, test_frames = split_frames_by_interval(frame_infos, test_interval)

        print(f"  Training set: {len(train_frames)} frames")
        print(f"  Test set: {len(test_frames)} frames")

        all_train_frames.extend(train_frames)
        all_test_frames.extend(test_frames)

    total_train = len(all_train_frames)
    total_test = len(all_test_frames)

    print(f"\nOverall statistics:")
    print(f"  Total training frames: {total_train} ({total_train / total_frames:.2%})")
    print(f"  Total test frames: {total_test} ({total_test / total_frames:.2%})")

    if total_train == 0 or total_test == 0:
        print("Warning: Training set or test set is empty!")
        return

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    if not create_lists_only:
        # Organize output files
        organize_output_files(all_train_frames, output_path, "train", copy_files)
        organize_output_files(all_test_frames, output_path, "test", copy_files)

    # Create file lists
    create_file_lists(all_train_frames, all_test_frames, output_path)

    # Create summary file
    summary = {
        "input_directory": str(input_path),
        "output_directory": str(output_path),
        "test_interval": test_interval,
        "total_sequences": total_sequences,
        "total_frames": total_frames,
        "train_frames": total_train,
        "test_frames": total_test,
        "train_ratio": f"{total_train / total_frames:.2%}",
        "test_ratio": f"{total_test / total_frames:.2%}",
        "operation": "copy" if copy_files else "move",
        "create_lists_only": create_lists_only,
        "sequences": list(all_sequences.keys()),
        "filename_conventions": {
            "rgb_images": "{sequence}_frame_data{frame:06d}.{ext}",
            "depth_maps": "depth_{sequence}_frame_data{frame:06d}.npz",
            "pose_files": "{sequence}_frame_data{frame:06d}.json"
        }
    }

    summary_path = output_path / "split_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("SCARED Dataset Split Summary\n")
        f.write("=" * 40 + "\n\n")

        for key, value in summary.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for subkey, subvalue in value.items():
                    f.write(f"  {subkey}: {subvalue}\n")
            elif isinstance(value, list):
                f.write(f"{key}:\n")
                for item in value:
                    f.write(f"  - {item}\n")
            else:
                f.write(f"{key}: {value}\n")

        # Add detailed sequence information
        f.write(f"\nSequence Details:\n")
        for seq_name, frame_infos in all_sequences.items():
            seq_train = len([f for f in frame_infos if f.frame_number % test_interval != 0])
            seq_test = len([f for f in frame_infos if f.frame_number % test_interval == 0])
            f.write(f"\n  {seq_name}:\n")
            f.write(f"    Total frames: {len(frame_infos)}\n")
            f.write(f"    Training frames: {seq_train}\n")
            f.write(f"    Test frames: {seq_test}\n")
            f.write(f"    Frame range: {frame_infos[0].frame_number} - {frame_infos[-1].frame_number}\n")

    print(f"\n{'=' * 60}")
    print(f"Split completed!")
    print(f"{'=' * 60}")
    print(f"Output directory structure:")
    print(f"  {output_path}/")
    print(f"    train/")
    print(f"      images/  - {total_train} RGB images")
    print(f"      depth/   - {total_train} depth maps")
    print(f"      poses/   - {total_train} pose files (with sequence names)")
    print(f"    test/")
    print(f"      images/  - {total_test} RGB images")
    print(f"      depth/   - {total_test} depth maps")
    print(f"      poses/   - {total_test} pose files (with sequence names)")
    print(f"    train.txt  - Training set file list")
    print(f"    test.txt   - Test set file list")
    print(f"    split_info.json - Detailed split information")
    print(f"    split_summary.txt - Split summary")
    print(f"\nFilename conventions:")
    print(f"  RGB images: {summary['filename_conventions']['rgb_images']}")
    print(f"  Depth maps: {summary['filename_conventions']['depth_maps']}")
    print(f"  Pose files: {summary['filename_conventions']['pose_files']}")
    print(f"\nSummary file: {summary_path}")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description='Split SCARED dataset - every Nth frame for test set'
    )

    parser.add_argument(
        'input_dir',
        help='SCARED dataset input directory'
    )

    parser.add_argument(
        'output_dir',
        help='Output directory for split dataset'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=8,
        help='Every Nth frame for test set (default: 8)'
    )

    parser.add_argument(
        '--move',
        action='store_true',
        help='Move files instead of copying them (default: copy)'
    )

    parser.add_argument(
        '--lists-only',
        action='store_true',
        help='Only create file lists, do not copy/move files'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode, skip confirmation prompts'
    )

    parser.add_argument(
        '--keep-original-pose-names',
        action='store_true',
        help='Keep original pose filenames (do not add sequence info)'
    )

    args = parser.parse_args()

    if not args.quick:
        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Test interval: Every {args.interval}th frame")
        print(f"Operation mode: {'Move' if args.move else 'Copy'}")
        print(f"Create lists only: {'Yes' if args.lists_only else 'No'}")
        print(f"Keep original pose names: {'Yes' if args.keep_original_pose_names else 'No'}")

        if args.move:
            confirm = input("\nWarning: Files will be MOVED from source directory!\nContinue? (y/n): ")
            if confirm.lower() != 'y':
                print("Operation cancelled")
                return

    # Global flag for pose filename handling
    global KEEP_ORIGINAL_POSE_NAMES
    KEEP_ORIGINAL_POSE_NAMES = args.keep_original_pose_names

    split_scared_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        test_interval=args.interval,
        copy_files=not args.move,
        create_lists_only=args.lists_only
    )


# Global flag for pose filename handling
KEEP_ORIGINAL_POSE_NAMES = False


def create_sequence_aware_filename(original_path: Path, sequence_name: str) -> str:
    """
    Create a filename that includes sequence information

    Args:
        original_path: Original file path
        sequence_name: Sequence name to include in filename

    Returns:
        New filename with sequence information
    """
    # Check global flag
    if KEEP_ORIGINAL_POSE_NAMES:
        return original_path.name

    stem = original_path.stem
    suffix = original_path.suffix

    # Check if filename already contains sequence information
    if sequence_name in stem:
        # If already contains sequence info, return as is
        return original_path.name

    # Add sequence name to filename
    # For pose files (frame_data000006.json) -> 1_1_frame_data000006.json
    # For depth files (depth_1_1_frame_data000006.npz) -> already has sequence info
    # For RGB files (1_1_frame_data000006.png) -> already has sequence info

    # For pose files: frame_data000006.json -> {sequence}_frame_data{frame:06d}.json
    if stem.startswith('frame_data'):
        # Extract frame number from pose filename
        frame_match = re.search(r'frame_data(\d{6})', stem)
        if frame_match:
            frame_num = frame_match.group(1)
            return f"{sequence_name}_frame_data{frame_num}{suffix}"

    # For other files, add sequence name prefix
    return f"{sequence_name}_{stem}{suffix}"


if __name__ == "__main__":
    main()