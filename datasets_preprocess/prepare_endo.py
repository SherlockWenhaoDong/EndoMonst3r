#!/usr/bin/python

import glob
import os
import shutil
from PIL import Image
import numpy as np
import argparse
import xml.etree.ElementTree as ET
import json
from pathlib import Path


def parse_calibration_xml(xml_path):
    """
    Parse camera calibration XML file to extract intrinsic parameters

    Expected XML structure:
    <calibration>
        <camera_matrix>
            <rows>3</rows>
            <cols>3</cols>
            <data>fx 0 cx 0 fy cy 0 0 1</data>
        </camera_matrix>
        <image_width>1920</image_width>
        <image_height>1080</image_height>
        <distortion_coefficients>
            <rows>1</rows>
            <cols>5</cols>
            <data>k1 k2 p1 p2 k3</data>
        </distortion_coefficients>
    </calibration>

    Args:
        xml_path: Path to calibration XML file

    Returns:
        dict: Camera intrinsic parameters including K matrix and distortion coefficients
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Calibration XML file not found: {xml_path}")

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Parse camera matrix
        camera_matrix_elem = root.find('camera_matrix')
        if camera_matrix_elem is None:
            raise ValueError("camera_matrix element not found in XML")

        # Parse data string and convert to 3x3 matrix
        data_str = camera_matrix_elem.find('data').text
        data_values = list(map(float, data_str.split()))

        # Assuming row-major order
        K = np.array([
            [data_values[0], data_values[1], data_values[2]],
            [data_values[3], data_values[4], data_values[5]],
            [data_values[6], data_values[7], data_values[8]]
        ]).reshape(3, 3)

        # Parse image dimensions
        width = int(root.find('image_width').text)
        height = int(root.find('image_height').text)

        # Parse distortion coefficients if present
        dist_coeffs = np.zeros(5, dtype=np.float32)
        dist_elem = root.find('distortion_coefficients')
        if dist_elem is not None:
            dist_data_str = dist_elem.find('data').text
            dist_values = list(map(float, dist_data_str.split()))
            dist_coeffs[:len(dist_values)] = dist_values

        return {
            'K': K,
            'width': width,
            'height': height,
            'distortion_coeffs': dist_coeffs,
            'fx': K[0, 0],
            'fy': K[1, 1],
            'cx': K[0, 2],
            'cy': K[1, 2]
        }

    except Exception as e:
        raise ValueError(f"Error parsing calibration XML {xml_path}: {e}")


def load_fixed_extrinsics(extrinsics_file):
    """
    Load fixed camera extrinsic parameters

    Args:
        extrinsics_file: Path to extrinsics file (JSON or text format)

    Returns:
        dict: Camera extrinsic parameters (rotation matrix R and translation vector t)
    """
    if not os.path.exists(extrinsics_file):
        # Return identity transformation if no extrinsics file
        print(f"Warning: Extrinsics file not found: {extrinsics_file}. Using identity transformation.")
        return {
            'R': np.eye(3, dtype=np.float32),
            't': np.zeros(3, dtype=np.float32),
            'pose': np.eye(4, dtype=np.float32)  # 4x4 pose matrix
        }

    ext = Path(extrinsics_file).suffix.lower()

    if ext == '.json':
        # JSON format
        with open(extrinsics_file, 'r') as f:
            data = json.load(f)

        # Try to parse different possible formats
        if 'rotation' in data and 'translation' in data:
            R = np.array(data['rotation'], dtype=np.float32).reshape(3, 3)
            t = np.array(data['translation'], dtype=np.float32).reshape(3, 1)
        elif 'R' in data and 't' in data:
            R = np.array(data['R'], dtype=np.float32).reshape(3, 3)
            t = np.array(data['t'], dtype=np.float32).reshape(3, 1)
        elif 'pose' in data:
            # 4x4 pose matrix
            pose = np.array(data['pose'], dtype=np.float32).reshape(4, 4)
            R = pose[:3, :3]
            t = pose[:3, 3:4]
        else:
            raise ValueError(f"Unknown JSON format in {extrinsics_file}")

    elif ext == '.txt':
        # Text format: could be 3x4 projection matrix or separate R and t
        with open(extrinsics_file, 'r') as f:
            lines = f.readlines()

        # Try to parse as 3x4 matrix
        if len(lines) >= 3:
            data = []
            for line in lines[:3]:
                row = list(map(float, line.strip().split()))
                data.extend(row)

            if len(data) == 12:
                # 3x4 projection matrix P = K[R|t]
                P = np.array(data, dtype=np.float32).reshape(3, 4)
                # Extract R and t (assuming K is known from calibration)
                # Note: This requires K to properly decompose
                print(f"Warning: Text file contains projection matrix, need K to extract R and t")
                R = np.eye(3, dtype=np.float32)
                t = np.zeros((3, 1), dtype=np.float32)
            else:
                raise ValueError(f"Invalid format in {extrinsics_file}")
        else:
            raise ValueError(f"Invalid format in {extrinsics_file}")
    else:
        raise ValueError(f"Unsupported extrinsics file format: {ext}")

    # Create 4x4 pose matrix
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = t.flatten()

    return {
        'R': R,
        't': t,
        'pose': pose
    }


def mask_read(filename):
    """
    Read mask file
    Returns numpy array where mask region is 1 and background is 0
    """
    mask_img = Image.open(filename)
    mask_np = np.array(mask_img, dtype=np.uint8)

    # If mask is grayscale
    if len(mask_np.shape) == 2:
        # Normalize to 0-1
        mask = (mask_np > 0).astype(np.float32)
    # If mask is RGB/RGBA
    else:
        # Convert to grayscale and binarize
        mask_gray = np.mean(mask_np, axis=2)
        mask = (mask_gray > 0).astype(np.float32)

    return mask


def gather_rgb_mask_calib_data(rgb_dir, mask_dir, calib_dir, output_base_dir,
                               extrinsics_file=None, subset_name="train", max_samples=None):
    """
    Collect RGB, mask, and calibration data into unified directory structure

    Args:
        rgb_dir: Path to RGB image directory
        mask_dir: Path to mask file directory
        calib_dir: Path to calibration XML directory
        output_base_dir: Base output directory
        extrinsics_file: Path to fixed extrinsics file
        subset_name: Dataset subset name (train, val, test, etc.)
        max_samples: Maximum number of samples (None means all)
    """
    # Get all RGB image files
    rgb_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    rgb_files = []

    for ext in rgb_extensions:
        rgb_files.extend(glob.glob(os.path.join(rgb_dir, ext)))

    rgb_files.sort()

    # Limit number of samples if specified
    if max_samples is not None and max_samples > 0:
        rgb_files = rgb_files[:max_samples]

    print(f"Found {len(rgb_files)} RGB image files")

    if len(rgb_files) == 0:
        print("❌ No RGB image files found")
        return

    # Load fixed extrinsics if provided
    extrinsics = None
    if extrinsics_file and os.path.exists(extrinsics_file):
        print(f"Loading fixed extrinsics from: {extrinsics_file}")
        extrinsics = load_fixed_extrinsics(extrinsics_file)

    # Create output directories
    rgb_output_dir = os.path.join(output_base_dir, f"{subset_name}_selection", "image_gathered")
    mask_output_dir = os.path.join(output_base_dir, f"{subset_name}_selection", "mask_gathered")
    calib_output_dir = os.path.join(output_base_dir, f"{subset_name}_selection", "calibration")

    os.makedirs(rgb_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)
    os.makedirs(calib_output_dir, exist_ok=True)

    # Create JSON file to store all calibration data
    calibration_json_path = os.path.join(calib_output_dir, "calibration_data.json")
    calibration_data = {}

    # Process each RGB image
    processed_count = 0
    for i, rgb_file in enumerate(rgb_files):
        rgb_filename = os.path.basename(rgb_file)
        rgb_name, rgb_ext = os.path.splitext(rgb_filename)

        # Find corresponding mask file
        mask_file = find_matching_mask(rgb_name, mask_dir)
        if mask_file is None:
            print(f"⚠️  No corresponding mask file found for {rgb_filename}, skipping")
            continue

        # Find corresponding calibration XML file
        calib_file = find_calibration_file(rgb_name, calib_dir)
        if calib_file is None:
            print(f"⚠️  No calibration file found for {rgb_filename}, skipping")
            continue

        try:
            # Parse calibration XML
            calib_data = parse_calibration_xml(calib_file)

            # Combine with extrinsics if available
            if extrinsics:
                calib_data.update({
                    'R': extrinsics['R'].tolist(),
                    't': extrinsics['t'].flatten().tolist(),
                    'pose': extrinsics['pose'].tolist(),
                    'extrinsics_source': extrinsics_file
                })
            else:
                # Use identity transformation
                calib_data.update({
                    'R': np.eye(3, dtype=np.float32).tolist(),
                    't': np.zeros(3, dtype=np.float32).tolist(),
                    'pose': np.eye(4, dtype=np.float32).tolist(),
                    'extrinsics_source': 'identity'
                })

            # Store calibration data
            calibration_data[rgb_name] = calib_data

            # Copy RGB file
            rgb_output_path = os.path.join(rgb_output_dir, f"{rgb_name}{rgb_ext}")
            shutil.copy(rgb_file, rgb_output_path)

            # Copy mask file
            mask_filename = os.path.basename(mask_file)
            mask_name, mask_ext = os.path.splitext(mask_filename)
            mask_output_path = os.path.join(mask_output_dir, f"{rgb_name}_mask{mask_ext}")
            shutil.copy(mask_file, mask_output_path)

            # Copy calibration file
            calib_output_path = os.path.join(calib_output_dir, f"{rgb_name}_calib.xml")
            shutil.copy(calib_file, calib_output_path)

            processed_count += 1

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(rgb_files)} files")

        except Exception as e:
            print(f"❌ Error processing {rgb_filename}: {e}")
            continue

    # Save calibration data to JSON
    if calibration_data:
        with open(calibration_json_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"Saved calibration data to: {calibration_json_path}")

    print(f"✅ Processing complete! Successfully processed {processed_count}/{len(rgb_files)} image pairs")
    print(f"RGB output directory: {rgb_output_dir}")
    print(f"Mask output directory: {mask_output_dir}")
    print(f"Calibration output directory: {calib_output_dir}")

    return calibration_json_path


def find_calibration_file(image_name, calib_dir):
    """
    Find corresponding calibration XML file based on image filename

    Args:
        image_name: Name of image file (without extension)
        calib_dir: Directory containing calibration XML files

    Returns:
        Path to matching calibration file or None if not found
    """
    calib_extensions = ['*.xml', '*.XML']

    # Try multiple possible calibration filename patterns
    possible_patterns = [
        os.path.join(calib_dir, f"{image_name}_calib.*"),
        os.path.join(calib_dir, f"{image_name}_calibration.*"),
        os.path.join(calib_dir, f"calib_{image_name}.*"),
        os.path.join(calib_dir, f"calibration_{image_name}.*"),
        os.path.join(calib_dir, f"{image_name}.*"),  # Calib file with same name as image
    ]

    for pattern in possible_patterns:
        for ext in calib_extensions:
            search_pattern = pattern.replace('.*', ext.replace('*', ''))
            matches = glob.glob(search_pattern)
            if matches:
                return matches[0]

    # If no specific pattern found, try to find any XML file
    xml_files = glob.glob(os.path.join(calib_dir, "*.xml")) + glob.glob(os.path.join(calib_dir, "*.XML"))
    if xml_files:
        # If there's only one calibration file, use it for all images
        if len(xml_files) == 1:
            print(f"Using single calibration file for all images: {xml_files[0]}")
            return xml_files[0]

    return None


def find_matching_mask(rgb_name, mask_dir):
    """
    Find corresponding mask file based on RGB filename

    Args:
        rgb_name: Name of RGB file (without extension)
        mask_dir: Directory containing mask files

    Returns:
        Path to matching mask file or None if not found
    """
    mask_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']

    # Try multiple possible mask filename patterns
    possible_patterns = [
        os.path.join(mask_dir, f"{rgb_name}_mask.*"),
        os.path.join(mask_dir, f"{rgb_name}_segmentation.*"),
        os.path.join(mask_dir, f"{rgb_name}_seg.*"),
        os.path.join(mask_dir, f"mask_{rgb_name}.*"),
        os.path.join(mask_dir, f"seg_{rgb_name}.*"),
        os.path.join(mask_dir, f"{rgb_name}.*"),  # Mask file with same name as RGB
    ]

    for pattern in possible_patterns:
        for ext in mask_extensions:
            search_pattern = pattern.replace('.*', ext.replace('*', ''))
            matches = glob.glob(search_pattern)
            if matches:
                return matches[0]

    return None


def create_dataset_splits_with_calib(rgb_dir, mask_dir, calib_dir, output_base_dir,
                                     extrinsics_file=None, split_ratios=(0.7, 0.15, 0.15)):
    """
    Create train/validation/test dataset splits with calibration data

    Args:
        rgb_dir: RGB image directory
        mask_dir: Mask file directory
        calib_dir: Calibration XML directory
        output_base_dir: Base output directory
        extrinsics_file: Path to fixed extrinsics file
        split_ratios: Train/val/test set ratios (train_ratio, val_ratio, test_ratio)
    """
    import random

    # Get all image pairs with calibration
    all_samples = []

    rgb_extensions = ['*.png', '*.jpg', '*.jpeg']
    rgb_files = []

    for ext in rgb_extensions:
        rgb_files.extend(glob.glob(os.path.join(rgb_dir, ext)))

    rgb_files.sort()

    for rgb_file in rgb_files:
        rgb_filename = os.path.basename(rgb_file)
        rgb_name, rgb_ext = os.path.splitext(rgb_filename)

        # Find corresponding mask file
        mask_file = find_matching_mask(rgb_name, mask_dir)
        if not mask_file:
            continue

        # Find corresponding calibration file
        calib_file = find_calibration_file(rgb_name, calib_dir)
        if not calib_file:
            continue

        all_samples.append((rgb_file, mask_file, calib_file, rgb_name))

    print(f"Found {len(all_samples)} total samples with calibration")

    if len(all_samples) == 0:
        return

    # Shuffle randomly
    random.shuffle(all_samples)

    # Calculate split sizes
    n_total = len(all_samples)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])
    n_test = n_total - n_train - n_val

    # Split dataset
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]

    print(f"Training set: {len(train_samples)} samples")
    print(f"Validation set: {len(val_samples)} samples")
    print(f"Test set: {len(test_samples)} samples")

    # Load fixed extrinsics if provided
    extrinsics = None
    if extrinsics_file and os.path.exists(extrinsics_file):
        print(f"Loading fixed extrinsics from: {extrinsics_file}")
        extrinsics = load_fixed_extrinsics(extrinsics_file)

    # Save each subset
    def save_subset(samples, subset_name):
        subset_rgb_dir = os.path.join(output_base_dir, f"{subset_name}_selection", "image_gathered")
        subset_mask_dir = os.path.join(output_base_dir, f"{subset_name}_selection", "mask_gathered")
        subset_calib_dir = os.path.join(output_base_dir, f"{subset_name}_selection", "calibration")

        os.makedirs(subset_rgb_dir, exist_ok=True)
        os.makedirs(subset_mask_dir, exist_ok=True)
        os.makedirs(subset_calib_dir, exist_ok=True)

        calibration_data = {}

        for i, (rgb_file, mask_file, calib_file, rgb_name) in enumerate(samples):
            # Parse calibration XML
            try:
                calib_data = parse_calibration_xml(calib_file)

                # Combine with extrinsics if available
                if extrinsics:
                    calib_data.update({
                        'R': extrinsics['R'].tolist(),
                        't': extrinsics['t'].flatten().tolist(),
                        'pose': extrinsics['pose'].tolist(),
                        'extrinsics_source': extrinsics_file
                    })
                else:
                    # Use identity transformation
                    calib_data.update({
                        'R': np.eye(3, dtype=np.float32).tolist(),
                        't': np.zeros(3, dtype=np.float32).tolist(),
                        'pose': np.eye(4, dtype=np.float32).tolist(),
                        'extrinsics_source': 'identity'
                    })

                # Store calibration data
                calibration_data[rgb_name] = calib_data

                # Copy RGB file
                rgb_output_path = os.path.join(subset_rgb_dir, f"{subset_name}_{i:06d}{Path(rgb_file).suffix}")
                shutil.copy(rgb_file, rgb_output_path)

                # Copy mask file
                mask_output_path = os.path.join(subset_mask_dir, f"{subset_name}_{i:06d}_mask{Path(mask_file).suffix}")
                shutil.copy(mask_file, mask_output_path)

                # Copy calibration file
                calib_output_path = os.path.join(subset_calib_dir, f"{subset_name}_{i:06d}_calib.xml")
                shutil.copy(calib_file, calib_output_path)

            except Exception as e:
                print(f"❌ Error processing sample {rgb_name}: {e}")
                continue

        # Save calibration data to JSON
        if calibration_data:
            calibration_json_path = os.path.join(subset_calib_dir, "calibration_data.json")
            with open(calibration_json_path, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            print(f"Saved {subset_name} calibration data to: {calibration_json_path}")

    save_subset(train_samples, "train")
    save_subset(val_samples, "val")
    save_subset(test_samples, "test")

    print("✅ Dataset splitting with calibration complete!")


def visualize_sample_with_calib(rgb_dir, mask_dir, calib_dir, num_samples=5):
    """
    Visualize sample images with calibration information

    Args:
        rgb_dir: RGB image directory
        mask_dir: Mask file directory
        calib_dir: Calibration XML directory
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt

    rgb_extensions = ['*.png', '*.jpg', '*.jpeg']
    rgb_files = []

    for ext in rgb_extensions:
        rgb_files.extend(glob.glob(os.path.join(rgb_dir, ext)))

    rgb_files.sort()

    if len(rgb_files) == 0:
        print("No RGB images found")
        return

    num_samples = min(num_samples, len(rgb_files))

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 3 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        rgb_file = rgb_files[i]
        rgb_filename = os.path.basename(rgb_file)
        rgb_name, rgb_ext = os.path.splitext(rgb_filename)

        # Load RGB image
        rgb_img = Image.open(rgb_file)
        rgb_array = np.array(rgb_img)

        # Find and load mask
        mask_file = find_matching_mask(rgb_name, mask_dir)
        if mask_file:
            mask_array = mask_read(mask_file)
        else:
            mask_array = np.zeros(rgb_array.shape[:2])

        # Find and parse calibration
        calib_file = find_calibration_file(rgb_name, calib_dir)
        calib_info = ""
        if calib_file:
            try:
                calib_data = parse_calibration_xml(calib_file)
                calib_info = f"K: [{calib_data['fx']:.1f}, {calib_data['fy']:.1f}]"
            except Exception as e:
                calib_info = f"Calib error: {e}"
        else:
            calib_info = "No calibration"

        # Display RGB image
        axes[i, 0].imshow(rgb_array)
        axes[i, 0].set_title(f"RGB: {rgb_filename[:20]}...")
        axes[i, 0].axis('off')

        # Display mask
        axes[i, 1].imshow(mask_array, cmap='gray')
        axes[i, 1].set_title(f"Mask")
        axes[i, 1].axis('off')

        # Display calibration info
        axes[i, 2].text(0.1, 0.5, f"Calibration:\n{calib_info}",
                        fontsize=10, verticalalignment='center')
        axes[i, 2].set_title("Calibration Info")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to parse command line arguments and execute appropriate function
    """
    parser = argparse.ArgumentParser(description='Process RGB, mask, and calibration dataset')
    parser.add_argument('--rgb_dir', type=str, required=True,
                        help='Path to RGB image directory')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Path to mask file directory')
    parser.add_argument('--calib_dir', type=str, required=True,
                        help='Path to calibration XML directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--extrinsics_file', type=str, default=None,
                        help='Path to fixed extrinsics file (JSON or TXT)')
    parser.add_argument('--subset', type=str, default='train',
                        help='Dataset subset name (train/val/test)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples')
    parser.add_argument('--split', action='store_true',
                        help='Automatically split dataset')
    parser.add_argument('--split_ratios', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                        help='Train/val/test set ratios')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize samples with calibration')

    args = parser.parse_args()

    # Visualize samples if requested
    if args.visualize:
        visualize_sample_with_calib(args.rgb_dir, args.mask_dir, args.calib_dir)
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.split:
        # Automatically split dataset with calibration
        create_dataset_splits_with_calib(
            args.rgb_dir, args.mask_dir, args.calib_dir, args.output_dir,
            args.extrinsics_file, args.split_ratios
        )
    else:
        # Directly gather data with calibration
        gather_rgb_mask_calib_data(
            args.rgb_dir, args.mask_dir, args.calib_dir, args.output_dir,
            args.extrinsics_file, args.subset, args.max_samples
        )


if __name__ == "__main__":
    main()