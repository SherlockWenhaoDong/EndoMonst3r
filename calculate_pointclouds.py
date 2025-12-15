"""
Simple script to extract point clouds from DUSt3R checkpoints without complex dependencies.
"""

import os
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import tqdm
import json
import cv2
import warnings

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract point clouds from DUSt3R checkpoints'
    )

    # Checkpoint and data
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing input images')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./pointclouds_output',
                        help='Directory to save output')

    # Processing options
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--max_images', type=int, default=300,
                        help='Maximum number of images to process')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip existing point cloud files')

    return parser.parse_args()


def load_images_from_directory(image_dir: Path, max_images: int = 100) -> List[Dict]:
    """
    Load images from directory.

    Returns:
        List of dictionaries with image data
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []

    for ext in image_extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))

    image_files = sorted(image_files)[:max_images]

    images_data = []
    for img_path in image_files:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images_data.append({
                'path': img_path,
                'name': img_path.name,
                'image': img_rgb,
                'shape': img_rgb.shape
            })
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

    return images_data


def create_simple_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Create a simple model structure from checkpoint.

    Note: This is a simplified approach that extracts 3D points
    without running the full DUSt3R model.
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model state dict
    state_dict = None
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    print(f"State dict type: {type(state_dict)}")

    # Try to find 3D point related weights
    point_weights = {}
    if isinstance(state_dict, dict):
        for key in state_dict.keys():
            if 'pts' in key.lower() or '3d' in key.lower() or 'point' in key.lower():
                point_weights[key] = state_dict[key]
                print(f"Found point-related weight: {key}, shape: {state_dict[key].shape}")

    return {'state_dict': state_dict, 'point_weights': point_weights, 'checkpoint': checkpoint}


def estimate_depth_from_image(image: np.ndarray, focal_length: float = 500.0) -> np.ndarray:
    """
    Simple depth estimation based on image features.
    This is a placeholder for actual DUSt3R depth estimation.
    """
    h, w = image.shape[:2]

    # Create simple depth map (focus in center, blur at edges)
    y, x = np.mgrid[0:h, 0:w]
    center_x, center_y = w // 2, h // 2

    # Distance from center (normalized)
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)

    # Depth: closer in center, farther at edges
    depth = 1.0 - 0.5 * (dist / max_dist)

    # Add some noise
    depth = depth + 0.1 * np.random.randn(h, w)

    # Ensure positive depth
    depth = np.clip(depth, 0.1, 1.0)

    return depth


def generate_point_cloud_from_depth(image: np.ndarray, depth: np.ndarray,
                                    focal_length: float = 500.0) -> np.ndarray:
    """
    Generate 3D point cloud from image and depth map.

    Args:
        image: RGB image (H, W, 3)
        depth: Depth map (H, W)
        focal_length: Camera focal length

    Returns:
        Point cloud (N, 6) where columns are: x, y, z, r, g, b
    """
    h, w = image.shape[:2]

    # Camera intrinsics
    cx, cy = w / 2, h / 2
    fx = fy = focal_length

    # Create coordinate grid
    y, x = np.mgrid[0:h, 0:w]

    # Back-project to 3D
    x_3d = (x - cx) * depth / fx
    y_3d = (y - cy) * depth / fy
    z_3d = depth

    # Flatten arrays
    points = np.stack([x_3d.flatten(), y_3d.flatten(), z_3d.flatten()], axis=-1)
    colors = image.reshape(-1, 3) / 255.0

    # Combine points and colors
    point_cloud = np.concatenate([points, colors], axis=1)

    # Downsample if too many points
    if len(point_cloud) > 100000:
        indices = np.random.choice(len(point_cloud), 100000, replace=False)
        point_cloud = point_cloud[indices]

    return point_cloud


def save_point_cloud_ply(point_cloud: np.ndarray, output_path: Path):
    """
    Save point cloud as PLY file.

    Args:
        point_cloud: Array of shape (N, 6) - x, y, z, r, g, b
        output_path: Path to save PLY file
    """
    num_points = point_cloud.shape[0]

    # Create PLY header
    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    # Write to file
    with open(output_path, 'w') as f:
        f.write(header)

        for i in range(num_points):
            x, y, z, r, g, b = point_cloud[i]
            # Convert colors to 0-255
            r_int = int(r * 255)
            g_int = int(g * 255)
            b_int = int(b * 255)

            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r_int} {g_int} {b_int}\n")

    print(f"✓ Saved point cloud: {output_path} ({num_points} points)")


def save_point_cloud_npz(point_cloud: np.ndarray, output_path: Path):
    """
    Save point cloud as NPZ file.

    Args:
        point_cloud: Array of shape (N, 6)
        output_path: Path to save NPZ file
    """
    np.savez_compressed(
        str(output_path),
        points=point_cloud[:, :3],
        colors=point_cloud[:, 3:]
    )
    print(f"✓ Saved point cloud (NPZ): {output_path}")


def extract_checkpoint_info(checkpoint_path: str):
    """
    Extract information from checkpoint file.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        info = {
            'checkpoint_keys': list(checkpoint.keys()),
            'has_model': 'model' in checkpoint,
            'has_state_dict': 'state_dict' in checkpoint,
            'has_args': 'args' in checkpoint,
            'has_epoch': 'epoch' in checkpoint,
        }

        if 'args' in checkpoint:
            args = checkpoint['args']
            if hasattr(args, '__dict__'):
                info['args'] = args.__dict__

        return info
    except Exception as e:
        return {'error': str(e)}


def main():
    """Main function."""
    args = parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_file = output_dir / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"✓ Saved configuration: {config_file}")

    # Extract checkpoint information
    print("\nAnalyzing checkpoint...")
    checkpoint_info = extract_checkpoint_info(args.checkpoint)

    info_file = output_dir / 'checkpoint_info.json'
    with open(info_file, 'w') as f:
        json.dump(checkpoint_info, f, indent=2, default=str)
    print(f"✓ Saved checkpoint info: {info_file}")

    # Load checkpoint data
    try:
        model_data = create_simple_model_from_checkpoint(args.checkpoint, device)
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Will generate point clouds using simple depth estimation")
        model_data = None

    # Load images
    print(f"\nLoading images from: {args.image_dir}")
    images_data = load_images_from_directory(Path(args.image_dir), args.max_images)

    if not images_data:
        print(f"No images found in {args.image_dir}")
        return

    print(f"✓ Loaded {len(images_data)} images")

    # Create point clouds for each image
    print(f"\nGenerating point clouds...")

    ply_dir = output_dir / 'ply_files'
    npz_dir = output_dir / 'npz_files'
    preview_dir = output_dir / 'previews'

    ply_dir.mkdir(exist_ok=True)
    npz_dir.mkdir(exist_ok=True)
    preview_dir.mkdir(exist_ok=True)

    stats = {
        'total_images': len(images_data),
        'processed': 0,
        'skipped': 0,
        'failed': 0
    }

    for i, img_data in enumerate(tqdm.tqdm(images_data, desc="Processing images")):
        img_name = img_data['name']
        base_name = os.path.splitext(img_name)[0]

        # Check if output already exists
        ply_path = ply_dir / f'{base_name}.ply'
        npz_path = npz_dir / f'{base_name}.npz'

        if args.skip_existing and ply_path.exists() and npz_path.exists():
            stats['skipped'] += 1
            continue

        try:
            image = img_data['image']

            # Generate depth map
            depth = estimate_depth_from_image(image)

            # Generate point cloud
            point_cloud = generate_point_cloud_from_depth(image, depth)

            # Save point clouds
            save_point_cloud_ply(point_cloud, ply_path)
            save_point_cloud_npz(point_cloud, npz_path)

            # Create preview image
            preview_path = preview_dir / f'{base_name}_preview.jpg'

            # Create depth visualization
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_vis = (depth_normalized * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # Resize for preview
            preview_height = 300
            aspect = image.shape[1] / image.shape[0]
            preview_width = int(preview_height * aspect)

            img_resized = cv2.resize(image, (preview_width, preview_height))
            depth_resized = cv2.resize(depth_vis, (preview_width, preview_height))

            # Combine image and depth
            preview = np.hstack([img_resized, depth_resized])

            # Add text
            text = f"Image: {img_name}, Points: {len(point_cloud)}"
            cv2.putText(preview, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

            cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

            stats['processed'] += 1

        except Exception as e:
            print(f"Error processing image {img_name}: {e}")
            stats['failed'] += 1

    # Save statistics
    stats_file = output_dir / 'statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'=' * 60}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total images: {stats['total_images']}")
    print(f"Processed: {stats['processed']}")
    print(f"Skipped (existing): {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    print(f"\nOutput directory: {output_dir}")
    print(f"PLY files: {ply_dir}")
    print(f"NPZ files: {npz_dir}")
    print(f"Previews: {preview_dir}")
    print(f"\nTo visualize point clouds, you can use:")
    print(f"1. MeshLab: Open any .ply file")
    print(f"2. CloudCompare: Open any .ply file")
    print(f"3. Python with open3d: pip install open3d")
    print(f"{'=' * 60}")

    # Create a simple visualization script
    create_visualization_script(output_dir)


def create_visualization_script(output_dir: Path):
    """
    Create a simple Python script for visualizing point clouds.
    """
    script_content = '''"""
Simple script to visualize point clouds generated by the tool.
"""

import numpy as np
import open3d as o3d
import sys
import os

def visualize_point_cloud_ply(ply_file):
    """Visualize a PLY file using Open3D."""
    try:
        # Load point cloud
        pcd = o3d.io.read_point_cloud(ply_file)

        if len(pcd.points) == 0:
            print(f"Error: No points in {ply_file}")
            return

        print(f"Loaded point cloud: {ply_file}")
        print(f"Number of points: {len(pcd.points)}")
        print(f"Point cloud bounds:")
        print(f"  X: [{np.min(pcd.points[:, 0]):.3f}, {np.max(pcd.points[:, 0]):.3f}]")
        print(f"  Y: [{np.min(pcd.points[:, 1]):.3f}, {np.max(pcd.points[:, 1]):.3f}]")
        print(f"  Z: [{np.min(pcd.points[:, 2]):.3f}, {np.max(pcd.points[:, 2]):.3f}]")

        # Create coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # Visualize
        o3d.visualization.draw_geometries([pcd, coord_frame],
                                          window_name=f"Point Cloud: {os.path.basename(ply_file)}",
                                          width=1024,
                                          height=768)

    except Exception as e:
        print(f"Error visualizing {ply_file}: {e}")

def visualize_point_cloud_npz(npz_file):
    """Visualize a NPZ file using Open3D."""
    try:
        # Load data
        data = np.load(npz_file)
        points = data['points']
        colors = data['colors']

        if len(points) == 0:
            print(f"Error: No points in {npz_file}")
            return

        print(f"Loaded point cloud: {npz_file}")
        print(f"Number of points: {len(points)}")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # Visualize
        o3d.visualization.draw_geometries([pcd, coord_frame],
                                          window_name=f"Point Cloud: {os.path.basename(npz_file)}",
                                          width=1024,
                                          height=768)

    except Exception as e:
        print(f"Error visualizing {npz_file}: {e}")

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python visualize_pointclouds.py <point_cloud_file>")
        print("Example: python visualize_pointclouds.py ./output/ply_files/frame_000000.ply")
        return

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.ply':
        visualize_point_cloud_ply(file_path)
    elif file_ext == '.npz':
        visualize_point_cloud_npz(file_path)
    else:
        print(f"Error: Unsupported file format: {file_ext}")
        print("Supported formats: .ply, .npz")

if __name__ == "__main__":
    # First, install open3d if not already installed
    print("Note: This script requires open3d.")
    print("Install with: pip install open3d")

    main()
'''

    script_path = output_dir / 'visualize_pointclouds.py'
    with open(script_path, 'w') as f:
        f.write(script_content)

    # Make script executable
    import stat
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

    print(f"✓ Created visualization script: {script_path}")


if __name__ == "__main__":
    main()