import time
import sys
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
import glob
import re
import json

import numpy as onp
from tqdm.auto import tqdm

import viser
import viser.extras
import viser.transforms as tf
import matplotlib.cm as cm
import open3d as o3d
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def parse_calibration_xml(xml_path: Path):
    """
    Parse camera calibration XML file to extract intrinsic parameters.

    Args:
        xml_path: Path to calibration XML file

    Returns:
        dict: Camera intrinsic parameters
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"Calibration XML file not found: {xml_path}")

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        calibration_data = {}

        # Parse image size
        image_size_elem = root.find("./param[@name='imageSize']")
        if image_size_elem is not None:
            size_values = list(map(float, image_size_elem.text.strip().split()))
            if len(size_values) >= 2:
                calibration_data['image_width'] = int(size_values[0])
                calibration_data['image_height'] = int(size_values[1])

        # Parse intrinsic matrix K
        K_elem = root.find("./param[@name='K']")
        if K_elem is not None:
            K_values = list(map(float, K_elem.text.strip().split()))
            if len(K_values) >= 9:
                K = onp.array(K_values, dtype=onp.float32).reshape(3, 3)
                calibration_data['K'] = K
                calibration_data['fx'] = K[0, 0]
                calibration_data['fy'] = K[1, 1]
                calibration_data['cx'] = K[0, 2]
                calibration_data['cy'] = K[1, 2]

        # Parse distortion coefficients
        distortion_elem = root.find("./param[@name='distortion']")
        if distortion_elem is not None:
            distortion_values = list(map(float, distortion_elem.text.strip().split()))
            calibration_data['distortion'] = onp.array(distortion_values, dtype=onp.float32)

        # Parse registration matrix
        registration_elem = root.find("./param[@name='registration']")
        if registration_elem is not None:
            reg_values = list(map(float, registration_elem.text.strip().split()))
            if len(reg_values) >= 16:
                registration = onp.array(reg_values, dtype=onp.float32).reshape(4, 4)
                calibration_data['registration'] = registration

        return calibration_data

    except Exception as e:
        print(f"Error parsing calibration XML {xml_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


class FrameData:
    """Simple class to hold frame data."""

    def __init__(self, rgb, depth=None, mask=None, K=None, pose=None, rgb_path=None):
        self.rgb = rgb  # RGB image (with mask applied)
        self.depth = depth
        self.mask = mask  # mask
        self.K = K
        self.T_world_camera = pose if pose is not None else onp.eye(4, dtype=onp.float32)
        self.rgb_path = rgb_path

    def get_point_cloud_with_mask(self, downsample_factor=1, mask_threshold=0.5,
                                  filter_by_mask=True):
        """
        Get point cloud filtered by mask.

        Args:
            downsample_factor: Downsample factor for points
            mask_threshold: Threshold for mask values
            filter_by_mask: Whether to filter points using mask

        Returns:
            Tuple of (foreground_positions, foreground_colors,
                     background_positions, background_colors)
        """
        if self.depth is None or self.K is None:
            # If no depth map, return empty arrays
            return onp.zeros((0, 3)), onp.zeros((0, 3)), onp.zeros((0, 3)), onp.zeros((0, 3))

        h, w = self.depth.shape[:2]

        # Create coordinate grid
        yy, xx = onp.meshgrid(onp.arange(h), onp.arange(w), indexing='ij')

        # Downsample if needed
        if downsample_factor > 1:
            yy = yy[::downsample_factor, ::downsample_factor]
            xx = xx[::downsample_factor, ::downsample_factor]
            depth = self.depth[::downsample_factor, ::downsample_factor]

            if self.mask is not None:
                mask = self.mask[::downsample_factor, ::downsample_factor]
            else:
                mask = None

            if self.rgb is not None:
                rgb = self.rgb[::downsample_factor, ::downsample_factor]
            else:
                rgb = onp.zeros((depth.shape[0], depth.shape[1], 3))
        else:
            depth = self.depth
            mask = self.mask
            rgb = self.rgb if self.rgb is not None else onp.zeros((h, w, 3))

        # Flatten arrays
        xx = xx.flatten()
        yy = yy.flatten()
        depth_flat = depth.flatten()

        if mask is not None:
            mask_flat = mask.flatten()
        else:
            mask_flat = None

        rgb_flat = rgb.reshape(-1, 3)

        # Filter by mask if mask exists and filter_by_mask is True
        if filter_by_mask and mask is not None:
            # Use mask for filtering
            foreground_mask = mask_flat > mask_threshold
            background_mask = ~foreground_mask
        else:
            # If not using mask or no mask, treat all points as foreground
            foreground_mask = onp.ones_like(depth_flat, dtype=bool)
            background_mask = onp.zeros_like(depth_flat, dtype=bool)

        # Filter valid depth points (depth > 0)
        valid_depth_mask = depth_flat > 0

        # Combine mask filtering and depth filtering
        if filter_by_mask and mask is not None:
            foreground_mask = foreground_mask & valid_depth_mask
            background_mask = background_mask & valid_depth_mask
        else:
            foreground_mask = valid_depth_mask
            background_mask = onp.zeros_like(depth_flat, dtype=bool)

        # Filter points
        xx_fg = xx[foreground_mask]
        yy_fg = yy[foreground_mask]
        depth_fg = depth_flat[foreground_mask]
        rgb_fg = rgb_flat[foreground_mask]

        xx_bg = xx[background_mask]
        yy_bg = yy[background_mask]
        depth_bg = depth_flat[background_mask]
        rgb_bg = rgb_flat[background_mask]

        # Backproject to 3D
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        if len(xx_fg) > 0:
            # Foreground points
            z_fg = depth_fg
            x_fg = (xx_fg - cx) * z_fg / fx
            y_fg = (yy_fg - cy) * z_fg / fy
            positions_fg = onp.stack([x_fg, y_fg, z_fg], axis=-1)

            # Transform to world coordinates
            R = self.T_world_camera[:3, :3]
            t = self.T_world_camera[:3, 3]
            positions_fg = (R @ positions_fg.T).T + t
            colors_fg = rgb_fg / 255.0  # Normalize to [0, 1]
        else:
            positions_fg = onp.zeros((0, 3))
            colors_fg = onp.zeros((0, 3))

        if len(xx_bg) > 0:
            # Background points
            z_bg = depth_bg
            x_bg = (xx_bg - cx) * z_bg / fx
            y_bg = (yy_bg - cy) * z_bg / fy
            positions_bg = onp.stack([x_bg, y_bg, z_bg], axis=-1)

            # Transform to world coordinates
            positions_bg = (R @ positions_bg.T).T + t
            colors_bg = rgb_bg / 255.0  # Normalize to [0, 1]
        else:
            positions_bg = onp.zeros((0, 3))
            colors_bg = onp.zeros((0, 3))

        return positions_fg, colors_fg, positions_bg, colors_bg


def load_frame_data(data_path: Path, rgb_path: Path, calibration_data=None, fixed_pose=True, verbose=False,
                    apply_mask_to_image=True):
    """
    Load frame data including RGB, depth, mask, and camera parameters.

    Args:
        data_path: Path to data directory (parent directory)
        rgb_path: Path to RGB image
        calibration_data: Calibration data from XML
        fixed_pose: Whether to use fixed identity pose
        verbose: Whether to print verbose information
        apply_mask_to_image: Whether to apply mask to RGB image

    Returns:
        FrameData object or None if loading fails
    """
    if verbose:
        print(f"Loading frame from: {rgb_path}")

    # Extract base name without extension
    rgb_filename = rgb_path.name
    rgb_name, rgb_ext = os.path.splitext(rgb_filename)

    # Load RGB image
    rgb_original = None
    try:
        rgb_original = cv2.imread(str(rgb_path))
        if rgb_original is not None:
            rgb_original = cv2.cvtColor(rgb_original, cv2.COLOR_BGR2RGB)

            # Ensure dimensions are divisible by 16 (same as EndoDUSt3R)
            h, w = rgb_original.shape[:2]
            h = h - (h % 16)
            w = w - (w % 16)
            rgb_original = rgb_original[:h, :w]
    except Exception as e:
        print(f"Error loading RGB image {rgb_path}: {e}")
        return None

    if rgb_original is None:
        if verbose:
            print(f"  Warning: Could not load RGB image")
        return None

    # Find depth image
    depth_path = None
    possible_depth_dirs = [
        data_path / 'depth',
        data_path / 'depths',
        data_path.parent / 'depth',
        data_path.parent / 'depths',
        data_path
    ]

    for depth_dir in possible_depth_dirs:
        if depth_dir.exists():
            possible_depth_files = [
                depth_dir / f'{rgb_name}.png',
                depth_dir / f'{rgb_name}_depth.png',
                depth_dir / f'{rgb_name}_depth.jpg',
                depth_dir / f'depth_{rgb_name}.png',
                depth_dir / f'depth_{rgb_name}.jpg',
            ]
            for depth_file in possible_depth_files:
                if depth_file.exists():
                    depth_path = depth_file
                    break
        if depth_path:
            break

    # Load depth image
    depth = None
    if depth_path and depth_path.exists():
        try:
            depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                # Crop depth map to match RGB image
                h, w = rgb_original.shape[:2]
                depth_img = depth_img[:h, :w]

                if len(depth_img.shape) == 2:
                    if depth_img.dtype == np.uint16:
                        depth = depth_img.astype(onp.float32) / 1000.0  # Convert mm to m
                    elif depth_img.dtype == np.uint8:
                        depth = depth_img.astype(onp.float32) / 255.0 * 10.0  # Assume max depth 10m
                    else:
                        depth = depth_img.astype(onp.float32)
                elif len(depth_img.shape) == 3:
                    # If 3-channel depth image, convert to grayscale
                    depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
                    depth = depth_img.astype(onp.float32) / 255.0 * 10.0

                if verbose:
                    print(f"  Loaded depth from: {depth_path}")
                    print(f"  Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}] m")

        except Exception as e:
            print(f"Error loading depth image {depth_path}: {e}")
    else:
        if verbose:
            print(f"  Warning: No depth image found for {rgb_name}")

    # Find mask image
    mask_path = None
    mask_dir = Path(str(data_path) + '_mask')
    mask_file = mask_dir / f'{rgb_name}_combined_masks.png'

    if mask_file.exists():
        mask_path = mask_file
    else:
        possible_mask_files_in_dir = [
            mask_dir / f'{rgb_name}_combined_masks_png.png',
            mask_dir / f'{rgb_name}_combined_masks.jpg',
            mask_dir / f'{rgb_name}_mask.png',
            mask_dir / f'{rgb_name}_mask.jpg',
            mask_dir / f'{rgb_name}.png',
            mask_dir / f'{rgb_name}.jpg',
        ]
        for possible_file in possible_mask_files_in_dir:
            if possible_file.exists():
                mask_path = possible_file
                break

    if mask_path is None:
        possible_mask_dirs = [
            data_path.parent / 'mask',
            data_path.parent / 'masks',
            data_path / 'mask',
            data_path / 'masks',
            data_path
        ]

        for possible_mask_dir in possible_mask_dirs:
            if possible_mask_dir.exists():
                possible_mask_files = [
                    possible_mask_dir / f'{rgb_name}_combined_masks.png',
                    possible_mask_dir / f'{rgb_name}_combined_masks_png.png',
                    possible_mask_dir / f'{rgb_name}_combined_masks.jpg',
                    possible_mask_dir / f'{rgb_name}_mask.png',
                    possible_mask_dir / f'{rgb_name}_mask.jpg',
                    possible_mask_dir / f'{rgb_name}.png',
                    possible_mask_dir / f'{rgb_name}.jpg',
                ]
                for possible_file in possible_mask_files:
                    if possible_file.exists():
                        mask_path = possible_file
                        break
            if mask_path:
                break

    # Load mask
    mask = None
    if mask_path and mask_path.exists():
        try:
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                # Crop mask to match RGB image
                h, w = rgb_original.shape[:2]
                mask_img = mask_img[:h, :w]
                mask = (mask_img > 0).astype(onp.float32)

                if verbose:
                    print(f"  Loaded mask from: {mask_path}")
                    print(f"  Mask shape: {mask.shape}")
                    print(f"  Mask values: unique {onp.unique(mask)}")

        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
    else:
        if verbose:
            print(f"  Warning: No mask found for {rgb_name}")

    # Apply mask to RGB image
    rgb_with_mask = rgb_original.copy()

    if apply_mask_to_image and mask is not None:
        if len(mask.shape) == 2:
            mask_3d = onp.repeat(mask[:, :, onp.newaxis], 3, axis=2)
        else:
            mask_3d = mask
        rgb_with_mask = (rgb_original * mask_3d).astype(onp.uint8)
        if verbose:
            print(f"  Applied mask to RGB image")

    # Set camera parameters
    if calibration_data and 'K' in calibration_data:
        K = calibration_data['K'].copy()

        # Adjust intrinsics for cropped image
        h, w = rgb_original.shape[:2]
        original_height = calibration_data.get('image_height', h)
        original_width = calibration_data.get('image_width', w)

        scale_x = w / original_width if original_width > 0 else 1
        scale_y = h / original_height if original_height > 0 else 1
        K[0, 0] *= scale_x  # fx
        K[1, 1] *= scale_y  # fy
        K[0, 2] *= scale_x  # cx
        K[1, 2] *= scale_y  # cy
    else:
        # Default intrinsics if no calibration
        h, w = rgb_original.shape[:2]
        K = onp.array([[0.5 * w, 0, 0.5 * w],
                       [0, 0.5 * h, 0.5 * h],
                       [0, 0, 1]], dtype=onp.float32)

    # Set camera pose
    if fixed_pose:
        pose = onp.eye(4, dtype=onp.float32)
        if calibration_data and 'registration' in calibration_data:
            pose = calibration_data['registration'].copy()
    else:
        # You might want to load actual poses from file if available
        pose = onp.eye(4, dtype=onp.float32)

    # Create FrameData object
    frame_data = FrameData(
        rgb=rgb_with_mask,
        depth=depth,
        mask=mask,
        K=K,
        pose=pose,
        rgb_path=rgb_path
    )

    return frame_data


class SimpleMultiViewRegistration:
    """
    Simple registration class that loads pre-aligned point clouds.
    Only shows two views (cam0 and cam1) aligned to cam0 coordinate system.
    """

    def __init__(self, registration_dir: Path, verbose: bool = True):
        """
        Initialize with registration results directory.

        Args:
            registration_dir: Directory containing registration results
            verbose: Whether to print verbose information
        """
        self.registration_dir = Path(registration_dir)
        self.verbose = verbose
        self.view_data = {}

    def load_aligned_pointclouds(self, views: list = ['cam0', 'cam1'], max_frames: int = 30):
        """
        Load aligned point clouds from disk.

        Args:
            views: List of views to load (default: ['cam0', 'cam1'])
            max_frames: Maximum number of frames per view to load

        Returns:
            dict: Dictionary containing loaded point cloud data
        """
        aligned_dir = self.registration_dir / 'aligned_pointclouds'
        if not aligned_dir.exists():
            raise FileNotFoundError(f"Aligned point clouds directory not found: {aligned_dir}")

        for view_name in views:
            view_dir = aligned_dir / view_name
            if not view_dir.exists():
                if self.verbose:
                    print(f"Warning: View directory not found: {view_dir}")
                continue

            self.view_data[view_name] = {
                'name': view_name,
                'frames': []
            }

            # Find aligned PLY files
            ply_files = sorted(glob.glob(str(view_dir / '*_aligned.ply')))

            for ply_file in ply_files[:max_frames]:
                try:
                    pcd = o3d.io.read_point_cloud(ply_file)
                    if len(pcd.points) > 0:
                        # Extract frame number
                        frame_idx = self.extract_frame_number(Path(ply_file).stem)

                        # Get point cloud data
                        points = np.asarray(pcd.points)
                        colors = np.asarray(pcd.colors) if pcd.has_colors() else None

                        self.view_data[view_name]['frames'].append({
                            'frame_idx': frame_idx,
                            'points': points,
                            'colors': colors,
                            'path': ply_file
                        })
                except Exception as e:
                    if self.verbose:
                        print(f"Error loading {ply_file}: {e}")

            if self.verbose:
                print(f"  {view_name}: {len(self.view_data[view_name]['frames'])} frames loaded")

        return self.view_data

    def extract_frame_number(self, filename: str) -> int:
        """Extract frame number from filename."""
        patterns = [r'frame_(\d+)', r'(\d+)']
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        return 0


def main(
        data_path: Path = Path("./demo_tmp/NULL"),
        data_path2: Path = None,
        calib_xml_path: Path = None,
        calib_xml_path2: Path = None,
        downsample_factor: int = 1,
        max_frames: int = 100,
        share: bool = False,
        point_size: float = 0.001,
        camera_frustum_scale: float = 0.02,
        axes_scale: float = 0.25,
        cam_thickness: float = 1.5,
        fixed_pose: bool = True,
        mask_threshold: float = 0.5,
        show_background: bool = False,
        verbose: bool = False,
        stride: int = 1,
        apply_mask_to_image: bool = True,
        filter_by_mask: bool = True,
        show_registration: bool = False,
        registration_dir: Path = None,
        registration_views: list = None,
        max_registration_frames: int = 30,
) -> None:
    """
    Integrated visualization system for comparing two datasets and registration results.

    This visualizer can display:
    1. Original RGB-D data with mask filtering
    2. Registration results (two views aligned to cam0 coordinate system)

    All in the same 3D viewer for easy comparison.
    """
    # Initialize server
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    server.scene.set_up_direction('-z')

    print("=" * 70)
    print("INTEGRATED REGISTRATION VISUALIZATION SYSTEM")
    print("Displaying two views registration results in cam0 coordinate system")
    print("=" * 70)

    # Load calibration data
    calibration_data = None
    calibration_data2 = None

    if calib_xml_path and calib_xml_path.exists():
        print(f"\nLoading calibration from: {calib_xml_path}")
        calibration_data = parse_calibration_xml(calib_xml_path)
        if calibration_data:
            print(
                f"  Image size: {calibration_data.get('image_width', 'N/A')}x{calibration_data.get('image_height', 'N/A')}")

    if calib_xml_path2 and calib_xml_path2.exists():
        print(f"\nLoading calibration for second dataset from: {calib_xml_path2}")
        calibration_data2 = parse_calibration_xml(calib_xml_path2)
        if calibration_data2:
            print(
                f"  Image size: {calibration_data2.get('image_width', 'N/A')}x{calibration_data2.get('image_height', 'N/A')}")

    # Find RGB images
    def find_rgb_files(data_path):
        """Find RGB image files in the given directory."""
        if 'frames_cam0' in str(data_path):
            rgb_files_dir = data_path
        elif (data_path / 'frames_cam0').exists():
            rgb_files_dir = data_path / 'frames_cam0'
        else:
            frames_dirs = list(data_path.glob('*/frames_cam0'))
            if frames_dirs:
                rgb_files_dir = frames_dirs[0]
                print(f"Found frames directory: {rgb_files_dir}")
            else:
                rgb_files_dir = data_path

        print(f"Looking for RGB images in: {rgb_files_dir}")

        rgb_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        rgb_files = []
        for ext in rgb_extensions:
            rgb_files.extend(glob.glob(str(rgb_files_dir / ext)))

        def extract_number(filename):
            """Extract numeric part from filename for sorting."""
            numbers = re.findall(r'\d+', os.path.basename(filename))
            return int(numbers[0]) if numbers else 0

        rgb_files = sorted(rgb_files, key=extract_number)
        rgb_files = [Path(f) for f in rgb_files]

        if not rgb_files:
            print(f"No RGB images found in {rgb_files_dir}")
            return None, []

        return rgb_files_dir, rgb_files

    # Load first dataset
    print(f"\n{'=' * 40}")
    print(f"LOADING FIRST DATASET: {data_path}")
    print(f"{'=' * 40}")

    rgb_files_dir1, rgb_files1 = find_rgb_files(data_path)
    if not rgb_files1:
        print("No RGB files found for first dataset")
        return

    rgb_files1 = rgb_files1[::stride]
    rgb_files1 = rgb_files1[:max_frames]
    num_frames1 = len(rgb_files1)
    print(f"Found {num_frames1} frames (with stride={stride})")

    # Load second dataset (if provided)
    num_frames2 = 0
    rgb_files_dir2 = None
    rgb_files2 = []

    if data_path2:
        print(f"\n{'=' * 40}")
        print(f"LOADING SECOND DATASET: {data_path2}")
        print(f"{'=' * 40}")

        rgb_files_dir2, rgb_files2 = find_rgb_files(data_path2)
        if rgb_files2:
            rgb_files2 = rgb_files2[::stride]
            rgb_files2 = rgb_files2[:max_frames]
            num_frames2 = len(rgb_files2)
            print(f"Found {num_frames2} frames (with stride={stride})")

    total_frames = max(num_frames1, num_frames2)
    if total_frames == 0:
        print("No frames to display")
        return

    # Load registration data if requested
    registration_viewer = None
    if show_registration and registration_dir:
        print(f"\n{'=' * 40}")
        print(f"LOADING REGISTRATION RESULTS")
        print(f"{'=' * 40}")

        # Use default views if not specified
        if registration_views is None:
            registration_views = ['cam0', 'cam1']

        print(f"Registration views: {registration_views}")

        try:
            registration_viewer = SimpleMultiViewRegistration(
                registration_dir=registration_dir,
                verbose=verbose
            )

            registration_viewer.load_aligned_pointclouds(
                views=registration_views,
                max_frames=max_registration_frames
            )

            print(f"Successfully loaded registration data for {len(registration_viewer.view_data)} views")

        except Exception as e:
            print(f"Error loading registration data: {e}")
            show_registration = False

    # Create UI
    print(f"\n{'=' * 40}")
    print(f"CREATING USER INTERFACE")
    print(f"{'=' * 40}")

    # Add playback controls
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=total_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", False)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=30
        )
        gui_show_all_frames = server.gui.add_checkbox("Show all frames", False)

    # Add dataset visibility controls
    with server.gui.add_folder("Data Visibility"):
        gui_show_dataset1 = server.gui.add_checkbox("Show Dataset 1", True)
        gui_show_dataset2 = server.gui.add_checkbox("Show Dataset 2", True) if data_path2 else None

        # Add registration visibility control
        if show_registration:
            gui_show_registration = server.gui.add_checkbox("Show Registration", True)

    # Add visualization settings
    with server.gui.add_folder("Visualization Settings"):
        gui_point_size = server.gui.add_slider(
            "Point Size",
            min=0.0001,
            max=0.01,
            step=0.0001,
            initial_value=point_size,
        )
        gui_camera_frustum_scale = server.gui.add_slider(
            "Camera Frustum Scale",
            min=0.001,
            max=0.1,
            step=0.001,
            initial_value=camera_frustum_scale,
        )

    # Store nodes
    frame_nodes1 = []  # Dataset 1 nodes
    frame_nodes2 = []  # Dataset 2 nodes
    pointcloud_nodes1 = []
    pointcloud_nodes2 = []
    registration_nodes = {}  # Registration result nodes

    # Create coordinate axes at origin
    server.scene.add_frame(
        "/origin",
        wxyz=tf.SO3.exp(onp.array([onp.pi / 2.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        axes_length=0.1,
        axes_radius=0.005,
    )

    # Color schemes for different data types
    dataset1_color = [0.1, 0.6, 0.9]  # Blue for dataset 1
    dataset2_color = [0.9, 0.5, 0.1]  # Orange for dataset 2
    registration_colors = {
        'cam0': [0.2, 0.8, 0.2],  # Green for cam0
        'cam1': [0.8, 0.2, 0.8],  # Purple for cam1
    }

    # Create visualization for Dataset 1
    print(f"\nCreating visualization for Dataset 1...")
    if rgb_files_dir1:
        # Load and display Dataset 1 frames
        for i, rgb_file in enumerate(tqdm(rgb_files1, desc="Loading Dataset 1")):
            frame = load_frame_data(
                rgb_files_dir1,
                rgb_file,
                calibration_data,
                fixed_pose,
                verbose,
                apply_mask_to_image=apply_mask_to_image
            )

            if frame is None:
                # Add empty node to maintain index consistency
                frame_node = server.scene.add_frame(f"/dataset1/t{i}", show_axes=False)
                frame_nodes1.append(frame_node)
                pointcloud_nodes1.append(None)
                continue

            # Get point cloud
            positions, colors, positions_bg, colors_bg = frame.get_point_cloud_with_mask(
                downsample_factor=downsample_factor,
                mask_threshold=mask_threshold,
                filter_by_mask=filter_by_mask
            )

            # Add frame node
            frame_node = server.scene.add_frame(f"/dataset1/t{i}", show_axes=False)
            frame_nodes1.append(frame_node)

            # Add point cloud
            if len(positions) > 0:
                pc_node = server.scene.add_point_cloud(
                    name=f"/dataset1/t{i}/point_cloud",
                    points=positions,
                    colors=colors,
                    point_size=point_size,
                    point_shape="rounded",
                    visible=gui_show_dataset1.value,
                )
                pointcloud_nodes1.append(pc_node)
            else:
                pointcloud_nodes1.append(None)

            # Calculate image dimensions and FOV for camera frustum
            if frame.rgb is not None:
                image_height, image_width = frame.rgb.shape[:2]
            elif calibration_data:
                image_width = calibration_data.get('image_width', 1920)
                image_height = calibration_data.get('image_height', 1080)
            else:
                image_width = 1920
                image_height = 1080

            fx = frame.K[0, 0]
            fy = frame.K[1, 1] if frame.K.shape[0] > 1 else fx
            fov_x = 2 * onp.arctan2(image_width / 2, fx)
            fov_y = 2 * onp.arctan2(image_height / 2, fy)
            fov = onp.max([fov_x, fov_y])
            aspect = image_width / image_height

            # Add camera frustum
            if frame.rgb is not None:
                if downsample_factor > 1:
                    image = frame.rgb[::downsample_factor, ::downsample_factor]
                else:
                    image = frame.rgb
            else:
                image = None

            # Use dataset-specific color
            frustum_color = dataset1_color

            server.scene.add_camera_frustum(
                f"/dataset1/t{i}/frustum",
                fov=fov,
                aspect=aspect,
                scale=camera_frustum_scale,
                image=image,
                wxyz=tf.SO3.from_matrix(frame.T_world_camera[:3, :3]).wxyz,
                position=frame.T_world_camera[:3, 3],
                color=frustum_color,
                thickness=cam_thickness,
                visible=gui_show_dataset1.value,
            )

    # Create visualization for Dataset 2 (if exists)
    if rgb_files_dir2 and rgb_files2:
        print(f"\nCreating visualization for Dataset 2...")

        for i, rgb_file in enumerate(tqdm(rgb_files2, desc="Loading Dataset 2")):
            frame = load_frame_data(
                rgb_files_dir2,
                rgb_file,
                calibration_data2,
                fixed_pose,
                verbose,
                apply_mask_to_image=apply_mask_to_image
            )

            if frame is None:
                frame_node = server.scene.add_frame(f"/dataset2/t{i}", show_axes=False)
                frame_nodes2.append(frame_node)
                pointcloud_nodes2.append(None)
                continue

            # Get point cloud
            positions, colors, positions_bg, colors_bg = frame.get_point_cloud_with_mask(
                downsample_factor=downsample_factor,
                mask_threshold=mask_threshold,
                filter_by_mask=filter_by_mask
            )

            # Add frame node
            frame_node = server.scene.add_frame(f"/dataset2/t{i}", show_axes=False)
            frame_nodes2.append(frame_node)

            # Add point cloud
            if len(positions) > 0:
                pc_node = server.scene.add_point_cloud(
                    name=f"/dataset2/t{i}/point_cloud",
                    points=positions,
                    colors=colors,
                    point_size=point_size,
                    point_shape="rounded",
                    visible=gui_show_dataset2.value if gui_show_dataset2 else False,
                )
                pointcloud_nodes2.append(pc_node)
            else:
                pointcloud_nodes2.append(None)

            # Calculate image dimensions and FOV
            if frame.rgb is not None:
                image_height, image_width = frame.rgb.shape[:2]
            elif calibration_data2:
                image_width = calibration_data2.get('image_width', 1920)
                image_height = calibration_data2.get('image_height', 1080)
            else:
                image_width = 1920
                image_height = 1080

            fx = frame.K[0, 0]
            fy = frame.K[1, 1] if frame.K.shape[0] > 1 else fx
            fov_x = 2 * onp.arctan2(image_width / 2, fx)
            fov_y = 2 * onp.arctan2(image_height / 2, fy)
            fov = onp.max([fov_x, fov_y])
            aspect = image_width / image_height

            # Add camera frustum
            if frame.rgb is not None:
                if downsample_factor > 1:
                    image = frame.rgb[::downsample_factor, ::downsample_factor]
                else:
                    image = frame.rgb
            else:
                image = None

            # Use dataset-specific color
            frustum_color = dataset2_color

            server.scene.add_camera_frustum(
                f"/dataset2/t{i}/frustum",
                fov=fov,
                aspect=aspect,
                scale=camera_frustum_scale,
                image=image,
                wxyz=tf.SO3.from_matrix(frame.T_world_camera[:3, :3]).wxyz,
                position=frame.T_world_camera[:3, 3],
                color=frustum_color,
                thickness=cam_thickness,
                visible=gui_show_dataset2.value if gui_show_dataset2 else False,
            )

    # Create visualization for Registration Results
    if show_registration and registration_viewer:
        print(f"\nCreating visualization for Registration Results...")

        for view_name, view_data in registration_viewer.view_data.items():
            print(f"  Adding view: {view_name}")

            # Get color for this view
            view_color = registration_colors.get(view_name, [0.5, 0.5, 0.5])

            # Create nodes for each frame
            registration_nodes[view_name] = []

            for i, frame_data in enumerate(view_data['frames']):
                points = frame_data['points']
                colors = frame_data['colors']

                if len(points) > 0:
                    # If no colors, use view color
                    if colors is None or len(colors) != len(points):
                        colors = np.tile(view_color, (len(points), 1))

                    # Create point cloud node
                    pc_node = server.scene.add_point_cloud(
                        name=f"/registration/{view_name}/frame_{i}",
                        points=points,
                        colors=colors,
                        point_size=point_size * 1.2,  # Slightly larger for registration
                        point_shape="rounded",
                        visible=gui_show_registration.value if 'gui_show_registration' in locals() else False,
                    )

                    registration_nodes[view_name].append(pc_node)

    # Event handler functions
    def update_frame_visibility():
        """Update frame visibility based on current settings."""
        current_timestep = gui_timestep.value

        if gui_show_all_frames.value:
            # Show all frames
            for i, frame_node in enumerate(frame_nodes1):
                if i < num_frames1:
                    visible = gui_show_dataset1.value
                    frame_node.visible = visible
                    if i < len(pointcloud_nodes1) and pointcloud_nodes1[i]:
                        pointcloud_nodes1[i].visible = visible

            for i, frame_node in enumerate(frame_nodes2):
                if i < num_frames2:
                    visible = gui_show_dataset2.value if gui_show_dataset2 else False
                    frame_node.visible = visible
                    if i < len(pointcloud_nodes2) and pointcloud_nodes2[i]:
                        pointcloud_nodes2[i].visible = visible
        else:
            # Show only current frame
            for i, frame_node in enumerate(frame_nodes1):
                if i < num_frames1:
                    visible = (i == current_timestep) and gui_show_dataset1.value
                    frame_node.visible = visible
                    if i < len(pointcloud_nodes1) and pointcloud_nodes1[i]:
                        pointcloud_nodes1[i].visible = visible

            for i, frame_node in enumerate(frame_nodes2):
                if i < num_frames2:
                    visible = (i == current_timestep) and (gui_show_dataset2.value if gui_show_dataset2 else False)
                    frame_node.visible = visible
                    if i < len(pointcloud_nodes2) and pointcloud_nodes2[i]:
                        pointcloud_nodes2[i].visible = visible

    # Setup event handlers
    @gui_timestep.on_update
    def _(_):
        update_frame_visibility()

    @gui_show_all_frames.on_update
    def _(_):
        gui_playing.disabled = gui_show_all_frames.value
        gui_timestep.disabled = gui_show_all_frames.value or gui_playing.value
        gui_next_frame.disabled = gui_show_all_frames.value or gui_playing.value
        gui_prev_frame.disabled = gui_show_all_frames.value or gui_playing.value
        update_frame_visibility()

    @gui_show_dataset1.on_update
    def _(_):
        update_frame_visibility()

    if gui_show_dataset2:
        @gui_show_dataset2.on_update
        def _(_):
            update_frame_visibility()

    if 'gui_show_registration' in locals():
        @gui_show_registration.on_update
        def _(_):
            # Update registration visibility
            for view_name, nodes in registration_nodes.items():
                for node in nodes:
                    node.visible = gui_show_registration.value

    @gui_next_frame.on_click
    def _(_):
        gui_timestep.value = (gui_timestep.value + 1) % total_frames

    @gui_prev_frame.on_click
    def _(_):
        gui_timestep.value = (gui_timestep.value - 1) % total_frames

    @gui_playing.on_update
    def _(_):
        gui_timestep.disabled = gui_playing.value or gui_show_all_frames.value
        gui_next_frame.disabled = gui_playing.value or gui_show_all_frames.value
        gui_prev_frame.disabled = gui_playing.value or gui_show_all_frames.value

    @gui_point_size.on_update
    def _(_):
        # Update point sizes for all point clouds
        new_size = gui_point_size.value

        # Update dataset 1
        for node in pointcloud_nodes1:
            if node:
                node.point_size = new_size

        # Update dataset 2
        for node in pointcloud_nodes2:
            if node:
                node.point_size = new_size

        # Update registration
        for view_name, nodes in registration_nodes.items():
            for node in nodes:
                node.point_size = new_size * 1.2  # Keep registration points slightly larger

    @gui_camera_frustum_scale.on_update
    def _(_):
        # Note: Camera frustum scale updates require recreating frustums
        # For simplicity, we'll just print a message
        print(f"Camera frustum scale changed to: {gui_camera_frustum_scale.value}")

    # Set initial visibility
    update_frame_visibility()

    # Main visualization loop
    print(f"\n{'=' * 70}")
    print(f"VISUALIZATION READY")
    print(f"Open browser to: http://localhost:8080")
    print(f"{'=' * 70}")

    print("\nCONTROLS:")
    print("1. Data Visibility: Toggle between datasets and registration results")
    print("2. Playback: Control animation and frame navigation")
    print("3. Visualization Settings: Adjust point size and other display parameters")

    if show_registration:
        print("\nREGISTRATION INFO:")
        print("- Two views (cam0 and cam1) are shown aligned to cam0 coordinate system")
        print("- Green points: cam0 (reference view)")
        print("- Purple points: cam1 (aligned to cam0)")

    print("\nPress Ctrl+C to exit...")

    try:
        while True:
            if gui_playing.value and not gui_show_all_frames.value:
                gui_timestep.value = (gui_timestep.value + 1) % total_frames
            time.sleep(1.0 / gui_framerate.value)
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    except Exception as e:
        print(f"\nError in visualization loop: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Integrated visualization system for comparing datasets and registration results."
    )

    # Basic parameters
    parser.add_argument("--data-path", type=Path, required=True,
                        help="Path to the first dataset directory")
    parser.add_argument("--data-path2", type=Path, default=None,
                        help="Path to the second dataset directory (optional)")

    # Calibration parameters
    parser.add_argument("--calib-xml-path", type=Path, default=None,
                        help="Path to calibration XML file for first dataset")
    parser.add_argument("--calib-xml-path2", type=Path, default=None,
                        help="Path to calibration XML file for second dataset")

    # Visualization parameters
    parser.add_argument("--downsample-factor", type=int, default=4,
                        help="Downsample factor for point cloud")
    parser.add_argument("--max-frames", type=int, default=100,
                        help="Maximum number of frames to load")
    parser.add_argument("--point-size", type=float, default=0.001,
                        help="Point size")
    parser.add_argument("--camera-frustum-scale", type=float, default=0.015,
                        help="Camera frustum scale")
    parser.add_argument("--cam-thickness", type=float, default=1.5,
                        help="Camera frustum thickness")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride for loading frames")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose information")

    # Registration parameters
    parser.add_argument("--show-registration", action="store_true",
                        help="Show registration results (two views: cam0 and cam1)")
    parser.add_argument("--registration-dir", type=Path, default=None,
                        help="Directory containing registration results")
    parser.add_argument("--registration-views", type=str, nargs='+',
                        default=['cam0', 'cam1'],
                        help="Views to show for registration (default: cam0 cam1)")
    parser.add_argument("--max-registration-frames", type=int, default=30,
                        help="Maximum frames per view for registration")

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        data_path2=args.data_path2,
        calib_xml_path=args.calib_xml_path,
        calib_xml_path2=args.calib_xml_path2,
        downsample_factor=args.downsample_factor,
        max_frames=args.max_frames,
        share=False,
        point_size=args.point_size,
        camera_frustum_scale=args.camera_frustum_scale,
        axes_scale=0.1,
        cam_thickness=args.cam_thickness,
        fixed_pose=True,
        mask_threshold=0.5,
        show_background=False,
        verbose=args.verbose,
        stride=args.stride,
        apply_mask_to_image=True,
        filter_by_mask=True,
        show_registration=args.show_registration,
        registration_dir=args.registration_dir,
        registration_views=args.registration_views,
        max_registration_frames=args.max_registration_frames,
    )