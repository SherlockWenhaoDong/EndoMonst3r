"""
Simplified Multi-view Point Cloud Registration and Visualization
Only shows point clouds aligned to cam0 coordinate system
"""

import time
import numpy as np
import open3d as o3d
import glob
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import argparse
import tqdm
import cv2
import warnings
import viser
import viser.transforms as tf
import matplotlib.cm as cm
import os
import re

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class MultiViewRegistration:
    """Simple multi-view registration aligning all views to cam0."""

    def __init__(self, base_dir: Path, verbose: bool = True):
        self.base_dir = Path(base_dir)
        self.verbose = verbose
        self.views = {}
        self.transformations = {}
        self.masks = {}  # Store masks for each view
        self.camera_poses = {}  # Store camera poses for each view

    def setup_views(self, view_names: List[str]):
        """Set up view configurations."""
        self.view_names = view_names
        for view_name in view_names:
            ply_dir = self.base_dir / f'frames_{view_name}_ply_files'
            if self.verbose:
                print(f"\nView: {view_name}")
                print(f"  Point cloud directory exists: {ply_dir.exists()}")

    def load_view_data(self, view_name: str, max_frames: int = 30):
        """Load point cloud data for a view."""
        print(f"\nLoading data for {view_name}...")

        ply_dir = self.base_dir / f'frames_{view_name}_pc/ply_files'

        if not ply_dir.exists():
            raise FileNotFoundError(f"Point cloud directory not found: {ply_dir}")

        ply_files = sorted(glob.glob(str(ply_dir / '*.ply')))
        if not ply_files:
            print(f"Warning: No PLY files found in {ply_dir}")
            return None

        ply_files = ply_files[:max_frames]

        view_data = {
            'name': view_name,
            'frames': [],
            'ply_dir': ply_dir
        }

        # Load each frame
        for ply_file in tqdm.tqdm(ply_files, desc=f"Loading {view_name}", disable=not self.verbose):
            try:
                pcd = o3d.io.read_point_cloud(ply_file)
                if len(pcd.points) == 0:
                    continue

                frame_idx = self.extract_frame_number(Path(ply_file).stem)

                # Store point cloud data
                frame_data = {
                    'points': np.asarray(pcd.points),
                    'colors': np.asarray(pcd.colors) if pcd.has_colors() else None,
                    'frame_idx': frame_idx,
                    'path': ply_file
                }

                view_data['frames'].append(frame_data)

            except Exception as e:
                if self.verbose:
                    print(f"Error processing {ply_file}: {e}")

        self.views[view_name] = view_data

        if self.verbose:
            print(f"  Loaded {len(view_data['frames'])} frames")
            total_points = sum(len(f['points']) for f in view_data['frames'])
            print(f"  Total points: {total_points:,}")

        return view_data

    def load_camera_poses(self, pose_dir: Path, view_name: str):
        """Load camera poses from directory."""
        print(f"\nLoading camera poses for {view_name}...")

        if not pose_dir.exists():
            print(f"Warning: Camera pose directory not found: {pose_dir}")
            return None

        # Look for pose files (txt, json, npy, etc.)
        pose_extensions = ['*.txt', '*.json', '*.npy', '*.npz']
        pose_files = []
        for ext in pose_extensions:
            pose_files.extend(glob.glob(str(pose_dir / ext)))

        if not pose_files:
            print(f"Warning: No pose files found in {pose_dir}")
            return None

        pose_files = sorted(pose_files)
        poses = []

        for pose_file in pose_files:
            try:
                # Try to load pose based on file extension
                pose_path = Path(pose_file)
                if pose_path.suffix == '.txt':
                    # Load 4x4 matrix from text file
                    pose_matrix = np.loadtxt(pose_file)
                    if pose_matrix.shape == (4, 4):
                        poses.append(pose_matrix)
                elif pose_path.suffix == '.json':
                    # Load from JSON
                    with open(pose_file, 'r') as f:
                        pose_data = json.load(f)
                        if isinstance(pose_data, list) and len(pose_data) == 16:
                            pose_matrix = np.array(pose_data).reshape(4, 4)
                            poses.append(pose_matrix)
                elif pose_path.suffix == '.npy':
                    # Load from numpy binary
                    pose_matrix = np.load(pose_file)
                    if pose_matrix.shape == (4, 4):
                        poses.append(pose_matrix)
                elif pose_path.suffix == '.npz':
                    # Load from numpy zip
                    data = np.load(pose_file)
                    if 'pose' in data:
                        pose_matrix = data['pose']
                        if pose_matrix.shape == (4, 4):
                            poses.append(pose_matrix)

            except Exception as e:
                if self.verbose:
                    print(f"Error processing pose file {pose_file}: {e}")

        if poses:
            self.camera_poses[view_name] = poses
            print(f"  Loaded {len(poses)} camera poses for {view_name}")
        else:
            print(f"  No valid camera poses found for {view_name}")

        return poses

    def load_masks(self, mask_dir: Path, view_name: str, max_frames: int = 30):
        """Load mask images for a view."""
        print(f"\nLoading masks for {view_name}...")

        if not mask_dir.exists():
            print(f"Warning: Mask directory not found: {mask_dir}")
            return None

        # Look for mask files (png, jpg, etc.)
        mask_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        mask_files = []
        for ext in mask_extensions:
            mask_files.extend(glob.glob(str(mask_dir / ext)))

        if not mask_files:
            print(f"Warning: No mask files found in {mask_dir}")
            return None

        mask_files = sorted(mask_files)[:max_frames]
        masks = []

        for mask_file in tqdm.tqdm(mask_files, desc=f"Loading masks for {view_name}", disable=not self.verbose):
            try:
                # Load mask image
                mask_img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                if mask_img is None:
                    continue

                # Binarize mask (0 for background, 1 for foreground)
                mask_binary = (mask_img > 0).astype(np.float32)

                # Extract frame number from filename
                frame_idx = self.extract_frame_number(Path(mask_file).stem)

                masks.append({
                    'frame_idx': frame_idx,
                    'mask': mask_binary,
                    'path': mask_file
                })

            except Exception as e:
                if self.verbose:
                    print(f"Error processing mask {mask_file}: {e}")

        self.masks[view_name] = masks

        if self.verbose:
            print(f"  Loaded {len(masks)} masks")

        return masks

    def extract_frame_number(self, filename: str) -> int:
        """Extract frame number from filename."""
        patterns = [r'frame_(\d+)', r'(\d+)', r'pc_(\d+)', r'points_(\d+)']

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue

        numbers = re.findall(r'\d+', filename)
        if numbers:
            try:
                return int(numbers[-1])
            except ValueError:
                pass

        return 0

    def apply_mask_to_pointcloud(self, points: np.ndarray, colors: Optional[np.ndarray],
                                 mask: np.ndarray, mask_threshold: float = 0.5):
        """Apply mask to filter point cloud."""
        if len(points) == 0:
            return points, colors

        # For simplicity, we'll randomly sample points based on mask density
        # In a real implementation, you would project 3D points to 2D and check mask
        mask_density = np.mean(mask)

        # Randomly sample points based on mask density
        n_points = len(points)
        keep_indices = np.random.rand(n_points) < mask_density

        filtered_points = points[keep_indices]
        if colors is not None:
            filtered_colors = colors[keep_indices]
        else:
            filtered_colors = None

        return filtered_points, filtered_colors

    def simple_center_alignment(self, reference_view: str = 'cam0'):
        """
        Simple registration by aligning centers of point clouds.
        All views will be aligned to cam0 coordinate system.
        """
        if reference_view not in self.views:
            raise ValueError(f"Reference view {reference_view} not loaded")

        print(f"\n{'=' * 60}")
        print(f"ALIGNING ALL VIEWS TO {reference_view} COORDINATE SYSTEM")
        print(f"{'=' * 60}")

        # Initialize transformations
        self.transformations = {reference_view: np.eye(4)}  # cam0 stays at origin

        # For reference view (cam0), compute average point cloud center
        ref_data = self.views[reference_view]
        ref_center = np.zeros(3)
        ref_count = 0

        for frame in ref_data['frames']:
            if len(frame['points']) > 0:
                ref_center += np.mean(frame['points'], axis=0)
                ref_count += 1

        if ref_count > 0:
            ref_center /= ref_count

        print(f"Reference view ({reference_view}) center: {ref_center}")

        # Align each view to reference view
        for view_name in self.view_names:
            if view_name == reference_view:
                continue

            print(f"\nAligning {view_name} to {reference_view}...")

            if view_name not in self.views:
                print(f"  View {view_name} not loaded")
                self.transformations[view_name] = np.eye(4)
                continue

            view_data = self.views[view_name]

            # Compute average center of this view
            view_center = np.zeros(3)
            view_count = 0

            for frame in view_data['frames']:
                if len(frame['points']) > 0:
                    view_center += np.mean(frame['points'], axis=0)
                    view_count += 1

            if view_count > 0:
                view_center /= view_count

            print(f"  View {view_name} center: {view_center}")

            # Compute translation to align to reference center
            translation = ref_center - view_center

            # Create transformation matrix (translation only)
            transform = np.eye(4)
            transform[:3, 3] = translation

            self.transformations[view_name] = transform

            print(f"  ✓ Aligned {view_name} (translation: {translation})")

        # Save results
        self.save_results()

        return self.transformations

    def save_results(self):
        """Save alignment results to disk."""
        output_dir = self.base_dir / 'registration_results'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save transformations
        transforms_dict = {}
        for view_name, transform in self.transformations.items():
            transforms_dict[view_name] = transform.tolist()

        with open(output_dir / 'transformations.json', 'w') as f:
            json.dump(transforms_dict, f, indent=2)

        print(f"\n✓ Transformations saved to: {output_dir / 'transformations.json'}")

        # Save aligned point clouds
        aligned_dir = output_dir / 'aligned_pointclouds'
        aligned_dir.mkdir(parents=True, exist_ok=True)

        for view_name, view_data in self.views.items():
            if view_name not in self.transformations:
                continue

            transform = self.transformations[view_name]
            view_aligned_dir = aligned_dir / view_name
            view_aligned_dir.mkdir(parents=True, exist_ok=True)

            for frame in view_data['frames']:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(frame['points'])
                if frame['colors'] is not None:
                    pcd.colors = o3d.utility.Vector3dVector(frame['colors'])

                # Apply transformation to align to cam0 coordinate system
                pcd.transform(transform)

                output_file = view_aligned_dir / f'frame_{frame["frame_idx"]:06d}_aligned.ply'
                o3d.io.write_point_cloud(str(output_file), pcd)

        print(f"✓ Aligned point clouds saved to: {aligned_dir}")

        self.output_dir = output_dir
        return output_dir


class MultiCameraVisualizer:
    """Interactive 3D visualization showing two views with their camera poses and aligned results."""

    def __init__(self, registration_results_dir: Path, verbose: bool = True):
        self.results_dir = Path(registration_results_dir)
        self.verbose = verbose

        # Visualization parameters
        self.params = {
            'point_size': 0.002,
            'downsample_factor': 4,
            'max_frames_display': 30,
            'show_all_frames': False,
            'current_frame': 0,
            'playing': False,
            'fps': 30,
            'apply_mask': False,
            'mask_threshold': 0.5,
        }

        # Load registration results
        self.transformations = self.load_transformations()
        self.view_data = self.load_aligned_pointclouds()

        # Initialize Viser server
        self.server = None
        self.camera_poses = {}  # Store camera poses for each view

    def load_transformations(self) -> Dict[str, np.ndarray]:
        """Load transformation matrices from JSON file."""
        transform_file = self.results_dir / 'transformations.json'
        if not transform_file.exists():
            raise FileNotFoundError(f"Transformations file not found: {transform_file}")

        with open(transform_file, 'r') as f:
            transforms_dict = json.load(f)

        transformations = {}
        for view_name, transform_list in transforms_dict.items():
            transformations[view_name] = np.array(transform_list)

        if self.verbose:
            print(f"Loaded transformations for {len(transformations)} views")

        return transformations

    def load_aligned_pointclouds(self) -> Dict[str, Dict]:
        """Load aligned point clouds from disk."""
        aligned_dir = self.results_dir / 'aligned_pointclouds'
        if not aligned_dir.exists():
            raise FileNotFoundError(f"Aligned point clouds directory not found: {aligned_dir}")

        view_data = {}

        # Find all view directories
        view_dirs = [d for d in aligned_dir.iterdir() if d.is_dir()]

        for view_dir in view_dirs:
            view_name = view_dir.name
            view_data[view_name] = {
                'name': view_name,
                'frames': []
            }

            # Find aligned PLY files
            ply_files = sorted(glob.glob(str(view_dir / '*_aligned.ply')))

            for ply_file in ply_files[:self.params['max_frames_display']]:
                try:
                    pcd = o3d.io.read_point_cloud(ply_file)
                    if len(pcd.points) > 0:
                        frame_idx = self.extract_frame_number(Path(ply_file).stem)
                        view_data[view_name]['frames'].append({
                            'frame_idx': frame_idx,
                            'pcd': pcd,
                            'path': ply_file
                        })
                except Exception as e:
                    if self.verbose:
                        print(f"Error loading {ply_file}: {e}")

            if self.verbose:
                print(f"  {view_name}: {len(view_data[view_name]['frames'])} frames (aligned to cam0)")

        return view_data

    def load_camera_poses(self, pose_dir1: Path, pose_dir2: Path, view1_name: str = 'cam0', view2_name: str = 'cam1'):
        """Load camera poses for both views."""
        print(f"\nLoading camera poses...")

        # Load poses for view1
        if pose_dir1.exists():
            self.camera_poses[view1_name] = self._load_poses_from_dir(pose_dir1, view1_name)
        else:
            print(f"Warning: Camera pose directory for {view1_name} not found: {pose_dir1}")
            self.camera_poses[view1_name] = [np.eye(4)] * self.params['max_frames_display']

        # Load poses for view2
        if pose_dir2.exists():
            self.camera_poses[view2_name] = self._load_poses_from_dir(pose_dir2, view2_name)
        else:
            print(f"Warning: Camera pose directory for {view2_name} not found: {pose_dir2}")
            self.camera_poses[view2_name] = [np.eye(4)] * self.params['max_frames_display']

        print(f"  Loaded {len(self.camera_poses.get(view1_name, []))} poses for {view1_name}")
        print(f"  Loaded {len(self.camera_poses.get(view2_name, []))} poses for {view2_name}")

    def _load_poses_from_dir(self, pose_dir: Path, view_name: str):
        """Helper function to load poses from directory."""
        # Look for pose files (txt, json, npy, etc.)
        pose_extensions = ['*.txt', '*.json', '*.npy', '*.npz']
        pose_files = []
        for ext in pose_extensions:
            pose_files.extend(glob.glob(str(pose_dir / ext)))

        if not pose_files:
            print(f"Warning: No pose files found in {pose_dir}")
            return [np.eye(4)] * self.params['max_frames_display']

        pose_files = sorted(pose_files)
        poses = []

        for pose_file in pose_files[:self.params['max_frames_display']]:
            try:
                pose_matrix = self._load_single_pose(pose_file)
                if pose_matrix is not None:
                    poses.append(pose_matrix)
                else:
                    poses.append(np.eye(4))
            except Exception as e:
                if self.verbose:
                    print(f"Error processing pose file {pose_file}: {e}")
                poses.append(np.eye(4))

        return poses

    def _load_single_pose(self, pose_file: str):
        """Load a single pose matrix from file."""
        pose_path = Path(pose_file)

        try:
            if pose_path.suffix == '.txt':
                # Load 4x4 matrix from text file
                pose_matrix = np.loadtxt(pose_file)
                if pose_matrix.shape == (4, 4):
                    return pose_matrix
            elif pose_path.suffix == '.json':
                # Load from JSON
                with open(pose_file, 'r') as f:
                    pose_data = json.load(f)
                    if isinstance(pose_data, list) and len(pose_data) == 16:
                        return np.array(pose_data).reshape(4, 4)
            elif pose_path.suffix == '.npy':
                # Load from numpy binary
                pose_matrix = np.load(pose_file)
                if pose_matrix.shape == (4, 4):
                    return pose_matrix
            elif pose_path.suffix == '.npz':
                # Load from numpy zip
                data = np.load(pose_file)
                if 'pose' in data:
                    pose_matrix = data['pose']
                    if pose_matrix.shape == (4, 4):
                        return pose_matrix
        except:
            pass

        return None

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

    def create_visualization(self, share: bool = False,
                             pose_dir1: Path = None,
                             pose_dir2: Path = None,
                             view1_name: str = 'cam0',
                             view2_name: str = 'cam1'):
        """Create and launch the interactive visualization."""
        self.server = viser.ViserServer()

        if share:
            self.server.request_share_url()

        self.server.scene.set_up_direction('-z')

        print(f"\n{'=' * 60}")
        print(f"MULTI-CAMERA VISUALIZATION WITH CAMERA POSES")
        print(f"{'=' * 60}")

        # Load camera poses if provided
        if pose_dir1 and pose_dir2:
            self.load_camera_poses(pose_dir1, pose_dir2, view1_name, view2_name)
        else:
            print("Warning: No camera pose directories provided, using identity poses")

        # Get all frame indices
        all_frame_indices = set()
        for view_data in self.view_data.values():
            for frame in view_data['frames']:
                all_frame_indices.add(frame['frame_idx'])

        self.frame_indices = sorted(all_frame_indices)
        self.total_frames = len(self.frame_indices)

        print(f"Views: {list(self.view_data.keys())}")
        print(f"Total unique frames: {self.total_frames}")
        print(f"Transformations available: {list(self.transformations.keys())}")

        # Create UI
        self.create_ui(view1_name, view2_name)

        # Create visualization nodes
        self.create_multi_camera_nodes(view1_name, view2_name)

        # Start visualization
        print("\nVisualization ready. Open the browser to view point clouds.")
        print(f"Server URL: http://localhost:8080")

        self.run_visualization_loop()

    def create_ui(self, view1_name: str, view2_name: str):
        """Create user interface."""
        # Playback controls
        with self.server.gui.add_folder("Playback"):
            self.gui_timestep = self.server.gui.add_slider(
                "Timestep",
                min=0,
                max=self.total_frames - 1,
                step=1,
                initial_value=0,
                disabled=True,
            )

            self.gui_next_frame = self.server.gui.add_button("Next Frame", disabled=True)
            self.gui_prev_frame = self.server.gui.add_button("Prev Frame", disabled=True)
            self.gui_playing = self.server.gui.add_checkbox("Playing", False)
            self.gui_framerate = self.server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=30)
            self.gui_show_all_frames = self.server.gui.add_checkbox("Show all frames", False)
            self.gui_stride = self.server.gui.add_slider("Stride", min=1, max=self.total_frames, step=1,
                                                         initial_value=1, disabled=True)

        # Visualization settings
        with self.server.gui.add_folder("Visualization"):
            self.gui_point_size = self.server.gui.add_slider("Point Size", min=0.0001, max=0.01, step=0.0001,
                                                             initial_value=self.params['point_size'])
            self.gui_downsample = self.server.gui.add_slider("Downsample", min=1, max=16, step=1,
                                                             initial_value=self.params['downsample_factor'])

        # View visibility - Show all camera poses
        with self.server.gui.add_folder("Camera Poses"):
            self.gui_show_cam0_pose = self.server.gui.add_checkbox(f"Show {view1_name} camera pose (red)", True)
            self.gui_show_cam1_pose = self.server.gui.add_checkbox(f"Show {view2_name} original camera pose (green)",
                                                                   True)
            self.gui_show_cam1_registered_pose = self.server.gui.add_checkbox(
                f"Show {view2_name} registered camera pose (blue)", True)

        # Point cloud visibility
        with self.server.gui.add_folder("Point Clouds"):
            self.gui_show_cam0_points = self.server.gui.add_checkbox(f"Show {view1_name} point cloud", True)
            self.gui_show_cam1_points = self.server.gui.add_checkbox(f"Show {view2_name} point cloud", True)
            self.gui_show_aligned_points = self.server.gui.add_checkbox(f"Show {view2_name} aligned point cloud", True)

        # Camera frustum settings
        with self.server.gui.add_folder("Camera Frustums"):
            self.gui_show_frustums = self.server.gui.add_checkbox("Show Camera Frustums", True)
            self.gui_frustum_scale = self.server.gui.add_slider("Frustum Scale", min=0.001, max=0.1, step=0.001,
                                                                initial_value=0.02)

        # Setup event handlers
        self.setup_event_handlers(view1_name, view2_name)

    def setup_event_handlers(self, view1_name: str, view2_name: str):
        """Setup event handlers."""

        @self.gui_next_frame.on_click
        def _(_):
            self.params['current_frame'] = (self.params['current_frame'] + 1) % self.total_frames
            self.gui_timestep.value = self.params['current_frame']
            self.update_frame_visibility()

        @self.gui_prev_frame.on_click
        def _(_):
            self.params['current_frame'] = (self.params['current_frame'] - 1) % self.total_frames
            self.gui_timestep.value = self.params['current_frame']
            self.update_frame_visibility()

        @self.gui_playing.on_update
        def _(_):
            self.params['playing'] = self.gui_playing.value
            self.gui_timestep.disabled = self.params['playing'] or self.params['show_all_frames']
            self.gui_next_frame.disabled = self.params['playing'] or self.params['show_all_frames']
            self.gui_prev_frame.disabled = self.params['playing'] or self.params['show_all_frames']

        @self.gui_timestep.on_update
        def _(_):
            self.params['current_frame'] = self.gui_timestep.value
            if not self.params['show_all_frames']:
                self.update_frame_visibility()

        @self.gui_show_all_frames.on_update
        def _(_):
            self.params['show_all_frames'] = self.gui_show_all_frames.value
            self.gui_stride.disabled = not self.params['show_all_frames']

            if self.params['show_all_frames']:
                stride = self.gui_stride.value
                self.show_all_frames_with_stride(stride)
                self.gui_playing.disabled = True
                self.gui_timestep.disabled = True
                self.gui_next_frame.disabled = True
                self.gui_prev_frame.disabled = True
            else:
                self.update_frame_visibility()
                self.gui_playing.disabled = False
                self.gui_timestep.disabled = self.params['playing']
                self.gui_next_frame.disabled = self.params['playing']
                self.gui_prev_frame.disabled = self.params['playing']

        @self.gui_stride.on_update
        def _(_):
            if self.params['show_all_frames']:
                stride = self.gui_stride.value
                self.show_all_frames_with_stride(stride)

        @self.gui_point_size.on_update
        def _(_):
            self.params['point_size'] = self.gui_point_size.value
            self.update_point_sizes()

        @self.gui_downsample.on_update
        def _(_):
            self.params['downsample_factor'] = int(self.gui_downsample.value)
            self.reload_point_clouds()

        @self.gui_show_cam0_pose.on_update
        def _(_):
            self.update_pose_visibility(view1_name, 'cam0_pose')

        @self.gui_show_cam1_pose.on_update
        def _(_):
            self.update_pose_visibility(view2_name, 'cam1_original_pose')

        @self.gui_show_cam1_registered_pose.on_update
        def _(_):
            self.update_pose_visibility(view2_name, 'cam1_registered_pose')

        @self.gui_show_cam0_points.on_update
        def _(_):
            self.update_pointcloud_visibility(view1_name, 'cam0_points')

        @self.gui_show_cam1_points.on_update
        def _(_):
            self.update_pointcloud_visibility(view2_name, 'cam1_points')

        @self.gui_show_aligned_points.on_update
        def _(_):
            self.update_pointcloud_visibility(view2_name, 'aligned_points')

        @self.gui_show_frustums.on_update
        def _(_):
            self.update_frustum_visibility()

        @self.gui_frustum_scale.on_update
        def _(_):
            self.update_frustum_scales()

    def create_multi_camera_nodes(self, view1_name: str, view2_name: str):
        """Create point cloud nodes with camera poses for both views."""
        # Create base frame at origin
        self.server.scene.add_frame(
            "/origin",
            wxyz=tf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0])).wxyz,
            position=(0, 0, 0),
            axes_length=0.1,
            axes_radius=0.01,
        )

        # Color map for different views
        colors = cm.tab10(np.linspace(0, 1, 5))  # 5 colors for different elements
        self.view_colors = {
            'cam0_pose': colors[0][:3],  # Red for cam0 camera pose
            'cam1_original_pose': colors[1][:3],  # Green for cam1 original camera pose
            'cam1_registered_pose': colors[2][:3],  # Blue for cam1 registered camera pose
            'cam0_points': colors[3][:3],  # Purple for cam0 point cloud
            'cam1_points': colors[4][:3],  # Orange for cam1 point cloud
            'aligned_points': colors[2][:3]  # Blue for aligned point cloud (same as registered pose)
        }

        # Store nodes
        self.pose_nodes = {}
        self.pointcloud_nodes = {}
        self.frustum_nodes = {}

        # Check if we have camera poses
        has_poses = (view1_name in self.camera_poses and view2_name in self.camera_poses)

        if not has_poses:
            print("Warning: Using identity poses for both cameras")
            self.camera_poses[view1_name] = [np.eye(4)] * self.params['max_frames_display']
            self.camera_poses[view2_name] = [np.eye(4)] * self.params['max_frames_display']

        # 修正：获取配准变换矩阵
        registration_transform = self.transformations.get(view2_name, np.eye(4))
        print(f"\nRegistration transformation for {view2_name}:")
        print(registration_transform)

        # 定义要创建的所有节点类型
        node_types = [
            ('cam0_pose', view1_name, None),  # cam0 pose (no additional transform)
            ('cam1_original_pose', view2_name, None),  # cam1 original pose
            ('cam1_registered_pose', view2_name, registration_transform),  # cam1 registered pose
            ('cam0_points', view1_name, None),  # cam0 point cloud
            ('cam1_points', view2_name, None),  # cam1 point cloud (original)
            ('aligned_points', view2_name, registration_transform)  # cam1 aligned point cloud
        ]

        for node_type, view_name, additional_transform in node_types:
            self.pose_nodes[node_type] = []
            self.pointcloud_nodes[node_type] = []
            self.frustum_nodes[node_type] = []

            color = self.view_colors[node_type]

            if view_name not in self.view_data:
                print(f"Warning: View {view_name} not found in data for node type {node_type}")
                continue

            view_data = self.view_data[view_name]

            # Get base poses for this view
            base_poses = self.camera_poses.get(view_name, [np.eye(4)])

            # Make sure we have enough poses
            n_frames = min(len(view_data['frames']), len(base_poses), self.params['max_frames_display'])

            for i in range(n_frames):
                frame_data = view_data['frames'][i]
                frame_idx = frame_data['frame_idx']

                # Get base pose for this frame
                if i < len(base_poses):
                    base_pose = base_poses[i]
                else:
                    base_pose = base_poses[0] if base_poses else np.eye(4)

                # Apply additional transform if needed
                if additional_transform is not None:
                    if 'pose' in node_type:
                        # For pose nodes: apply transform to pose
                        final_pose = additional_transform @ base_pose
                    else:
                        # For point cloud nodes: points are already aligned
                        final_pose = base_pose
                else:
                    final_pose = base_pose

                # Extract camera position and rotation
                camera_position = final_pose[:3, 3]
                camera_rotation = final_pose[:3, :3]

                # Create pose node (only for pose types)
                if 'pose' in node_type:
                    pose_node = self.server.scene.add_frame(
                        f"/{node_type}/t{frame_idx}",
                        wxyz=tf.SO3.from_matrix(camera_rotation).wxyz,
                        position=camera_position,
                        show_axes=False,
                        axes_length=0.05,
                        axes_radius=0.005,
                    )
                    self.pose_nodes[node_type].append(pose_node)

                # Create point cloud node (only for point cloud types)
                if 'points' in node_type:
                    pcd = frame_data['pcd']

                    # Downsample if needed
                    if self.params['downsample_factor'] > 1:
                        pcd = pcd.voxel_down_sample(self.params['downsample_factor'] / 1000.0)

                    # Get points and colors
                    points = np.asarray(pcd.points)
                    colors_array = np.asarray(pcd.colors) if pcd.has_colors() else None

                    if len(points) > 0:
                        # Transform points based on node type
                        if node_type == 'cam0_points':
                            # cam0 points: use original pose
                            R = base_pose[:3, :3]
                            t = base_pose[:3, 3]
                            transformed_points = (R @ points.T).T + t
                        elif node_type == 'cam1_points':
                            # cam1 points: use original pose
                            R = base_pose[:3, :3]
                            t = base_pose[:3, 3]
                            transformed_points = (R @ points.T).T + t
                        elif node_type == 'aligned_points':
                            # aligned points: points are already in cam0 coordinates
                            # No additional transformation needed
                            transformed_points = points
                        else:
                            transformed_points = points

                        # Use original colors if available, otherwise use type color
                        if colors_array is not None:
                            final_colors = colors_array
                        else:
                            final_colors = np.tile(color, (len(points), 1))

                        # Create point cloud node
                        pointcloud_node = self.server.scene.add_point_cloud(
                            name=f"/{node_type}/t{frame_idx}/points",
                            points=transformed_points,
                            colors=final_colors,
                            point_size=self.params['point_size'],
                            point_shape="rounded",
                        )
                        self.pointcloud_nodes[node_type].append(pointcloud_node)

                # Create camera frustum node (only for pose types)
                if 'pose' in node_type and self.gui_show_frustums.value:
                    frustum_node = self.server.scene.add_camera_frustum(
                        name=f"/{node_type}/t{frame_idx}/frustum",
                        fov=0.8,  # Field of view in radians
                        aspect=1.333,  # Aspect ratio (4:3)
                        scale=self.gui_frustum_scale.value,
                        wxyz=tf.SO3.from_matrix(camera_rotation).wxyz,
                        position=camera_position,
                        color=color,
                        thickness=2.0,
                    )
                    self.frustum_nodes[node_type].append(frustum_node)

        # Update initial visibility
        self.update_frame_visibility()

    def update_frame_visibility(self):
        """Update which frames are visible based on current timestep."""
        if self.total_frames == 0:
            return

        current_frame_idx = self.frame_indices[self.params['current_frame']]

        with self.server.atomic():
            # Update pose nodes visibility
            for node_type in self.pose_nodes.keys():
                # Determine if this node type should be visible
                if node_type == 'cam0_pose':
                    type_enabled = self.gui_show_cam0_pose.value
                elif node_type == 'cam1_original_pose':
                    type_enabled = self.gui_show_cam1_pose.value
                elif node_type == 'cam1_registered_pose':
                    type_enabled = self.gui_show_cam1_registered_pose.value
                else:
                    type_enabled = True

                pose_list = self.pose_nodes.get(node_type, [])
                frustum_list = self.frustum_nodes.get(node_type, [])

                for i, pose_node in enumerate(pose_list):
                    should_show = type_enabled

                    # For single frame mode, only show current frame
                    if not self.params['show_all_frames']:
                        # Try to match frame indices
                        frame_data = None
                        for view_data in self.view_data.values():
                            if i < len(view_data['frames']):
                                frame_data = view_data['frames'][i]
                                break

                        if frame_data:
                            should_show = should_show and (frame_data['frame_idx'] == current_frame_idx)

                    pose_node.visible = should_show

                    # Update frustum visibility
                    if i < len(frustum_list):
                        frustum_list[i].visible = should_show and self.gui_show_frustums.value

            # Update point cloud nodes visibility
            for node_type in self.pointcloud_nodes.keys():
                # Determine if this node type should be visible
                if node_type == 'cam0_points':
                    type_enabled = self.gui_show_cam0_points.value
                elif node_type == 'cam1_points':
                    type_enabled = self.gui_show_cam1_points.value
                elif node_type == 'aligned_points':
                    type_enabled = self.gui_show_aligned_points.value
                else:
                    type_enabled = True

                pc_list = self.pointcloud_nodes.get(node_type, [])

                for i, pc_node in enumerate(pc_list):
                    should_show = type_enabled

                    # For single frame mode, only show current frame
                    if not self.params['show_all_frames']:
                        # Try to match frame indices
                        frame_data = None
                        for view_data in self.view_data.values():
                            if i < len(view_data['frames']):
                                frame_data = view_data['frames'][i]
                                break

                        if frame_data:
                            should_show = should_show and (frame_data['frame_idx'] == current_frame_idx)

                    pc_node.visible = should_show

    def show_all_frames_with_stride(self, stride: int):
        """Show all frames with stride."""
        with self.server.atomic():
            # Update pose nodes visibility
            for node_type in self.pose_nodes.keys():
                # Determine if this node type should be visible
                if node_type == 'cam0_pose':
                    type_enabled = self.gui_show_cam0_pose.value
                elif node_type == 'cam1_original_pose':
                    type_enabled = self.gui_show_cam1_pose.value
                elif node_type == 'cam1_registered_pose':
                    type_enabled = self.gui_show_cam1_registered_pose.value
                else:
                    type_enabled = True

                pose_list = self.pose_nodes.get(node_type, [])
                frustum_list = self.frustum_nodes.get(node_type, [])

                for i, pose_node in enumerate(pose_list):
                    should_show = type_enabled and (i % stride == 0)
                    pose_node.visible = should_show

                    if i < len(frustum_list):
                        frustum_list[i].visible = should_show and self.gui_show_frustums.value

            # Update point cloud nodes visibility
            for node_type in self.pointcloud_nodes.keys():
                # Determine if this node type should be visible
                if node_type == 'cam0_points':
                    type_enabled = self.gui_show_cam0_points.value
                elif node_type == 'cam1_points':
                    type_enabled = self.gui_show_cam1_points.value
                elif node_type == 'aligned_points':
                    type_enabled = self.gui_show_aligned_points.value
                else:
                    type_enabled = True

                pc_list = self.pointcloud_nodes.get(node_type, [])

                for i, pc_node in enumerate(pc_list):
                    should_show = type_enabled and (i % stride == 0)
                    pc_node.visible = should_show

    def update_pose_visibility(self, view_name: str, pose_type: str):
        """Update visibility of a specific pose type."""
        if self.params['show_all_frames']:
            stride = self.gui_stride.value
            self.show_all_frames_with_stride(stride)
        else:
            self.update_frame_visibility()

    def update_pointcloud_visibility(self, view_name: str, pc_type: str):
        """Update visibility of a specific point cloud type."""
        if self.params['show_all_frames']:
            stride = self.gui_stride.value
            self.show_all_frames_with_stride(stride)
        else:
            self.update_frame_visibility()

    def update_point_sizes(self):
        """Update point sizes."""
        with self.server.atomic():
            for node_type in self.pointcloud_nodes.keys():
                for node in self.pointcloud_nodes[node_type]:
                    node.point_size = self.params['point_size']

    def update_frustum_visibility(self):
        """Update camera frustum visibility."""
        with self.server.atomic():
            for node_type in self.frustum_nodes.keys():
                for node in self.frustum_nodes[node_type]:
                    node.visible = self.gui_show_frustums.value

    def update_frustum_scales(self):
        """Update camera frustum scales."""
        with self.server.atomic():
            for node_type in self.frustum_nodes.keys():
                for node in self.frustum_nodes[node_type]:
                    node.scale = self.gui_frustum_scale.value

    def reload_point_clouds(self):
        """Reload point clouds with new downsample factor."""
        print("Reloading point clouds...")

        # Remove old nodes
        with self.server.atomic():
            for node_type in self.pointcloud_nodes.keys():
                for node in self.pointcloud_nodes[node_type]:
                    node.remove()
                self.pointcloud_nodes[node_type] = []

        # Get view names from existing nodes
        view1_name = 'cam0'
        view2_name = 'cam1'
        if hasattr(self, 'pose_nodes') and 'cam0_pose' in self.pose_nodes:
            # Recreate nodes with current parameters
            self.create_multi_camera_nodes(view1_name, view2_name)

        print("Point clouds reloaded")

    def run_visualization_loop(self):
        """Run visualization loop."""
        prev_time = time.time()

        try:
            while True:
                current_time = time.time()
                delta_time = current_time - prev_time

                if self.params['playing'] and not self.params['show_all_frames']:
                    fps = self.gui_framerate.value
                    frame_duration = 1.0 / fps if fps > 0 else 0.033

                    if delta_time >= frame_duration:
                        self.params['current_frame'] = (self.params['current_frame'] + 1) % self.total_frames
                        self.gui_timestep.value = self.params['current_frame']
                        self.update_frame_visibility()
                        prev_time = current_time

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nVisualization stopped")
        except Exception as e:
            print(f"Error in visualization loop: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Multi-view point cloud alignment to cam0 coordinate system'
    )

    # Registration parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory containing data')
    parser.add_argument('--views', type=str, nargs='+', default=['cam0', 'cam1'],
                        help='List of view names')
    parser.add_argument('--reference_view', type=str, default='cam0',
                        help='Reference view for alignment (default: cam0)')
    parser.add_argument('--max_frames', type=int, default=30,
                        help='Maximum frames per view to process')

    # Camera pose parameters
    parser.add_argument('--pose_dir1', type=str, default=None,
                        help='Directory containing camera poses for view1')
    parser.add_argument('--pose_dir2', type=str, default=None,
                        help='Directory containing camera poses for view2')
    parser.add_argument('--view1_name', type=str, default='cam0',
                        help='Name of view1 (default: cam0)')
    parser.add_argument('--view2_name', type=str, default='cam1',
                        help='Name of view2 (default: cam1)')

    # Mask parameters
    parser.add_argument('--mask_dir1', type=str, default=None,
                        help='Directory containing mask images for view1')
    parser.add_argument('--mask_dir2', type=str, default=None,
                        help='Directory containing mask images for view2')
    parser.add_argument('--apply_masks', action='store_true', default=False,
                        help='Apply masks to filter point clouds')

    # Visualization parameters
    parser.add_argument('--point_size', type=float, default=0.002,
                        help='Point size for visualization')
    parser.add_argument('--downsample_factor', type=int, default=4,
                        help='Downsample factor for point clouds')
    parser.add_argument('--share', action='store_true', default=False,
                        help='Request share URL for remote viewing')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Enable verbose output')

    parser.add_argument('--skip_alignment', action='store_true', default=False,
                        help='Skip alignment and only run visualization')

    args = parser.parse_args()

    if not args.skip_alignment:
        # Step 1: Perform simple center alignment
        print(f"{'=' * 60}")
        print("STEP 1: ALIGNING ALL VIEWS TO CAM0")
        print(f"{'=' * 60}")

        registrar = MultiViewRegistration(
            base_dir=Path(args.data_dir),
            verbose=args.verbose
        )

        registrar.setup_views(args.views)

        # Load data for each view
        for view_name in args.views:
            registrar.load_view_data(view_name, max_frames=args.max_frames)

        # Load camera poses if provided
        if args.pose_dir1:
            registrar.load_camera_poses(Path(args.pose_dir1), args.view1_name)
        if args.pose_dir2:
            registrar.load_camera_poses(Path(args.pose_dir2), args.view2_name)

        # Load masks if specified
        if args.mask_dir1 and args.apply_masks:
            print(f"\nLoading masks for {args.view1_name} from: {args.mask_dir1}")
            mask_path = Path(args.mask_dir1)
            registrar.load_masks(mask_path, args.view1_name, max_frames=args.max_frames)

        if args.mask_dir2 and args.apply_masks:
            print(f"\nLoading masks for {args.view2_name} from: {args.mask_dir2}")
            mask_path = Path(args.mask_dir2)
            registrar.load_masks(mask_path, args.view2_name, max_frames=args.max_frames)

        # Simple center alignment to cam0
        transformations = registrar.simple_center_alignment(reference_view=args.reference_view)

        registration_results_dir = registrar.output_dir

        print(f"\nAlignment completed!")
        print(f"Results saved to: {registration_results_dir}")
        print(f"All point clouds are now in {args.reference_view} coordinate system")

    else:
        # Skip alignment, use existing results
        registration_results_dir = Path(args.data_dir) / 'registration_results'
        if not registration_results_dir.exists():
            raise FileNotFoundError(f"Registration results not found: {registration_results_dir}")

        print(f"Using existing alignment results from: {registration_results_dir}")

    # Step 2: Visualize with camera poses
    print(f"\n{'=' * 60}")
    print("STEP 2: VISUALIZING WITH CAMERA POSES")
    print(f"{'=' * 60}")
    print(f"Showing six different elements:")
    print(f"  1. {args.view1_name} camera pose (red)")
    print(f"  2. {args.view2_name} original camera pose (green)")
    print(f"  3. {args.view2_name} registered camera pose in {args.view1_name} coordinate system (blue)")
    print(f"  4. {args.view1_name} point cloud (purple)")
    print(f"  5. {args.view2_name} point cloud (orange)")
    print(f"  6. {args.view2_name} aligned point cloud in {args.view1_name} coordinate system (blue)")

    visualizer = MultiCameraVisualizer(
        registration_results_dir=registration_results_dir,
        verbose=args.verbose
    )

    # Update visualization parameters
    visualizer.params['point_size'] = args.point_size
    visualizer.params['downsample_factor'] = args.downsample_factor
    visualizer.params['max_frames_display'] = args.max_frames

    # Start visualization with camera poses
    visualizer.create_visualization(
        share=args.share,
        pose_dir1=Path(args.pose_dir1) if args.pose_dir1 else None,
        pose_dir2=Path(args.pose_dir2) if args.pose_dir2 else None,
        view1_name=args.view1_name,
        view2_name=args.view2_name
    )


if __name__ == "__main__":
    main()