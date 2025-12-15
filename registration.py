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

        ply_dir = self.base_dir / f'frames_{view_name}_ply_files'

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


class ViserVisualizer:
    """Interactive 3D visualization showing only aligned point clouds in cam0 coordinate system."""

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
        }

        # Load registration results
        self.transformations = self.load_transformations()
        self.view_data = self.load_aligned_pointclouds()

        # Initialize Viser server
        self.server = None

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

    def create_visualization(self, share: bool = False):
        """Create and launch the interactive visualization."""
        self.server = viser.ViserServer()

        if share:
            self.server.request_share_url()

        self.server.scene.set_up_direction('-z')

        print(f"\n{'=' * 60}")
        print(f"ALIGNED POINT CLOUDS VISUALIZATION (cam0 coordinate system)")
        print(f"{'=' * 60}")

        # Get all frame indices
        all_frame_indices = set()
        for view_data in self.view_data.values():
            for frame in view_data['frames']:
                all_frame_indices.add(frame['frame_idx'])

        self.frame_indices = sorted(all_frame_indices)
        self.total_frames = len(self.frame_indices)

        print(f"Views: {list(self.view_data.keys())}")
        print(f"Total unique frames: {self.total_frames}")
        print(f"All point clouds are aligned to cam0 coordinate system")

        # Create UI
        self.create_ui()

        # Create visualization nodes
        self.create_pointcloud_nodes()

        # Start visualization
        print("\nVisualization ready. Open the browser to view aligned point clouds.")
        print(f"Server URL: http://localhost:8080")
        print("Note: All point clouds are shown in cam0 coordinate system")

        self.run_visualization_loop()

    def create_ui(self):
        """Create simplified user interface."""
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

        # View visibility - CRITICAL: Control which views to show
        with self.server.gui.add_folder("View Visibility"):
            self.view_checkboxes = {}
            for view_name in self.view_data.keys():
                # By default, show all views
                self.view_checkboxes[view_name] = self.server.gui.add_checkbox(f"Show {view_name}", True)

        # Setup event handlers
        self.setup_event_handlers()

    def setup_event_handlers(self):
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

        for view_name, checkbox in self.view_checkboxes.items():
            @checkbox.on_update
            def handler(_, view_name=view_name):
                self.update_view_visibility(view_name)

    def create_pointcloud_nodes(self):
        """Create point cloud visualization nodes."""
        # Create base frame at origin (cam0 coordinate system)
        self.server.scene.add_frame(
            "/origin",
            wxyz=tf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0])).wxyz,
            position=(0, 0, 0),
            axes_length=0.1,
            axes_radius=0.01,
        )

        # Color map for different views
        colors = cm.tab10(np.linspace(0, 1, len(self.view_data)))
        self.view_colors = {}
        for i, view_name in enumerate(self.view_data.keys()):
            self.view_colors[view_name] = colors[i][:3]

        # Store nodes
        self.frame_nodes = {}
        self.pointcloud_nodes = {}

        # Create nodes for each view
        for view_name, data in self.view_data.items():
            self.frame_nodes[view_name] = []
            self.pointcloud_nodes[view_name] = []

            view_color = self.view_colors[view_name]

            for i, frame_data in enumerate(data['frames']):
                frame_idx = frame_data['frame_idx']

                # Create a simple frame for organization
                frame_node = self.server.scene.add_frame(
                    f"/views/{view_name}/t{frame_idx}",
                    show_axes=False
                )
                self.frame_nodes[view_name].append(frame_node)

                # Get point cloud
                pcd = frame_data['pcd']

                # Downsample if needed
                if self.params['downsample_factor'] > 1:
                    pcd = pcd.voxel_down_sample(self.params['downsample_factor'] / 1000.0)

                # Get points and colors
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors) if pcd.has_colors() else None

                if len(points) > 0:
                    if colors is None:
                        colors = np.tile(view_color, (len(points), 1))

                    # Create point cloud node
                    pointcloud_node = self.server.scene.add_point_cloud(
                        name=f"/views/{view_name}/t{frame_idx}/points",
                        points=points,
                        colors=colors,
                        point_size=self.params['point_size'],
                        point_shape="rounded",
                    )
                    self.pointcloud_nodes[view_name].append(pointcloud_node)

        # Update initial visibility
        self.update_frame_visibility()

    def update_frame_visibility(self):
        """Update which frames are visible based on current timestep."""
        if self.total_frames == 0:
            return

        current_frame_idx = self.frame_indices[self.params['current_frame']]

        with self.server.atomic():
            for view_name, frame_list in self.frame_nodes.items():
                view_enabled = self.view_checkboxes[view_name].value

                for i, frame_node in enumerate(frame_list):
                    if i < len(self.view_data[view_name]['frames']):
                        frame_data = self.view_data[view_name]['frames'][i]
                        should_show = (frame_data['frame_idx'] == current_frame_idx) and view_enabled

                        frame_node.visible = should_show

                        # Update point cloud visibility
                        if i < len(self.pointcloud_nodes[view_name]):
                            self.pointcloud_nodes[view_name][i].visible = should_show

    def show_all_frames_with_stride(self, stride: int):
        """Show all frames with stride."""
        with self.server.atomic():
            for view_name, frame_list in self.frame_nodes.items():
                view_enabled = self.view_checkboxes[view_name].value

                for i, frame_node in enumerate(frame_list):
                    if i < len(self.view_data[view_name]['frames']):
                        frame_data = self.view_data[view_name]['frames'][i]

                        try:
                            idx_in_sorted = self.frame_indices.index(frame_data['frame_idx'])
                            should_show = (idx_in_sorted % stride == 0) and view_enabled
                        except ValueError:
                            should_show = False

                        frame_node.visible = should_show

                        if i < len(self.pointcloud_nodes[view_name]):
                            self.pointcloud_nodes[view_name][i].visible = should_show

    def update_view_visibility(self, view_name: str):
        """Update visibility of a specific view."""
        if self.params['show_all_frames']:
            stride = self.gui_stride.value
            self.show_all_frames_with_stride(stride)
        else:
            self.update_frame_visibility()

    def update_point_sizes(self):
        """Update point sizes."""
        with self.server.atomic():
            for view_name in self.view_data.keys():
                for node in self.pointcloud_nodes[view_name]:
                    node.point_size = self.params['point_size']

    def reload_point_clouds(self):
        """Reload point clouds with new downsample factor."""
        print("Reloading point clouds...")

        with self.server.atomic():
            for view_name in self.view_data.keys():
                for node in self.pointcloud_nodes[view_name]:
                    node.remove()
                self.pointcloud_nodes[view_name] = []

        for view_name, data in self.view_data.items():
            view_color = self.view_colors[view_name]

            for i, frame_data in enumerate(data['frames']):
                pcd = frame_data['pcd']

                if self.params['downsample_factor'] > 1:
                    pcd = pcd.voxel_down_sample(self.params['downsample_factor'] / 1000.0)

                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors) if pcd.has_colors() else None

                if len(points) > 0:
                    if colors is None:
                        colors = np.tile(view_color, (len(points), 1))

                    node = self.server.scene.add_point_cloud(
                        name=f"/views/{view_name}/t{frame_data['frame_idx']}/points",
                        points=points,
                        colors=colors,
                        point_size=self.params['point_size'],
                        point_shape="rounded",
                    )
                    self.pointcloud_nodes[view_name].append(node)

        if self.params['show_all_frames']:
            stride = self.gui_stride.value
            self.show_all_frames_with_stride(stride)
        else:
            self.update_frame_visibility()

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
    parser.add_argument('--views', type=str, nargs='+', default=['cam0', 'cam1', 'cam2', 'cam3'],
                        help='List of view names')
    parser.add_argument('--reference_view', type=str, default='cam0',
                        help='Reference view for alignment (default: cam0)')
    parser.add_argument('--max_frames', type=int, default=30,
                        help='Maximum frames per view to process')

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

    # # Step 2: Visualize aligned point clouds
    # print(f"\n{'=' * 60}")
    # print("STEP 2: VISUALIZING ALIGNED POINT CLOUDS")
    # print(f"{'=' * 60}")
    #
    # visualizer = ViserVisualizer(
    #     registration_results_dir=registration_results_dir,
    #     verbose=args.verbose
    # )
    #
    # # Update visualization parameters
    # visualizer.params['point_size'] = args.point_size
    # visualizer.params['downsample_factor'] = args.downsample_factor
    # visualizer.params['max_frames_display'] = args.max_frames
    #
    # # Start visualization
    # visualizer.create_visualization(share=args.share)


if __name__ == "__main__":
    main()