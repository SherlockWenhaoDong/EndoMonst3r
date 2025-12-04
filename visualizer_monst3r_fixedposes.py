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

import numpy as onp
from tqdm.auto import tqdm

import viser
import viser.extras
import viser.transforms as tf
import matplotlib.cm as cm  # For colormap


def parse_calibration_xml(xml_path: Path):
    """
    Parse camera calibration XML file to extract intrinsic parameters.

    Expected XML structure:
    <?xml version="1.0" encoding="utf-8"?>
    <propertyfile version="1.1" name="">
        <param name="imageSize">1264 1008 </param>
        <param name="K">1045.87471656702 0 589.864029607369 0 1044.63816741387 506.467596734399 0 0 1 </param>
        <param name="distortion">-0.0219845811218631 0.0258416946521303 -0.000632101689148172 0.00221558249094752 0 </param>
        <param name="registration">1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 </param>
    </propertyfile>

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


def load_mask(mask_path: Path, target_shape=None):
    """
    Load and process mask image.

    Args:
        mask_path: Path to mask image
        target_shape: Target shape for resizing (height, width)

    Returns:
        Binary mask array (0 or 1)
    """
    if not mask_path.exists():
        return None

    try:
        # Load mask image
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None

        # Binarize mask (0 for background, 1 for foreground)
        mask_binary = (mask > 0).astype(onp.float32)

        # Resize if target shape is provided
        if target_shape is not None:
            h_target, w_target = target_shape
            mask_resized = cv2.resize(mask_binary, (w_target, h_target),
                                      interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask_resized > 0.5).astype(onp.float32)

        return mask_binary

    except Exception as e:
        print(f"Error loading mask {mask_path}: {e}")
        return None


class FrameData:
    """Simple class to hold frame data."""

    def __init__(self, rgb, depth=None, mask=None, K=None, pose=None, rgb_path=None):
        self.rgb = rgb  # RGB图像（已应用mask）
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
            Tuple of (foreground_positions, foreground_colors, background_positions, background_colors)
        """
        if self.depth is None or self.K is None:
            # 如果没有深度图，返回空数组
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
            # 使用mask进行过滤
            foreground_mask = mask_flat > mask_threshold
            background_mask = ~foreground_mask
        else:
            # 如果不使用mask或没有mask，将所有点视为前景
            foreground_mask = onp.ones_like(depth_flat, dtype=bool)
            background_mask = onp.zeros_like(depth_flat, dtype=bool)

        # 过滤有效深度点（深度大于0）
        valid_depth_mask = depth_flat > 0

        # 结合mask过滤和深度过滤
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
            colors_fg = rgb_fg / 255.0  # 归一化到[0, 1]
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
            colors_bg = rgb_bg / 255.0  # 归一化到[0, 1]
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
        FrameData object
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

    # Find depth image (similar to EndoDUSt3R logic)
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
                        depth = depth_img.astype(onp.float32) / 1000.0  # 毫米转换为米
                    elif depth_img.dtype == np.uint8:
                        depth = depth_img.astype(onp.float32) / 255.0 * 10.0  # 假设最大深度10米
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

    # Find mask image (similar to EndoDUSt3R logic)
    mask_path = None

    # Try mask directory pattern: rgb_dir + '_mask'
    mask_dir = Path(str(data_path) + '_mask')
    mask_file = mask_dir / f'{rgb_name}_combined_masks.png'

    if mask_file.exists():
        mask_path = mask_file
    else:
        # Try other possible patterns
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

    # If standard mask directory doesn't exist, try other possible directories
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

    # 应用mask到RGB图像
    rgb_with_mask = rgb_original.copy()

    if apply_mask_to_image and mask is not None:
        # 确保mask是3通道的
        if len(mask.shape) == 2:
            mask_3d = onp.repeat(mask[:, :, onp.newaxis], 3, axis=2)
        else:
            mask_3d = mask
        # 应用mask：背景变为黑色
        rgb_with_mask = (rgb_original * mask_3d).astype(onp.uint8)
        if verbose:
            print(f"  Applied mask to RGB image")

    # Set camera parameters
    if calibration_data and 'K' in calibration_data:
        K = calibration_data['K'].copy()

        # Adjust intrinsics for cropped image (same as EndoDUSt3R)
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

    # 创建FrameData对象
    frame_data = FrameData(
        rgb=rgb_with_mask,
        depth=depth,
        mask=mask,
        K=K,
        pose=pose,
        rgb_path=rgb_path
    )

    return frame_data


def main(
        data_path: Path = Path("./demo_tmp/NULL"),
        calib_xml_path: Path = None,
        downsample_factor: int = 1,
        max_frames: int = 100,
        share: bool = False,
        point_size: float = 0.001,
        camera_frustum_scale: float = 0.02,
        xyzw: bool = True,
        axes_scale: float = 0.25,
        bg_downsample_factor: int = 1,
        cam_thickness: float = 1.5,
        fixed_pose: bool = True,
        mask_threshold: float = 0.5,
        show_background: bool = False,
        verbose: bool = False,
        stride: int = 1,
        apply_mask_to_image: bool = True,  # 是否将mask应用到图像
        filter_by_mask: bool = True,  # 是否使用mask过滤点云
) -> None:
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    server.scene.set_up_direction('-z')

    print("Loading frames with mask filtering...")
    print(f"  Apply mask to image: {apply_mask_to_image}")
    print(f"  Filter points by mask: {filter_by_mask}")
    print(f"  Mask threshold: {mask_threshold}")
    print(f"  Show background: {show_background}")

    # Load calibration data if XML path is provided
    calibration_data = None
    if calib_xml_path and calib_xml_path.exists():
        print(f"Loading calibration from: {calib_xml_path}")
        calibration_data = parse_calibration_xml(calib_xml_path)
        if calibration_data:
            print(
                f"  Image size: {calibration_data.get('image_width', 'N/A')}x{calibration_data.get('image_height', 'N/A')}")
            print(
                f"  Focal length: fx={calibration_data.get('fx', 'N/A'):.2f}, fy={calibration_data.get('fy', 'N/A'):.2f}")
            print(
                f"  Principal point: cx={calibration_data.get('cx', 'N/A'):.2f}, cy={calibration_data.get('cy', 'N/A'):.2f}")

    # 根据EndoDUSt3R的逻辑查找图像
    # 检查是否为frames_cam0目录
    if 'frames_cam0' in str(data_path):
        rgb_files_dir = data_path
    elif (data_path / 'frames_cam0').exists():
        rgb_files_dir = data_path / 'frames_cam0'
    else:
        # 查找包含frames_cam0的子目录
        frames_dirs = list(data_path.glob('*/frames_cam0'))
        if frames_dirs:
            rgb_files_dir = frames_dirs[0]
            print(f"Found frames directory: {rgb_files_dir}")
        else:
            # 直接使用数据目录
            rgb_files_dir = data_path

    print(f"Looking for RGB images in: {rgb_files_dir}")

    # 获取所有RGB图像，按文件名中的数字排序
    rgb_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    rgb_files = []
    for ext in rgb_extensions:
        rgb_files.extend(glob.glob(str(rgb_files_dir / ext)))

    # 按文件名中的数字排序
    def extract_number(filename):
        numbers = re.findall(r'\d+', os.path.basename(filename))
        return int(numbers[0]) if numbers else 0

    rgb_files = sorted(rgb_files, key=extract_number)
    rgb_files = [Path(f) for f in rgb_files]

    if not rgb_files:
        print(f"No RGB images found in {rgb_files_dir}")
        return

    # 应用stride，只加载部分帧
    rgb_files = rgb_files[::stride]
    rgb_files = rgb_files[:max_frames]  # 限制最大帧数

    num_frames = len(rgb_files)
    print(f"Found {num_frames} frames (with stride={stride})")

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=30
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )
        gui_show_all_frames = server.gui.add_checkbox("Show all frames", False)
        gui_stride = server.gui.add_slider(
            "Stride",
            min=1,
            max=num_frames,
            step=1,
            initial_value=1,
            disabled=True,  # Initially disabled
        )

    with server.gui.add_folder("Mask Settings"):
        gui_mask_threshold = server.gui.add_slider(
            "Mask Threshold",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=mask_threshold,
        )
        gui_show_background = server.gui.add_checkbox(
            "Show Background",
            initial_value=show_background,
        )
        gui_apply_mask_to_image = server.gui.add_checkbox(
            "Apply Mask to Image",
            initial_value=apply_mask_to_image,
        )
        gui_filter_by_mask = server.gui.add_checkbox(
            "Filter Points by Mask",
            initial_value=filter_by_mask,
        )

    # Add recording UI.
    with server.gui.add_folder("Recording"):
        gui_record_scene = server.gui.add_button("Record Scene")

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value or gui_show_all_frames.value
        gui_next_frame.disabled = gui_playing.value or gui_show_all_frames.value
        gui_prev_frame.disabled = gui_playing.value or gui_show_all_frames.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        if not gui_show_all_frames.value:
            with server.atomic():
                frame_nodes[current_timestep].visible = True
                frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Show or hide all frames based on the checkbox.
    @gui_show_all_frames.on_update
    def _(_) -> None:
        gui_stride.disabled = not gui_show_all_frames.value  # Enable/disable stride slider
        if gui_show_all_frames.value:
            # Show frames with stride
            stride = gui_stride.value
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = (i % stride == 0)
            # Disable playback controls
            gui_playing.disabled = True
            gui_timestep.disabled = True
            gui_next_frame.disabled = True
            gui_prev_frame.disabled = True
        else:
            # Show only the current frame
            current_timestep = gui_timestep.value
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = i == current_timestep
            # Re-enable playback controls
            gui_playing.disabled = False
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

    # Update frame visibility when the stride changes.
    @gui_stride.on_update
    def _(_) -> None:
        if gui_show_all_frames.value:
            # Update frame visibility based on new stride
            stride = gui_stride.value
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = (i % stride == 0)

    # Recording handler
    @gui_record_scene.on_click
    def _(_):
        gui_record_scene.disabled = True

        # Save the original frame visibility state
        original_visibility = [frame_node.visible for frame_node in frame_nodes]

        rec = server._start_scene_recording()
        rec.set_loop_start()

        # Determine sleep duration based on current FPS
        sleep_duration = 1.0 / gui_framerate.value if gui_framerate.value > 0 else 0.033  # Default to ~30 FPS

        if gui_show_all_frames.value:
            # Record all frames according to the stride
            stride = gui_stride.value
            frames_to_record = [i for i in range(num_frames) if i % stride == 0]
        else:
            # Record the frames in sequence
            frames_to_record = range(num_frames)

        for t in frames_to_record:
            # Update the scene to show frame t
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = (i == t) if not gui_show_all_frames.value else (i % gui_stride.value == 0)
            server.flush()
            rec.insert_sleep(sleep_duration)

        # set all invisible
        with server.atomic():
            for frame_node in frame_nodes:
                frame_node.visible = False

        # Finish recording
        bs = rec.end_and_serialize()

        # Save the recording to a file
        output_path = Path(f"./viser_result/recording_{str(data_path).split('/')[-1]}.viser")
        # make sure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(bs)
        print(f"Recording saved to {output_path.resolve()}")

        # Restore the original frame visibility state
        with server.atomic():
            for frame_node, visibility in zip(frame_nodes, original_visibility):
                frame_node.visible = visibility
        server.flush()

        gui_record_scene.disabled = False

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(onp.array([onp.pi / 2.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )

    frame_nodes: list[viser.FrameHandle] = []
    bg_nodes: list[viser.PointCloudHandle] = []

    # 存储点云数据，以便在设置更改时更新
    frame_pointclouds = []
    bg_pointclouds = []

    for i, rgb_file in enumerate(tqdm(rgb_files, desc="Loading frames")):
        if verbose:
            print(f"\nProcessing frame {i}: {rgb_file.name}")

        # Load frame data using EndoDUSt3R compatible logic
        frame = load_frame_data(
            rgb_files_dir,
            rgb_file,
            calibration_data,
            fixed_pose,
            verbose,
            apply_mask_to_image=apply_mask_to_image
        )

        if frame is None or frame.rgb is None:
            print(f"Warning: Could not load frame {i}: {rgb_file.name}")
            # 添加空节点以保持索引一致
            frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))
            frame_pointclouds.append((None, None, None, None))
            continue

        # Get point cloud filtered by mask
        positions, colors, positions_bg, colors_bg = frame.get_point_cloud_with_mask(
            downsample_factor=downsample_factor,
            mask_threshold=mask_threshold,
            filter_by_mask=filter_by_mask
        )

        # 存储点云数据
        frame_pointclouds.append((positions, colors, positions_bg, colors_bg))

        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # Place the foreground point cloud in the frame.
        if len(positions) > 0:
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/point_cloud",
                points=positions,
                colors=colors,
                point_size=point_size,
                point_shape="rounded",
            )

        # 添加背景点云（如果存在且需要显示）
        if show_background and len(positions_bg) > 0:
            bg_pc = server.scene.add_point_cloud(
                name=f"/frames/t{i}/background",
                points=positions_bg,
                colors=colors_bg,
                point_size=point_size * 0.3,  # 背景点更小
                point_shape="rounded",
                visible=show_background,  # 初始可见性
            )
            bg_nodes.append(bg_pc)
        else:
            bg_nodes.append(None)

        # Compute color for frustum based on frame index.
        norm_i = i / (num_frames - 1) if num_frames > 1 else 0  # Normalize index to [0, 1]
        color_rgba = cm.viridis(norm_i)  # Get RGBA color from colormap
        color_rgb = color_rgba[:3]  # Use RGB components

        # Calculate image dimensions
        if frame.rgb is not None:
            image_height, image_width = frame.rgb.shape[:2]
        elif calibration_data and 'image_width' in calibration_data and 'image_height' in calibration_data:
            image_width = calibration_data['image_width']
            image_height = calibration_data['image_height']
        else:
            image_width = 1920
            image_height = 1080

        # Calculate FOV from intrinsic matrix
        fx = frame.K[0, 0]
        fy = frame.K[1, 1] if frame.K.shape[0] > 1 else fx
        fov_x = 2 * onp.arctan2(image_width / 2, fx)
        fov_y = 2 * onp.arctan2(image_height / 2, fy)
        fov = onp.max([fov_x, fov_y])  # Use larger FOV
        aspect = image_width / image_height

        # Place the frustum with the computed color.
        if frame.rgb is not None:
            # Use downsampled image for frustum preview
            if downsample_factor > 1:
                image = frame.rgb[::downsample_factor, ::downsample_factor]
            else:
                image = frame.rgb
        else:
            image = None

        server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            scale=camera_frustum_scale,
            image=image,
            wxyz=tf.SO3.from_matrix(frame.T_world_camera[:3, :3]).wxyz,
            position=frame.T_world_camera[:3, 3],
            color=color_rgb,
            thickness=cam_thickness,
        )

        # Add some axes.
        server.scene.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=camera_frustum_scale * axes_scale * 10,
            axes_radius=camera_frustum_scale * axes_scale,
        )

    if not frame_nodes:
        print("No valid frames loaded. Exiting.")
        return

    # Initialize frame visibility.
    for i, frame_node in enumerate(frame_nodes):
        if gui_show_all_frames.value:
            frame_node.visible = (i % gui_stride.value == 0)
        else:
            frame_node.visible = i == gui_timestep.value

    # 更新mask设置的回调函数
    @gui_mask_threshold.on_update
    def _(_):
        nonlocal filter_by_mask, mask_threshold
        mask_threshold = gui_mask_threshold.value

        # 重新计算所有帧的点云
        if gui_filter_by_mask.value:
            for i, (frame_node, pointcloud_data) in enumerate(zip(frame_nodes, frame_pointclouds)):
                if pointcloud_data[0] is not None:  # 如果有有效的点云数据
                    # 重新加载帧数据
                    if i < len(rgb_files):
                        frame = load_frame_data(
                            rgb_files_dir,
                            rgb_files[i],
                            calibration_data,
                            fixed_pose,
                            False,  # 不显示详细信息
                            apply_mask_to_image=gui_apply_mask_to_image.value
                        )

                        if frame is not None:
                            # 重新计算点云
                            positions, colors, positions_bg, colors_bg = frame.get_point_cloud_with_mask(
                                downsample_factor=downsample_factor,
                                mask_threshold=mask_threshold,
                                filter_by_mask=gui_filter_by_mask.value
                            )

                            # 移除旧的点云
                            for child in server.scene.get_children(f"/frames/t{i}"):
                                if "point_cloud" in child.name or "background" in child.name:
                                    child.remove()

                            # 添加新的前景点云
                            if len(positions) > 0:
                                server.scene.add_point_cloud(
                                    name=f"/frames/t{i}/point_cloud",
                                    points=positions,
                                    colors=colors,
                                    point_size=point_size,
                                    point_shape="rounded",
                                )

                            # 更新背景点云
                            if i < len(bg_nodes) and bg_nodes[i] is not None:
                                bg_nodes[i].remove()
                                bg_nodes[i] = None

                            if gui_show_background.value and len(positions_bg) > 0:
                                bg_pc = server.scene.add_point_cloud(
                                    name=f"/frames/t{i}/background",
                                    points=positions_bg,
                                    colors=colors_bg,
                                    point_size=point_size * 0.3,
                                    point_shape="rounded",
                                    visible=gui_show_background.value,
                                )
                                bg_nodes[i] = bg_pc

    @gui_show_background.on_update
    def _(_):
        nonlocal show_background
        show_background = gui_show_background.value

        # 更新所有背景点云的可见性
        for i, bg_node in enumerate(bg_nodes):
            if bg_node is not None:
                bg_node.visible = show_background

    @gui_apply_mask_to_image.on_update
    def _(_):
        # 重新加载所有帧并应用新的mask设置
        with server.atomic():
            for i, (frame_node, rgb_file) in enumerate(zip(frame_nodes, rgb_files)):
                if rgb_file is not None:
                    frame = load_frame_data(
                        rgb_files_dir,
                        rgb_file,
                        calibration_data,
                        fixed_pose,
                        False,
                        apply_mask_to_image=gui_apply_mask_to_image.value
                    )

                    if frame is not None:
                        # 更新frustum的图像
                        if frame.rgb is not None:
                            if downsample_factor > 1:
                                image = frame.rgb[::downsample_factor, ::downsample_factor]
                            else:
                                image = frame.rgb

                            # 更新frustum的图像
                            frustum_path = f"/frames/t{i}/frustum"
                            if frustum_path in server.scene.node_names:
                                server.scene.node_names[frustum_path].image = image

    @gui_filter_by_mask.on_update
    def _(_):
        nonlocal filter_by_mask
        filter_by_mask = gui_filter_by_mask.value

        # 重新计算所有帧的点云
        with server.atomic():
            for i, (frame_node, rgb_file) in enumerate(zip(frame_nodes, rgb_files)):
                if rgb_file is not None:
                    frame = load_frame_data(
                        rgb_files_dir,
                        rgb_file,
                        calibration_data,
                        fixed_pose,
                        False,
                        apply_mask_to_image=gui_apply_mask_to_image.value
                    )

                    if frame is not None:
                        # 重新计算点云
                        positions, colors, positions_bg, colors_bg = frame.get_point_cloud_with_mask(
                            downsample_factor=downsample_factor,
                            mask_threshold=mask_threshold,
                            filter_by_mask=filter_by_mask
                        )

                        # 移除旧的点云
                        for child in server.scene.get_children(f"/frames/t{i}"):
                            if "point_cloud" in child.name or "background" in child.name:
                                child.remove()

                        # 添加新的前景点云
                        if len(positions) > 0:
                            server.scene.add_point_cloud(
                                name=f"/frames/t{i}/point_cloud",
                                points=positions,
                                colors=colors,
                                point_size=point_size,
                                point_shape="rounded",
                            )

                        # 更新背景点云
                        if i < len(bg_nodes) and bg_nodes[i] is not None:
                            bg_nodes[i].remove()
                            bg_nodes[i] = None

                        if gui_show_background.value and len(positions_bg) > 0:
                            bg_pc = server.scene.add_point_cloud(
                                name=f"/frames/t{i}/background",
                                points=positions_bg,
                                colors=colors_bg,
                                point_size=point_size * 0.3,
                                point_shape="rounded",
                                visible=gui_show_background.value,
                            )
                            bg_nodes[i] = bg_pc

    # Playback update loop.
    prev_timestep = gui_timestep.value
    print("\nVisualization ready. Open the browser to view the point clouds.")
    print("Use the Mask Settings panel to adjust mask filtering.")

    while True:
        if gui_playing.value and not gui_show_all_frames.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize 3D reconstruction with mask filtering (EndoDUSt3R compatible).")

    # Define arguments
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("./demo_tmp/NULL"),
        help="Path to the data directory (frames_cam0 or parent directory)"
    )
    parser.add_argument(
        "--calib-xml-path",
        type=Path,
        default=None,
        help="Path to calibration XML file"
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=4,
        help="Downsample factor for point cloud"
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold for mask filtering"
    )
    parser.add_argument(
        "--show-background",
        action="store_true",
        help="Show background points"
    )
    parser.add_argument(
        "--fixed-pose",
        action="store_true",
        default=True,
        help="Use fixed identity matrix for camera poses"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum number of frames to load"
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.001,
        help="Point size"
    )
    parser.add_argument(
        "--camera-frustum-scale",
        type=float,
        default=0.015,
        help="Camera frustum scale"
    )
    parser.add_argument(
        "--cam-thickness",
        type=float,
        default=1.5,
        help="Camera frustum thickness"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for loading frames (skip frames)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose information"
    )
    parser.add_argument(
        "--apply-mask-to-image",
        action="store_true",
        default=True,
        help="Apply mask to RGB image (make background black)"
    )
    parser.add_argument(
        "--filter-by-mask",
        action="store_true",
        default=True,
        help="Filter point cloud points by mask"
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        calib_xml_path=args.calib_xml_path,
        downsample_factor=args.downsample_factor,
        max_frames=args.max_frames,
        share=False,
        point_size=args.point_size,
        camera_frustum_scale=args.camera_frustum_scale,
        xyzw=True,
        axes_scale=0.1,
        bg_downsample_factor=1,
        cam_thickness=args.cam_thickness,
        fixed_pose=args.fixed_pose,
        mask_threshold=args.mask_threshold,
        show_background=args.show_background,
        verbose=args.verbose,
        stride=args.stride,
        apply_mask_to_image=args.apply_mask_to_image,
        filter_by_mask=args.filter_by_mask,
    )