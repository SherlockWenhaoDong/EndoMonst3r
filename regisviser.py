import time
import sys
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm.auto import tqdm

import viser
import viser.transforms as tf
import open3d as o3d
import warnings

warnings.filterwarnings('ignore')


def parse_calibration_xml(xml_path: Path, ignore_extrinsics=False):
    """Parse camera calibration XML file for intrinsics and extrinsics."""
    if not xml_path.exists():
        raise FileNotFoundError(f"Calibration XML file not found: {xml_path}")

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        calibration_data = {}

        # Parse intrinsic matrix K
        K_elem = root.find("./param[@name='K']")
        if K_elem is not None:
            K_values = list(map(float, K_elem.text.strip().split()))
            if len(K_values) >= 9:
                K = np.array(K_values, dtype=np.float32).reshape(3, 3)
                calibration_data['K'] = K
                calibration_data['fx'] = K[0, 0]
                calibration_data['fy'] = K[1, 1]
                calibration_data['cx'] = K[0, 2]
                calibration_data['cy'] = K[1, 2]
                print(f"✓ Loaded camera intrinsics")

        if not ignore_extrinsics:
            # Try to find extrinsic parameters (rotation and translation)
            extrinsics = np.eye(4, dtype=np.float32)

            # Look for common patterns in calibration files
            # Pattern 1: 4x4 transformation matrix
            transform_elem = root.find(".//transform")
            if transform_elem is not None:
                try:
                    values = list(map(float, transform_elem.text.strip().split()))
                    if len(values) == 16:
                        extrinsics = np.array(values, dtype=np.float32).reshape(4, 4)
                        print(f"✓ Found 4x4 transformation matrix")
                except:
                    pass

            # Pattern 2: Separate rotation and translation
            R_elem = root.find(".//rotation")
            T_elem = root.find(".//translation")
            if R_elem is not None and T_elem is not None:
                try:
                    R_values = list(map(float, R_elem.text.strip().split()))
                    T_values = list(map(float, T_elem.text.strip().split()))

                    if len(R_values) == 9:  # 3x3 matrix
                        R = np.array(R_values, dtype=np.float32).reshape(3, 3)
                    elif len(R_values) == 3:  # Rodrigues vector
                        R, _ = cv2.Rodrigues(np.array(R_values, dtype=np.float32))
                    else:
                        R = np.eye(3, dtype=np.float32)

                    if len(T_values) >= 3:
                        T = np.array(T_values[:3], dtype=np.float32)
                    else:
                        T = np.zeros(3, dtype=np.float32)

                    extrinsics[:3, :3] = R
                    extrinsics[:3, 3] = T
                    print(f"✓ Found separate rotation and translation")
                except Exception as e:
                    print(f"Warning: Could not parse extrinsics: {e}")

            # If no extrinsics found, create a default transformation
            if np.allclose(extrinsics, np.eye(4)):
                print(f"✗ No extrinsics found in XML")
                return calibration_data
            else:
                print(f"✓ Loaded extrinsics from XML")
                calibration_data['extrinsics'] = extrinsics

                # Print transformation details
                R = extrinsics[:3, :3]
                t = extrinsics[:3, 3]

                # Compute rotation angles
                sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
                singular = sy < 1e-6

                if not singular:
                    rx = np.arctan2(R[2, 1], R[2, 2])
                    ry = np.arctan2(-R[2, 0], sy)
                    rz = np.arctan2(R[1, 0], R[0, 0])
                else:
                    rx = np.arctan2(-R[1, 2], R[1, 1])
                    ry = np.arctan2(-R[2, 0], sy)
                    rz = 0

                rotation_deg = np.array([rx, ry, rz]) * 180 / np.pi

                print(f"\nView2 -> View1 Transformation from XML:")
                print(f"  Rotation: [{rotation_deg[0]:.1f}, {rotation_deg[1]:.1f}, {rotation_deg[2]:.1f}]°")
                print(f"  Translation: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")

        return calibration_data

    except Exception as e:
        print(f"Error parsing calibration XML {xml_path}: {e}")
        return None


def load_depth_model(model_path, device):
    """Load depth estimation model."""
    print(f"Loading depth model from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"✓ Checkpoint loaded successfully")

        # Import model
        import sys
        project_root = '/home/bygpu/Downloads/EndoMonst3r-main'
        sys.path.insert(0, project_root)
        sys.path.insert(0, os.path.join(project_root, 'dust3r'))

        from dust3r.model_noposes import AsymmetricCroCo3DStereo

        model = AsymmetricCroCo3DStereo(
            pos_embed='RoPE100',
            patch_embed_cls='ManyAR_PatchEmbed',
            img_size=(512, 512),
            head_type='dpt',
            output_mode='pts3d',
            depth_mode=('exp', -float('inf'), float('inf')),
            conf_mode=('exp', 1, float('inf')),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            freeze='encoder',
            use_known_poses=True,
            pose_input_key='camera_pose'
        )

        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.to(device)
        model.eval()
        print("✓ Depth model loaded successfully!")
        return model

    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def estimate_depth_with_model(model, image_path, camera_pose, intrinsics, device='cuda',
                              min_depth=0.01, max_depth=2.0):
    """Estimate depth using model with given camera pose."""
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Preprocess for model
    scale = 512 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    new_h = new_h - (new_h % 16)
    new_w = new_w - (new_w % 16)

    if new_h < 16: new_h = 16
    if new_w < 16: new_w = 16

    img_resized = cv2.resize(img, (new_w, new_h))

    # Adjust intrinsics for resized image
    scale_x = new_w / w
    scale_y = new_h / h
    K_resized = intrinsics.copy()
    K_resized[0, 0] *= scale_x
    K_resized[1, 1] *= scale_y
    K_resized[0, 2] *= scale_x
    K_resized[1, 2] *= scale_y

    # Prepare views for model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    view1 = {
        'img': transform(Image.fromarray(img_resized)).unsqueeze(0).to(device),
        'camera_pose': torch.from_numpy(camera_pose).float().unsqueeze(0).to(device),
        'camera_intrinsics': torch.from_numpy(K_resized).float().unsqueeze(0).to(device),
        'original_size': torch.tensor([[h, w]]).to(device),
        'index': torch.tensor([[0]]).to(device),
        'instance': os.path.basename(image_path)
    }

    view2 = {
        'img': transform(Image.fromarray(img_resized)).unsqueeze(0).to(device),
        'camera_pose': torch.from_numpy(camera_pose).float().unsqueeze(0).to(device),
        'camera_intrinsics': torch.from_numpy(K_resized).float().unsqueeze(0).to(device),
        'original_size': torch.tensor([[h, w]]).to(device),
        'index': torch.tensor([[1]]).to(device),
        'instance': os.path.basename(image_path)
    }

    # Inference
    with torch.no_grad():
        output = model(view1, view2)

    # Extract depth
    if isinstance(output, tuple):
        pred1, _ = output
        if 'pts3d' in pred1:
            pts3d = pred1['pts3d']
            depth = pts3d[0, :, :, 2].cpu()
        elif 'depth' in pred1:
            depth = pred1['depth'][0].cpu() if isinstance(pred1['depth'], list) else pred1['depth'].cpu()
        else:
            raise ValueError("Cannot extract depth from tuple output")
    elif isinstance(output, dict):
        if 'pred1' in output and 'pts3d' in output['pred1']:
            pts3d = output['pred1']['pts3d']
            depth = pts3d[0, :, :, 2].cpu()
        elif 'depth' in output:
            depth = output['depth'][0].cpu() if isinstance(output['depth'], list) else output['depth'].cpu()
        else:
            raise ValueError("Cannot extract depth from dict output")
    else:
        raise ValueError(f"Unexpected output type: {type(output)}")

    # Resize to original size
    depth_np = depth.numpy()
    if depth_np.shape != (h, w):
        depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)

    # Process depth
    depth_np = np.abs(depth_np)
    depth_np = np.clip(depth_np, min_depth, max_depth)
    depth_np[depth_np < min_depth] = np.nan
    depth_np[depth_np > max_depth] = np.nan

    return depth_np


def depth_to_point_cloud(depth_map, rgb_image, K, camera_pose=None,
                         downsample_factor=1, max_points=50000,
                         min_brightness=0.3, min_depth=0.01, apply_filter=True):
    """Convert depth map to point cloud with filtering."""
    h, w = depth_map.shape

    # Create coordinate grid
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Downsample
    if downsample_factor > 1:
        yy = yy[::downsample_factor, ::downsample_factor]
        xx = xx[::downsample_factor, ::downsample_factor]
        depth_map = depth_map[::downsample_factor, ::downsample_factor]
        rgb_image = rgb_image[::downsample_factor, ::downsample_factor]

    # Flatten
    yy = yy.flatten()
    xx = xx.flatten()
    depth_flat = depth_map.flatten()
    rgb_flat = rgb_image.reshape(-1, 3) / 255.0

    # Filter valid points
    valid_depth = ~np.isnan(depth_flat) & (depth_flat > min_depth)

    if apply_filter:
        brightness = rgb_flat.sum(axis=1)
        valid_brightness = brightness > min_brightness
        valid_mask = valid_depth & valid_brightness
    else:
        valid_mask = valid_depth

    if not np.any(valid_mask):
        return np.zeros((0, 3)), np.zeros((0, 3))

    # Get valid points
    yy_valid = yy[valid_mask]
    xx_valid = xx[valid_mask]
    depth_valid = depth_flat[valid_mask]
    colors_valid = rgb_flat[valid_mask]

    # Backproject to camera coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    z = depth_valid
    x = (xx_valid - cx) * z / fx
    y = (yy_valid - cy) * z / fy

    # Points in camera coordinates
    points_cam = np.stack([x, y, z], axis=1)

    # Apply statistical filtering
    if apply_filter and len(points_cam) > 100:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_cam)
            pcd.colors = o3d.utility.Vector3dVector(colors_valid)

            # Statistical outlier removal
            filtered_pcd, ind = pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )

            points_cam = np.asarray(filtered_pcd.points)
            colors_valid = np.asarray(filtered_pcd.colors)
        except Exception as e:
            print(f"  Statistical filtering failed: {e}")

    # Transform to world coordinates if camera_pose is provided
    if camera_pose is not None:
        points_cam_h = np.hstack([points_cam, np.ones((len(points_cam), 1))])
        points_world = (camera_pose @ points_cam_h.T).T[:, :3]
        return points_world, colors_valid
    else:
        return points_cam, colors_valid


class SimpleRegistration:
    """Simple registration module."""

    def __init__(self, verbose=True):
        self.verbose = verbose

    def compute_simple_transform(self, points1, points2):
        """Compute simple transformation using SVD."""
        if len(points1) < 3 or len(points2) < 3:
            return np.eye(4)

        try:
            # Center points
            center1 = np.mean(points1, axis=0)
            center2 = np.mean(points2, axis=0)

            centered1 = points1 - center1
            centered2 = points2 - center2

            # Compute covariance - 注意维度对齐
            if len(centered1) > len(centered2):
                # 随机采样使维度一致
                indices = np.random.choice(len(centered1), len(centered2), replace=False)
                centered1 = centered1[indices]
            elif len(centered2) > len(centered1):
                indices = np.random.choice(len(centered2), len(centered1), replace=False)
                centered2 = centered2[indices]

            H = centered2.T @ centered1

            U, S, Vt = np.linalg.svd(H)
            R = U @ Vt

            # Ensure proper rotation
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = U @ Vt

            # Compute translation
            t = center1 - R @ center2

            # Build transformation
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t

            if self.verbose:
                print(f"  Simple transform: R det={np.linalg.det(R):.3f}, t={t}")

            return T

        except Exception as e:
            print(f"  Simple transform failed: {e}")
            return np.eye(4)


def estimate_transform_icp(source_points, target_points, initial_transform=None, verbose=False):
    """使用ICP进行点云配准"""
    try:
        # 转换为Open3D点云
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_points)

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)

        if verbose:
            print(f"  Source points: {len(source_points)}, Target points: {len(target_points)}")

        # 下采样以加速配准
        voxel_size = 0.01  # 1cm体素
        source_down = source_pcd.voxel_down_sample(voxel_size)
        target_down = target_pcd.voxel_down_sample(voxel_size)

        if verbose:
            print(f"  After downsampling: Source: {len(source_down.points)}, Target: {len(target_down.points)}")

        # 如果点太少，跳过配准
        if len(source_down.points) < 10 or len(target_down.points) < 10:
            if verbose:
                print(f"  Not enough points for ICP after downsampling")
            return np.eye(4), 0.0, float('inf')

        # 估计法向量
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

        # ICP配准
        if initial_transform is None:
            initial_transform = np.eye(4)

        # 使用点对平面ICP
        reg_result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, max_correspondence_distance=0.1,
            init=initial_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=100,
                relative_fitness=1e-6,
                relative_rmse=1e-6
            )
        )

        if verbose:
            print(f"  ICP fitness: {reg_result.fitness:.3f}, RMSE: {reg_result.inlier_rmse:.4f}")

        return reg_result.transformation, reg_result.fitness, reg_result.inlier_rmse

    except Exception as e:
        if verbose:
            print(f"  ICP registration failed: {e}")
        return np.eye(4), 0.0, float('inf')


def robust_registration(points1_list, points2_list, num_samples=5, verbose=True):
    """使用多帧进行鲁棒配准"""
    all_transforms = []
    all_fitness_scores = []

    # 从多帧中采样进行配准
    sample_indices = np.linspace(0, len(points1_list) - 1, num_samples, dtype=int)

    for idx in sample_indices:
        if len(points1_list[idx]) < 100 or len(points2_list[idx]) < 100:
            if verbose:
                print(f"  Frame {idx}: Not enough points for registration")
            continue

        if verbose:
            print(f"  Registration with frame {idx} ({len(points1_list[idx])} vs {len(points2_list[idx])} points)")

        # 首先使用简单SVD进行初始配准
        registrar = SimpleRegistration(verbose=False)
        initial_T = registrar.compute_simple_transform(
            points1_list[idx], points2_list[idx]
        )

        # 使用ICP进行精细配准
        T_icp, fitness, rmse = estimate_transform_icp(
            points2_list[idx], points1_list[idx], initial_T, verbose=verbose
        )

        if fitness > 0.1:  # 只接受合理的配准结果
            all_transforms.append(T_icp)
            all_fitness_scores.append(fitness)
            if verbose:
                print(f"    Fitness: {fitness:.3f}, RMSE: {rmse:.4f}")

    if len(all_transforms) == 0:
        if verbose:
            print("  No valid registration found, using identity transform")
        return np.eye(4)

    # 选择最佳配准结果
    best_idx = np.argmax(all_fitness_scores)
    best_transform = all_transforms[best_idx]

    if verbose:
        print(f"\n✓ Best registration from frame {sample_indices[best_idx]}:")
        print(f"  Fitness: {all_fitness_scores[best_idx]:.3f}")

        # 计算旋转角度
        R = best_transform[:3, :3]
        t = best_transform[:3, 3]

        # 提取欧拉角
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0

        rotation_deg = np.array([rx, ry, rz]) * 180 / np.pi

        print(f"  Rotation: [{rotation_deg[0]:.1f}, {rotation_deg[1]:.1f}, {rotation_deg[2]:.1f}]°")
        print(f"  Translation: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")

    return best_transform


def rotation_matrix_to_quaternion(R):
    """将旋转矩阵转换为四元数 (xyzw格式)"""
    trace = np.trace(R)

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return np.array([qx, qy, qz, qw])


def main(
        view1_rgb_dir: Path,
        view2_rgb_dir: Path,
        view1_mask_dir: Path = None,
        view2_mask_dir: Path = None,
        model_path: Path = None,
        calib_xml_path: Path = None,
        point_size: float = 0.003,
        verbose: bool = False,
        max_frames: int = 20,
        stride: int = 1,
        min_brightness: float = 0.3,
        min_depth: float = 0.01,
        max_depth: float = 2.0,
        use_registration: bool = True,
        registration_samples: int = 5,
        use_xml_extrinsics: bool = False,
):
    """
    Main function for registration visualization with point cloud registration.
    显示三个点云：
    1. View1点云（在View1坐标系）
    2. View2点云（在View2坐标系）
    3. View2点云（变换到View1坐标系）
    """
    # Initialize server
    server = viser.ViserServer()
    server.scene.set_up_direction('-z')

    print("=" * 80)
    print("POINT CLOUD REGISTRATION VISUALIZATION - THREE POINT CLOUDS")
    print("=" * 80)

    # Load camera calibration
    print(f"\nLoading calibration: {calib_xml_path}")
    calib_data = parse_calibration_xml(calib_xml_path, ignore_extrinsics=not use_xml_extrinsics)

    if calib_data is None:
        print("✗ Failed to load calibration")
        return

    K = calib_data['K']

    print(f"\n✓ Camera intrinsics:")
    print(f"  fx={K[0, 0]:.2f}, fy={K[1, 1]:.2f}")
    print(f"  cx={K[0, 2]:.2f}, cy={K[1, 2]:.2f}")

    # 如果有XML中的外参，使用它
    if use_xml_extrinsics and 'extrinsics' in calib_data:
        T_view2_to_view1 = calib_data['extrinsics']
        print(f"\n✓ Using extrinsics from XML file")
        print(f"  Transformation matrix loaded from XML")
    else:
        # 初始化为单位矩阵，将通过配准估计
        T_view2_to_view1 = np.eye(4)
        print(f"\n✓ Will estimate extrinsics through point cloud registration")

    # Load depth model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nLoading depth model: {model_path}")
    depth_model = load_depth_model(model_path, device)
    if depth_model is None:
        return

    # Find image files
    def find_images(img_dir, mask_dir=None):
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images.extend(sorted(glob.glob(str(img_dir / ext))))

        matched = []
        for img_path in images:
            img_path = Path(img_path)
            mask_path = None
            if mask_dir and mask_dir.exists():
                stem = img_path.stem
                for pattern in [f"{stem}.png", f"{stem}.jpg", f"{stem}_mask.png"]:
                    potential = mask_dir / pattern
                    if potential.exists():
                        mask_path = potential
                        break

            matched.append({
                'rgb': img_path,
                'mask': mask_path
            })

        return matched

    view1_images = find_images(view1_rgb_dir, view1_mask_dir)[:max_frames:stride]
    view2_images = find_images(view2_rgb_dir, view2_mask_dir)[:max_frames:stride]

    total_frames = min(len(view1_images), len(view2_images))
    print(f"\nProcessing {total_frames} frame pairs")

    # Process frames - 生成相机坐标系下的点云
    view1_points_cam = []  # View1相机坐标系下的点云
    view1_colors = []
    view2_points_cam = []  # View2相机坐标系下的点云
    view2_colors = []

    print("\n" + "=" * 50)
    print("GENERATING POINT CLOUDS IN CAMERA COORDINATES")
    print("=" * 50)

    for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
        # Process View1 - 在View1相机坐标系下
        view1_data = view1_images[frame_idx]
        rgb1 = cv2.imread(str(view1_data['rgb']))
        if rgb1 is None:
            print(f"✗ Could not load View1 image: {view1_data['rgb']}")
            continue
        rgb1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2RGB)

        # 估计View1的深度
        depth1 = estimate_depth_with_model(
            depth_model, view1_data['rgb'], np.eye(4), K, device,
            min_depth=min_depth, max_depth=max_depth
        )

        # 生成点云（View1相机坐标系）
        points1_cam, colors1 = depth_to_point_cloud(
            depth1, rgb1, K, None,  # 不应用变换，保持相机坐标系
            downsample_factor=2, max_points=20000,
            min_brightness=min_brightness, min_depth=min_depth,
            apply_filter=True
        )

        view1_points_cam.append(points1_cam)  # View1坐标系
        view1_colors.append(colors1)

        # Process View2 - 在View2相机坐标系下
        view2_data = view2_images[frame_idx]
        rgb2 = cv2.imread(str(view2_data['rgb']))
        if rgb2 is None:
            print(f"✗ Could not load View2 image: {view2_data['rgb']}")
            continue
        rgb2 = cv2.cvtColor(rgb2, cv2.COLOR_BGR2RGB)

        # 估计View2的深度
        depth2 = estimate_depth_with_model(
            depth_model, view2_data['rgb'], np.eye(4), K, device,
            min_depth=min_depth, max_depth=max_depth
        )

        # 生成点云（View2相机坐标系）
        points2_cam, colors2 = depth_to_point_cloud(
            depth2, rgb2, K, None,  # 不应用变换，保持相机坐标系
            downsample_factor=2, max_points=20000,
            min_brightness=min_brightness, min_depth=min_depth,
            apply_filter=True
        )

        view2_points_cam.append(points2_cam)  # View2坐标系
        view2_colors.append(colors2)

        if verbose and frame_idx < 3:
            print(f"\nFrame {frame_idx}:")
            print(f"  View1 points (camera coords): {len(points1_cam)}")
            print(f"  View2 points (camera coords): {len(points2_cam)}")

    print(f"\n✓ Generated point clouds for all {total_frames} frames")

    # 点云配准来估计变换矩阵（如果不使用XML中的外参）
    if use_registration and not use_xml_extrinsics:
        print("\n" + "=" * 50)
        print("POINT CLOUD REGISTRATION")
        print("=" * 50)

        # 使用多帧进行配准，估计 View2 -> View1 的变换
        T_view2_to_view1 = robust_registration(
            view1_points_cam,  # View1坐标系下的点云
            view2_points_cam,  # View2坐标系下的点云
            num_samples=min(registration_samples, total_frames),
            verbose=True
        )

        print(f"\n✓ Estimated transformation matrix (View2 -> View1):")
        for i in range(4):
            print(f"    [{T_view2_to_view1[i, 0]:8.4f} {T_view2_to_view1[i, 1]:8.4f} "
                  f"{T_view2_to_view1[i, 2]:8.4f} {T_view2_to_view1[i, 3]:8.4f}]")
    elif use_xml_extrinsics:
        print(f"\n✓ Using extrinsics from XML, skipping registration")
    else:
        print(f"\n✓ Registration disabled, using identity transform")

    # 准备三个点云用于可视化
    print("\n" + "=" * 50)
    print("PREPARING THREE POINT CLOUDS FOR VISUALIZATION")
    print("=" * 50)

    # 1. View1点云（在View1坐标系原点）
    view1_points = view1_points_cam  # 直接在View1坐标系

    # 2. View2点云（在View2坐标系，需要计算View2相机的位置）
    # T_view1_to_view2 是 View1 -> View2 的变换（T_view2_to_view1的逆）
    T_view1_to_view2 = np.linalg.inv(T_view2_to_view1)

    # 将View2点云放在View2相机坐标系位置
    view2_points_in_view2_coords = []
    for frame_idx in range(total_frames):
        points2_cam = view2_points_cam[frame_idx]
        if len(points2_cam) > 0:
            # View2点云在View2坐标系，需要变换到世界坐标系显示
            # View2点云在View2相机坐标系的原点
            # 所以它们相对于世界坐标系的位置就是View2相机的位置
            points2_cam_h = np.hstack([points2_cam, np.ones((len(points2_cam), 1))])
            points2_in_world = (T_view1_to_view2 @ points2_cam_h.T).T[:, :3]
        else:
            points2_in_world = np.zeros((0, 3))
        view2_points_in_view2_coords.append(points2_in_world)

    # 3. View2点云（变换到View1坐标系）
    view2_points_transformed = []
    for frame_idx in range(total_frames):
        points2_cam = view2_points_cam[frame_idx]
        if len(points2_cam) > 0:
            points2_cam_h = np.hstack([points2_cam, np.ones((len(points2_cam), 1))])
            points2_transformed = (T_view2_to_view1 @ points2_cam_h.T).T[:, :3]
        else:
            points2_transformed = np.zeros((0, 3))
        view2_points_transformed.append(points2_transformed)

    print(f"✓ Prepared three point clouds for visualization")

    # 创建可视化界面
    with server.gui.add_folder("Camera Information"):
        server.gui.add_markdown("**Camera Configuration:**")
        server.gui.add_markdown(f"- View1 Camera: At origin (blue camera frame)")
        server.gui.add_markdown(f"- View2 Camera: At estimated position (orange camera frame)")
        server.gui.add_markdown(f"- Total frames: {total_frames}")

        if use_xml_extrinsics:
            server.gui.add_markdown(f"- **Using extrinsics from XML file**")
        else:
            server.gui.add_markdown(f"- **Using point cloud registration**")
            server.gui.add_markdown(f"- Registration samples: {registration_samples}")

        t = T_view2_to_view1[:3, 3]
        view2_camera_position = T_view1_to_view2[:3, 3]

        server.gui.add_markdown(f"**Transformation Details:**")
        server.gui.add_markdown(f"- View2 -> View1 Translation: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
        server.gui.add_markdown(f"- View2 Camera Position: [{view2_camera_position[0]:.3f}, "
                                f"{view2_camera_position[1]:.3f}, {view2_camera_position[2]:.3f}] m")

    with server.gui.add_folder("Point Cloud Controls"):
        gui_current_frame = server.gui.add_slider(
            "Current Frame",
            min=0,
            max=total_frames - 1,
            step=1,
            initial_value=0
        )

        gui_show_view1 = server.gui.add_checkbox("Show View1 Point Cloud", True)
        gui_show_view2_original = server.gui.add_checkbox("Show View2 Original (at View2 camera)", True)
        gui_show_view2_transformed = server.gui.add_checkbox("Show View2 Transformed (to View1)", True)

        gui_show_view1_camera = server.gui.add_checkbox("Show View1 Camera Frame", True)
        gui_show_view2_camera = server.gui.add_checkbox("Show View2 Camera Frame", True)

        with server.gui.add_folder("Animation"):
            gui_play_animation = server.gui.add_button("Play Animation")
            gui_stop_animation = server.gui.add_button("Stop Animation")
            gui_animation_speed = server.gui.add_slider(
                "Speed (FPS)",
                min=1,
                max=30,
                step=1,
                initial_value=10
            )
            gui_loop = server.gui.add_checkbox("Loop", True)

        gui_point_size = server.gui.add_slider(
            "Point Size",
            min=0.001,
            max=0.01,
            step=0.0005,
            initial_value=point_size
        )

        gui_color_intensity = server.gui.add_slider(
            "Color Intensity",
            min=0.5,
            max=2.0,
            step=0.1,
            initial_value=1.0
        )

    with server.gui.add_folder("Frame Info"):
        frame_info = server.gui.add_markdown(f"**Frame 0:** "
                                             f"View1: {len(view1_points[0])} pts, "
                                             f"View2 Original: {len(view2_points_in_view2_coords[0])} pts, "
                                             f"View2 Transformed: {len(view2_points_transformed[0])} pts")

    # 创建相机坐标系
    # View1相机（在原点）
    view1_camera_frame = server.scene.add_frame(
        "/cameras/view1",
        wxyz=tf.SO3.exp(np.array([0.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        axes_length=0.1,
        axes_radius=0.003,
    )
    view1_camera_frame.visible = True

    # View2相机（实际位置）
    R_view2_in_world = T_view1_to_view2[:3, :3]
    t_view2_in_world = T_view1_to_view2[:3, 3]
    quaternion_xyzw = rotation_matrix_to_quaternion(R_view2_in_world)

    view2_camera_frame = server.scene.add_frame(
        "/cameras/view2",
        wxyz=tf.SO3.from_quaternion_xyzw(quaternion_xyzw).wxyz,
        position=tuple(t_view2_in_world),
        axes_length=0.1,
        axes_radius=0.003,
    )
    view2_camera_frame.visible = True

    # 颜色方案
    view1_color = [0.1, 0.6, 0.9]  # 蓝色 - View1点云
    view2_original_color = [0.9, 0.1, 0.1]  # 红色 - View2原始点云（在View2坐标系）
    view2_transformed_color = [0.9, 0.5, 0.1]  # 橙色 - View2变换到View1坐标系的点云

    # 为每一帧创建三个点云节点
    pc_view1_frames = []
    pc_view2_original_frames = []
    pc_view2_transformed_frames = []

    print(f"\nCreating {total_frames} frames with three point clouds each...")

    for frame_idx in range(total_frames):
        # 1. View1点云（蓝色，在View1坐标系原点）
        if len(view1_points[frame_idx]) > 0:
            pc1 = server.scene.add_point_cloud(
                name=f"/view1/frame_{frame_idx}",
                points=view1_points[frame_idx],
                colors=view1_colors[frame_idx],
                point_size=point_size,
                point_shape="rounded",
                visible=(frame_idx == 0)
            )
            pc_view1_frames.append(pc1)
        else:
            pc_view1_frames.append(None)

        # 2. View2原始点云（红色，在View2相机坐标系位置）
        if len(view2_points_in_view2_coords[frame_idx]) > 0:
            pc2_orig = server.scene.add_point_cloud(
                name=f"/view2_original/frame_{frame_idx}",
                points=view2_points_in_view2_coords[frame_idx],
                colors=view2_colors[frame_idx],
                point_size=point_size,
                point_shape="rounded",
                visible=(frame_idx == 0)
            )
            pc_view2_original_frames.append(pc2_orig)
        else:
            pc_view2_original_frames.append(None)

        # 3. View2变换后的点云（橙色，变换到View1坐标系）
        if len(view2_points_transformed[frame_idx]) > 0:
            pc2_trans = server.scene.add_point_cloud(
                name=f"/view2_transformed/frame_{frame_idx}",
                points=view2_points_transformed[frame_idx],
                colors=view2_colors[frame_idx],
                point_size=point_size,
                point_shape="rounded",
                visible=(frame_idx == 0)
            )
            pc_view2_transformed_frames.append(pc2_trans)
        else:
            pc_view2_transformed_frames.append(None)

    print(f"✓ Created {total_frames} frames with three point clouds each")

    # 动画状态
    animation_playing = False
    current_frame = 0

    def update_colors():
        """更新点云颜色基于强度"""
        intensity = gui_color_intensity.value

        if pc_view1_frames[current_frame]:
            colors1 = view1_colors[current_frame] * np.array(view1_color) * intensity
            colors1 = np.clip(colors1, 0, 1)
            pc_view1_frames[current_frame].colors = colors1

        if pc_view2_original_frames[current_frame]:
            colors2_orig = view2_colors[current_frame] * np.array(view2_original_color) * intensity
            colors2_orig = np.clip(colors2_orig, 0, 1)
            pc_view2_original_frames[current_frame].colors = colors2_orig

        if pc_view2_transformed_frames[current_frame]:
            colors2_trans = view2_colors[current_frame] * np.array(view2_transformed_color) * intensity
            colors2_trans = np.clip(colors2_trans, 0, 1)
            pc_view2_transformed_frames[current_frame].colors = colors2_trans

    def show_frame(frame_idx):
        """显示特定帧的点云"""
        nonlocal current_frame
        current_frame = frame_idx

        # 隐藏所有帧
        for pc in pc_view1_frames:
            if pc:
                pc.visible = False
        for pc in pc_view2_original_frames:
            if pc:
                pc.visible = False
        for pc in pc_view2_transformed_frames:
            if pc:
                pc.visible = False

        # 显示当前帧
        if pc_view1_frames[frame_idx]:
            pc_view1_frames[frame_idx].visible = gui_show_view1.value

        if pc_view2_original_frames[frame_idx]:
            pc_view2_original_frames[frame_idx].visible = gui_show_view2_original.value

        if pc_view2_transformed_frames[frame_idx]:
            pc_view2_transformed_frames[frame_idx].visible = gui_show_view2_transformed.value

        # 更新颜色
        update_colors()

        # 更新帧信息
        frame_info.content = (f"**Frame {frame_idx}:** "
                              f"View1: {len(view1_points[frame_idx])} pts, "
                              f"View2 Original: {len(view2_points_in_view2_coords[frame_idx])} pts, "
                              f"View2 Transformed: {len(view2_points_transformed[frame_idx])} pts")

    def animate():
        """动画循环"""
        nonlocal animation_playing, current_frame

        while animation_playing:
            current_frame += 1
            if current_frame >= total_frames:
                if gui_loop.value:
                    current_frame = 0
                else:
                    animation_playing = False
                    break

            gui_current_frame.value = current_frame
            show_frame(current_frame)

            time.sleep(1.0 / gui_animation_speed.value)

    # 事件处理器
    @gui_current_frame.on_update
    def _(_):
        show_frame(int(gui_current_frame.value))

    @gui_show_view1.on_update
    def _(_):
        if pc_view1_frames[current_frame]:
            pc_view1_frames[current_frame].visible = gui_show_view1.value

    @gui_show_view2_original.on_update
    def _(_):
        if pc_view2_original_frames[current_frame]:
            pc_view2_original_frames[current_frame].visible = gui_show_view2_original.value

    @gui_show_view2_transformed.on_update
    def _(_):
        if pc_view2_transformed_frames[current_frame]:
            pc_view2_transformed_frames[current_frame].visible = gui_show_view2_transformed.value

    @gui_show_view1_camera.on_update
    def _(_):
        view1_camera_frame.visible = gui_show_view1_camera.value

    @gui_show_view2_camera.on_update
    def _(_):
        view2_camera_frame.visible = gui_show_view2_camera.value

    @gui_point_size.on_update
    def _(_):
        size = gui_point_size.value
        for pc in pc_view1_frames:
            if pc:
                pc.point_size = size
        for pc in pc_view2_original_frames:
            if pc:
                pc.point_size = size
        for pc in pc_view2_transformed_frames:
            if pc:
                pc.point_size = size

    @gui_color_intensity.on_update
    def _(_):
        update_colors()

    @gui_play_animation.on_click
    def _(_):
        nonlocal animation_playing
        if not animation_playing:
            animation_playing = True
            import threading
            threading.Thread(target=animate, daemon=True).start()

    @gui_stop_animation.on_click
    def _(_):
        nonlocal animation_playing
        animation_playing = False

    # 显示第一帧
    show_frame(0)
    update_colors()

    print(f"\n{'=' * 80}")
    print("VISUALIZATION READY")
    print(f"Open: http://localhost:8080")
    print(f"{'=' * 80}")

    print(f"\nTHREE POINT CLOUDS VISUALIZATION:")
    print(f"1. BLUE: View1 point cloud (at View1 camera origin)")
    print(f"2. RED: View2 original point cloud (at View2 camera position)")
    print(f"3. ORANGE: View2 transformed point cloud (aligned to View1 coordinates)")
    print(f"\nCamera Frames:")
    print(f"- Blue camera: View1 at origin")
    print(f"- Orange camera: View2 at estimated position")

    print(f"\nTRANSFORMATION INFO:")
    print(f"- View2 -> View1: Translation = [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
    print(
        f"- View2 Camera Position: [{view2_camera_position[0]:.3f}, {view2_camera_position[1]:.3f}, {view2_camera_position[2]:.3f}] m")

    print(f"\nEXPECTED RESULT:")
    print(f"- Blue and orange point clouds should align (if registration is accurate)")
    print(f"- Red point cloud should be at View2 camera position")
    print(f"- Blue and orange point clouds should be at the origin")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped by user")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Registration with three point clouds visualization")

    parser.add_argument("--view1-rgb-dir", type=Path, required=True)
    parser.add_argument("--view2-rgb-dir", type=Path, required=True)
    parser.add_argument("--view1-mask-dir", type=Path)
    parser.add_argument("--view2-mask-dir", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--point-size", type=float, default=0.003)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-brightness", type=float, default=0.3)
    parser.add_argument("--min-depth", type=float, default=0.01)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--use-registration", action="store_true", default=True,
                        help="Use point cloud registration to estimate extrinsics")
    parser.add_argument("--registration-samples", type=int, default=5,
                        help="Number of frames to use for registration")
    parser.add_argument("--use-xml-extrinsics", action="store_true", default=False,
                        help="Use extrinsics from XML file instead of registration")

    args = parser.parse_args()

    main(
        view1_rgb_dir=args.view1_rgb_dir,
        view2_rgb_dir=args.view2_rgb_dir,
        view1_mask_dir=args.view1_mask_dir,
        view2_mask_dir=args.view2_mask_dir,
        model_path=args.model,
        calib_xml_path=args.calib,
        point_size=args.point_size,
        verbose=args.verbose,
        max_frames=args.max_frames,
        stride=args.stride,
        min_brightness=args.min_brightness,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        use_registration=args.use_registration,
        registration_samples=args.registration_samples,
        use_xml_extrinsics=args.use_xml_extrinsics,
    )