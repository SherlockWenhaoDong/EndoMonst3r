import sys

sys.path.append('.')
import os
import torch
import numpy as np
import os.path as osp
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import glob
import cv2
import json
import re
from torchvision.transforms import ColorJitter, GaussianBlur
from functools import partial

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from dust3r.utils.misc import get_stride_distribution

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')


class SCAREDDUSt3R(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='data/scared',
                 dset='train',
                 use_augs=False,
                 S=2,
                 N=16,
                 strides=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                 clip_step=2,
                 quick=False,
                 verbose=False,
                 dist_type=None,
                 clip_step_last_skip=0,
                 fixed_pose=False,  # SCARED数据集有真实位姿，所以不使用固定位姿
                 *args,
                 **kwargs):

        print('loading SCARED dataset...')
        super().__init__(*args, **kwargs)
        self.dataset_label = 'scared'
        self.split = dset
        self.S = S  # stride
        self.N = N  # min num points
        self.verbose = verbose
        self.use_augs = use_augs
        self.dset = dset
        self.fixed_pose = fixed_pose

        # 存储路径
        self.rgb_paths = []
        self.depth_paths = []
        self.pose_paths = []
        self.calib_paths = []
        self.full_idxs = []
        self.sample_stride = []
        self.strides = strides

        self.sequences = []

        # SCARED数据集结构
        if os.path.isdir(dataset_location):
            # 查找所有序列文件夹
            for seq in glob.glob(os.path.join(dataset_location, "*")):
                if os.path.isdir(seq):
                    seq_name = os.path.basename(seq)
                    # 检查是否是有效序列（如1-1, 1-2, 1-3等）
                    if re.match(r'^\d+-\d+$', seq_name):
                        self.sequences.append({
                            'path': seq,
                            'name': seq_name
                        })
        else:
            raise ValueError(f"Dataset location {dataset_location} does not exist")

        self.sequences = sorted(self.sequences, key=lambda x: x['name'])
        if self.verbose:
            print(f"Found sequences: {[s['name'] for s in self.sequences]}")

        print(f'found {len(self.sequences)} unique sequences in {dataset_location} (dset={dset})')

        # 加载所有图像对
        self._load_image_pairs(strides, clip_step, clip_step_last_skip, quick)

        # 如果需要重新采样
        if len(strides) > 1 and dist_type is not None:
            self._resample_clips(strides, dist_type)

        print(f'collected {len(self.rgb_paths)} clips of length {self.S} in {dataset_location} (dset={dset})')

    def _parse_pose_json(self, json_path):
        """
        解析SCARED数据集的位姿JSON文件

        JSON结构示例：
        {
            "camera-calibration": {
                "DL": [[-0.0005951574421487749], ...],
                "DR": [[-0.00023428065469488502], ...],
                "KL": [[1035.30810546875, 0.0, 596.9550170898438], ...],
                "KR": [[1035.1741943359375, 0.0, 688.3618774414062], ...],
                "R": [[1.0, 1.9485649318085052e-05, -0.00015232479199767113], ...],
                "T": [[-4.14339017868042], [-0.023819703608751297], [-0.0019068525871261954]]
            },
            "camera-pose": [
                [0.9999965796048702, -0.0025485094056294315, -0.0002287824528534348, -0.02928418649968023],
                [0.002548294731588833, 0.9999962957559811, -0.0008537852186089054, -0.10912194375544004],
                [0.00023095502338880993, 0.0008533662118120291, 0.9999997628384426, 0.3536461362789396],
                [0.0, 0.0, 0.0, 1.0]
            ],
            "timestamp": 1558660318896583
        }
        """
        if not os.path.exists(json_path):
            print(f"Warning: Pose JSON file not found: {json_path}")
            return None, None

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # 提取相机内参（使用左相机KL）
            if 'camera-calibration' in data and 'KL' in data['camera-calibration']:
                K_data = data['camera-calibration']['KL']
                # KL是一个3x3矩阵
                K = np.array([
                    [K_data[0][0], K_data[0][1], K_data[0][2]],
                    [K_data[1][0], K_data[1][1], K_data[1][2]],
                    [K_data[2][0], K_data[2][1], K_data[2][2]]
                ], dtype=np.float32)
            else:
                # 如果没有内参，使用默认值
                K = np.eye(3, dtype=np.float32)

            # 提取相机位姿（4x4矩阵）
            if 'camera-pose' in data:
                pose_data = data['camera-pose']
                pose = np.array(pose_data, dtype=np.float32)
            else:
                # 如果没有位姿，使用单位矩阵
                pose = np.eye(4, dtype=np.float32)

            # 提取时间戳（可选）
            timestamp = data.get('timestamp', 0)

            return K, pose

        except Exception as e:
            print(f"Error parsing pose JSON {json_path}: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _load_npz_depth(self, npz_path):
        """
        加载NPZ格式的深度图
        """
        try:
            data = np.load(npz_path)
            # NPZ文件通常包含一个名为'arr_0'的数组
            if 'arr_0' in data:
                depth = data['arr_0']
            else:
                # 或者使用第一个键
                depth = data[data.files[0]]

            # 深度图通常是单通道的
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]

            return depth.astype(np.float32)

        except Exception as e:
            print(f"Error loading NPZ depth {npz_path}: {e}")
            return None

    def _load_image_pairs(self, strides, clip_step, clip_step_last_skip, quick):
        """加载所有图像对"""
        for seq_info in self.sequences:
            seq_path = seq_info['path']
            seq_name = seq_info['name']

            if self.verbose:
                print(f'Processing sequence: {seq_name}')

            # SCARED数据集结构：image/input文件夹
            input_dir = os.path.join(seq_path, 'image', 'input')
            if not os.path.exists(input_dir):
                if self.verbose:
                    print(f'  Skipping sequence {seq_name}: input directory not found')
                continue

            # 获取所有RGB图像
            rgb_extensions = ['*.png', '*.jpg', '*.jpeg']
            rgb_files = []
            for ext in rgb_extensions:
                rgb_files.extend(glob.glob(os.path.join(input_dir, ext)))

            # 按文件名排序（提取帧号）
            def extract_frame_number(filename):
                # 文件名格式如: 1_1_frame_data000036.png
                basename = os.path.basename(filename)
                # 提取最后6位数字作为帧号
                match = re.search(r'frame_data(\d{6})', basename)
                if match:
                    return int(match.group(1))
                return 0

            rgb_files = sorted(rgb_files, key=extract_frame_number)

            if len(rgb_files) < self.S * max(strides) + 1:
                if self.verbose:
                    print(f'  Skipping sequence {seq_name}: not enough frames ({len(rgb_files)} frames)')
                continue

            print(f'  Found {len(rgb_files)} frames in {seq_name}')

            # 为每个stride创建图像对
            for stride in strides:
                # 计算可能的起始位置
                max_start = len(rgb_files) - self.S * stride
                step = max(clip_step, 1)

                start_indices = list(range(0, max_start + 1, step))
                if quick and len(start_indices) > 20:
                    # 快速模式下只取前20个
                    start_indices = start_indices[:20]

                for start_idx in start_indices:
                    full_idx = [start_idx + i * stride for i in range(self.S)]

                    # 收集RGB图像路径
                    rgb_pair = []
                    depth_pair = []
                    pose_pair = []
                    calib_pair = []
                    valid_pair = True

                    for idx in full_idx:
                        if idx < len(rgb_files):
                            rgb_path = rgb_files[idx]
                            rgb_pair.append(rgb_path)

                            # 提取帧信息
                            rgb_filename = os.path.basename(rgb_path)
                            # 格式: 1_1_frame_data000036.png
                            # 提取序列ID和帧号
                            parts = rgb_filename.split('_')
                            if len(parts) >= 4:
                                seq_id = f"{parts[0]}_{parts[1]}"  # 如 "1_1"
                                frame_match = re.search(r'frame_data(\d{6})', rgb_filename)
                                if frame_match:
                                    frame_num = frame_match.group(1)

                                    # 构建深度图路径
                                    depth_dir = os.path.join(seq_path, 'monodep')
                                    depth_filename = f"depth_{seq_id}_frame_data{frame_num}.npz"
                                    depth_path = os.path.join(depth_dir, depth_filename)

                                    # 构建位姿文件路径
                                    pose_dir = os.path.join(seq_path, 'poses', f"{parts[0]}-{parts[1]}")
                                    pose_filename = f"frame_data{frame_num}.json"
                                    pose_path = os.path.join(pose_dir, pose_filename)

                                    depth_pair.append(depth_path if os.path.exists(depth_path) else None)
                                    pose_pair.append(pose_path if os.path.exists(pose_path) else None)
                                    calib_pair.append(pose_path if os.path.exists(pose_path) else None)
                                else:
                                    valid_pair = False
                                    break
                            else:
                                valid_pair = False
                                break
                        else:
                            valid_pair = False
                            break

                    if not valid_pair:
                        continue

                    # 检查所有必要文件是否存在
                    all_exist = True
                    for i in range(len(rgb_pair)):
                        if not os.path.exists(rgb_pair[i]):
                            all_exist = False
                            break
                        # 深度图和位姿文件是可选的

                    if not all_exist:
                        continue

                    self.rgb_paths.append(rgb_pair)
                    self.depth_paths.append(depth_pair)
                    self.pose_paths.append(pose_pair)
                    self.calib_paths.append(calib_pair)
                    self.full_idxs.append(full_idx)
                    self.sample_stride.append(stride)

                    if self.verbose and len(self.rgb_paths) % 100 == 0:
                        sys.stdout.write('.')
                        sys.stdout.flush()

            if quick and len(self.rgb_paths) > 100:
                print(f'  Quick mode: limiting to 100 samples')
                break

    def _resample_clips(self, strides, dist_type):
        """重新采样图像对"""
        dist = get_stride_distribution(strides, dist_type=dist_type)
        dist = dist / np.max(dist)

        # 计算每个stride的数量
        stride_counts = {}
        for i, stride in enumerate(strides):
            stride_counts[stride] = sum(1 for s in self.sample_stride if s == stride)

        max_num_clips = stride_counts[strides[np.argmax(dist)]]
        num_clips_each_stride = [
            min(stride_counts[stride], int(dist[i] * max_num_clips))
            for i, stride in enumerate(strides)
        ]

        print('resampled_num_clips_each_stride:', num_clips_each_stride)

        # 收集每个stride的索引
        stride_idxs = {stride: [] for stride in strides}
        for i, stride in enumerate(self.sample_stride):
            stride_idxs[stride].append(i)

        resampled_idxs = []
        for i, stride in enumerate(strides):
            available_idxs = stride_idxs[stride]
            if len(available_idxs) > 0:
                n_samples = min(num_clips_each_stride[i], len(available_idxs))
                selected = np.random.choice(available_idxs, n_samples, replace=False)
                resampled_idxs.extend(selected.tolist())

        # 更新所有列表
        self.rgb_paths = [self.rgb_paths[i] for i in resampled_idxs]
        self.depth_paths = [self.depth_paths[i] for i in resampled_idxs]
        self.pose_paths = [self.pose_paths[i] for i in resampled_idxs]
        self.calib_paths = [self.calib_paths[i] for i in resampled_idxs]
        self.full_idxs = [self.full_idxs[i] for i in resampled_idxs]
        self.sample_stride = [self.sample_stride[i] for i in resampled_idxs]

    def __len__(self):
        return len(self.rgb_paths)

    def _get_views(self, index, resolution, rng):
        """获取一对视图"""
        rgb_paths = self.rgb_paths[index]
        depth_paths = self.depth_paths[index]
        pose_paths = self.pose_paths[index]
        calib_paths = self.calib_paths[index]

        views = []

        for i in range(self.S):
            rgb_path = rgb_paths[i]
            depth_path = depth_paths[i] if i < len(depth_paths) else None
            pose_path = pose_paths[i] if i < len(pose_paths) else None
            calib_path = calib_paths[i] if i < len(calib_paths) else None

            # 加载RGB图像
            try:
                rgb_image = imread_cv2(rgb_path)
                if rgb_image is None:
                    print(f"Error: Cannot read image: {rgb_path}")
                    # 创建一个占位符图像
                    rgb_image = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
            except Exception as e:
                print(f"Error reading image {rgb_path}: {e}")
                rgb_image = np.ones((1080, 1920, 3), dtype=np.uint8) * 128

            # 检查图像尺寸
            if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
                print(f"Warning: Unexpected image shape {rgb_image.shape} for {rgb_path}")
                # 转换为3通道图像
                if len(rgb_image.shape) == 2:
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
                elif len(rgb_image.shape) == 3 and rgb_image.shape[2] == 4:
                    rgb_image = rgb_image[:, :, :3]  # 去掉alpha通道

            h, w = rgb_image.shape[:2]

            # 确保尺寸能被16整除
            h = h - (h % 16)
            w = w - (w % 16)
            rgb_image = rgb_image[:h, :w]

            # 加载相机内参和位姿
            if pose_path and os.path.exists(pose_path):
                K, camera_pose = self._parse_pose_json(pose_path)
                if K is None or camera_pose is None:
                    # 如果解析失败，使用默认值
                    K = np.array([
                        [0.5 * w, 0, 0.5 * w],
                        [0, 0.5 * h, 0.5 * h],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    camera_pose = np.eye(4, dtype=np.float32)
            else:
                # 如果没有位姿文件，使用默认值
                K = np.array([
                    [0.5 * w, 0, 0.5 * w],
                    [0, 0.5 * h, 0.5 * h],
                    [0, 0, 1]
                ], dtype=np.float32)
                camera_pose = np.eye(4, dtype=np.float32)

            # 调整内参以适应裁剪后的图像
            original_height, original_width = h, w
            # SCARED数据集的内参通常是针对原始图像的，需要调整
            if 'original_size' not in locals():
                # 假设原始图像尺寸与当前裁剪后相同
                scale_x = w / original_width if original_width > 0 else 1
                scale_y = h / original_height if original_height > 0 else 1
                K[0, 0] *= scale_x  # fx
                K[1, 1] *= scale_y  # fy
                K[0, 2] *= scale_x  # cx
                K[1, 2] *= scale_y  # cy

            # 使用固定位姿（如果设置）
            if self.fixed_pose:
                camera_pose = np.eye(4, dtype=np.float32)

            # 加载深度图（如果存在）
            depthmap = None
            if depth_path and os.path.exists(depth_path):
                try:
                    depthmap = self._load_npz_depth(depth_path)
                    if depthmap is not None:
                        # 裁剪深度图以匹配RGB图像
                        depthmap = depthmap[:h, :w]

                        # SCARED深度图通常是真实深度值（毫米或米）
                        # 这里假设是毫米，转换为米
                        depthmap = depthmap.astype(np.float32) / 1000.0

                        # 处理无效深度值
                        depthmap[depthmap <= 0] = 0
                except Exception as e:
                    print(f"Warning: Could not load depth map {depth_path}: {e}")

            # 如果需要，进一步调整到目标分辨率
            if resolution:
                rgb_image, depthmap, K = self._crop_resize_if_necessary(
                    rgb_image, depthmap, K, resolution, rng=rng, info=rgb_path)

                # 再次确保尺寸能被16整除
                h_final, w_final = rgb_image.shape[:2]
                if h_final % 16 != 0 or w_final % 16 != 0:
                    h_final = h_final - (h_final % 16)
                    w_final = w_final - (w_final % 16)
                    rgb_image = rgb_image[:h_final, :w_final]
                    if depthmap is not None:
                        depthmap = depthmap[:h_final, :w_final]

                    # 调整内参
                    scale_x = w_final / w if w > 0 else 1
                    scale_y = h_final / h if h > 0 else 1
                    K[0, 0] *= scale_x
                    K[1, 1] *= scale_y
                    K[0, 2] *= scale_x
                    K[1, 2] *= scale_y

            # 将NumPy数组转换为PIL图像
            pil_image = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))

            # 构建视图字典
            view_data = dict(
                img=pil_image,
                camera_pose=camera_pose,
                camera_intrinsics=K,
                dataset=self.dataset_label,
                label=os.path.basename(os.path.dirname(os.path.dirname(rgb_path))),
                instance=os.path.basename(rgb_path),
                original_size=(original_height, original_width)
            )

            if depthmap is not None:
                view_data['depthmap'] = depthmap

            views.append(view_data)

        return views

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng, info=""):
        """裁剪和调整大小"""
        # 如果image是PIL图像，先转换为NumPy数组
        if isinstance(image, Image.Image):
            # PIL图像转换为NumPy数组
            image_np = np.array(image)
            # PIL是RGB，OpenCV是BGR，所以需要转换
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            is_pil = True
        else:
            image_np = image
            is_pil = False

        h, w = image_np.shape[:2]

        if isinstance(resolution, (list, tuple)):
            # 随机选择分辨率
            if isinstance(resolution[0], (list, tuple)):
                # 多个分辨率选项
                chosen_res = resolution[rng.randint(0, len(resolution))]
                target_h, target_w = chosen_res
            else:
                # 单个分辨率
                target_h, target_w = resolution
        else:
            # 单一数值，保持宽高比
            target_h = resolution
            target_w = int(w * resolution / h)

        # 调整图像大小
        if (h, w) != (target_h, target_w):
            image_resized = cv2.resize(image_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            image_resized = image_np.copy()

        # 调整深度图大小
        depthmap_resized = None
        if depthmap is not None:
            if (h, w) != (target_h, target_w):
                depthmap_resized = cv2.resize(depthmap, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            else:
                depthmap_resized = depthmap.copy()

        # 调整内参
        scale_x = target_w / w
        scale_y = target_h / h

        intrinsics_resized = intrinsics.copy()
        intrinsics_resized[0, 0] *= scale_x  # fx
        intrinsics_resized[1, 1] *= scale_y  # fy
        intrinsics_resized[0, 2] *= scale_x  # cx
        intrinsics_resized[1, 2] *= scale_y  # cy

        # 如果需要返回PIL图像，则转换回来
        if is_pil:
            image_resized = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))

        return image_resized, depthmap_resized, intrinsics_resized


# 测试脚本
if __name__ == "__main__":
    # 测试数据集
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import random

    # 数据集路径 - 修改为您的实际路径
    dataset_location = '/path/to/scared/dataset'

    print("Testing SCARED dataset...")

    dataset = SCAREDDUSt3R(
        dataset_location=dataset_location,
        dset='train',
        use_augs=False,
        S=2,
        N=1,
        strides=[1, 2],
        clip_step=2,
        quick=True,
        verbose=True,
        resolution=[(512, 288)],
        aug_crop=16,
        dist_type=None,
        aug_focal=1.5,
        z_far=80,
        fixed_pose=False,  # SCARED有真实位姿，所以不使用固定位姿
    )

    print(f"Dataset size: {len(dataset)}")

    # 可视化一个样本
    if len(dataset) > 0:
        idx = random.randint(0, min(len(dataset) - 1, 10))
        print(f"\nVisualizing sample {idx}...")

        views = dataset[idx]
        print(f"Number of views: {len(views)}")

        for i, view in enumerate(views):
            print(f"\nView {i}:")
            print(f"  Image shape: {view['img'].shape}")
            print(f"  Camera pose:\n{view['camera_pose']}")
            print(f"  Camera intrinsics:\n{view['camera_intrinsics']}")

            if 'depthmap' in view:
                print(f"  Depth map shape: {view['depthmap'].shape}")
                if view['depthmap'] is not None:
                    print(f"  Depth range: {view['depthmap'].min():.2f} - {view['depthmap'].max():.2f}")

        # 保存样本图像以供检查
        import matplotlib.pyplot as plt

        os.makedirs('./test_output', exist_ok=True)

        fig, axes = plt.subplots(1, len(views), figsize=(10, 5))
        if len(views) == 1:
            axes = [axes]

        for i, view in enumerate(views):
            # 转换为numpy数组显示
            img_np = np.array(view['img'])
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            axes[i].imshow(img_np)
            axes[i].set_title(f'View {i}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig('./test_output/scared_sample.png')
        plt.show()
        print("Sample saved to ./test_output/scared_sample.png")