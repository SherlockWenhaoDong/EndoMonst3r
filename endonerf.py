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
import re
from torchvision.transforms import ColorJitter, GaussianBlur
from functools import partial

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from dust3r.utils.misc import get_stride_distribution

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')


class EndoNeRFDUSt3R(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='data/endonerf',
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
                 fixed_pose=True,
                 poses_bounds_file='poses_bounds.npy',
                 *args,
                 **kwargs):

        print('loading EndoNeRF dataset...')
        super().__init__(*args, **kwargs)
        self.dataset_label = 'endonerf'
        self.split = dset
        self.S = S  # stride
        self.N = N  # min num points
        self.verbose = verbose
        self.use_augs = use_augs
        self.dset = dset
        self.fixed_pose = fixed_pose
        self.poses_bounds_file = poses_bounds_file

        # 存储路径
        self.rgb_paths = []
        self.depth_paths = []
        self.mask_paths = []
        self.full_idxs = []
        self.sample_stride = []
        self.strides = strides

        # 存储相机位姿和内参
        self.poses = []
        self.intrinsics = []

        self.sequences = []

        # EndoNeRF数据集结构
        if os.path.isdir(dataset_location):
            # 检查是否是单个场景目录
            if os.path.exists(os.path.join(dataset_location, 'images')):
                self.sequences.append({
                    'path': dataset_location,
                    'name': os.path.basename(dataset_location)
                })
            else:
                # 查找所有场景文件夹
                for seq in glob.glob(os.path.join(dataset_location, "*")):
                    if os.path.isdir(seq) and os.path.exists(os.path.join(seq, 'images')):
                        seq_name = os.path.basename(seq)
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

    def _parse_poses_bounds(self, poses_bounds_path):
        """
        解析poses_bounds.npy文件

        格式: [N, 17] 数组
        前15个元素: 3x5的矩阵，其中:
          - 前3x3: 旋转矩阵
          - 后3x2: 平移向量和焦距(?)
        最后2个元素: 近平面和远平面
        """
        if not os.path.exists(poses_bounds_path):
            print(f"Warning: poses_bounds.npy not found: {poses_bounds_path}")
            return None, None

        try:
            data = np.load(poses_bounds_path)

            if len(data.shape) != 2 or data.shape[1] != 17:
                print(f"Warning: Unexpected shape of poses_bounds.npy: {data.shape}")
                return None, None

            num_frames = data.shape[0]
            poses = []
            intrinsics = []

            for i in range(num_frames):
                # 提取3x5矩阵
                pose_matrix = data[i, :15].reshape(3, 5)

                # 前3x3是旋转矩阵
                rotation = pose_matrix[:, :3]

                # 第4列是平移向量
                translation = pose_matrix[:, 3]

                # 第5列可能是焦距等信息
                focal_info = pose_matrix[:, 4]

                # 构建4x4位姿矩阵 (camera-to-world)
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = rotation
                pose[:3, 3] = translation
                poses.append(pose)

                # 估计内参 (通常从focal_info获取)
                # 假设焦距在focal_info的第一个元素
                fx = fy = abs(focal_info[0]) if abs(focal_info[0]) > 0 else 500.0

                # 使用默认图像中心 (后面会根据实际图像尺寸调整)
                cx = 320.0  # 假设图像宽度640
                cy = 240.0  # 假设图像高度480

                K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float32)
                intrinsics.append(K)

            return poses, intrinsics

        except Exception as e:
            print(f"Error parsing poses_bounds.npy {poses_bounds_path}: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _load_image_pairs(self, strides, clip_step, clip_step_last_skip, quick):
        """加载所有图像对"""
        for seq_info in self.sequences:
            seq_path = seq_info['path']
            seq_name = seq_info['name']

            if self.verbose:
                print(f'Processing sequence: {seq_name}')

            # EndoNeRF数据集结构
            images_dir = os.path.join(seq_path, 'images')
            depth_dir = os.path.join(seq_path, 'depth')
            masks_dir = os.path.join(seq_path, 'masks')
            poses_bounds_path = os.path.join(seq_path, self.poses_bounds_file)

            if not os.path.exists(images_dir):
                if self.verbose:
                    print(f'  Skipping sequence {seq_name}: images directory not found')
                continue

            # 获取所有RGB图像
            rgb_pattern = os.path.join(images_dir, 'frame-*.color.png')
            rgb_files = glob.glob(rgb_pattern)

            # 按帧号排序
            def extract_frame_number(filename):
                # 文件名格式如: frame-000001.color.png
                basename = os.path.basename(filename)
                match = re.search(r'frame-(\d{6})\.color\.png', basename)
                if match:
                    return int(match.group(1))
                return 0

            rgb_files = sorted(rgb_files, key=extract_frame_number)

            if len(rgb_files) < self.S * max(strides) + 1:
                if self.verbose:
                    print(f'  Skipping sequence {seq_name}: not enough frames ({len(rgb_files)} frames)')
                continue

            print(f'  Found {len(rgb_files)} frames in {seq_name}')

            # 加载位姿信息
            poses, intrinsics = None, None
            if os.path.exists(poses_bounds_path):
                poses, intrinsics = self._parse_poses_bounds(poses_bounds_path)
                if poses is None or intrinsics is None:
                    print(f"  Warning: Failed to load poses from {poses_bounds_path}")

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
                    mask_pair = []
                    pose_pair = []
                    intrinsic_pair = []
                    valid_pair = True

                    for idx in full_idx:
                        if idx < len(rgb_files):
                            rgb_path = rgb_files[idx]
                            rgb_pair.append(rgb_path)

                            # 提取帧号
                            rgb_filename = os.path.basename(rgb_path)
                            frame_match = re.search(r'frame-(\d{6})\.color\.png', rgb_filename)

                            if frame_match:
                                frame_num = frame_match.group(1)

                                # 构建深度图路径
                                depth_filename = f'frame-{frame_num}.depth.png'
                                depth_path = os.path.join(depth_dir, depth_filename) if os.path.exists(
                                    depth_dir) else None

                                # 构建mask路径
                                mask_filename = f'frame-{frame_num}.mask.png'
                                mask_path = os.path.join(masks_dir, mask_filename) if os.path.exists(
                                    masks_dir) else None

                                depth_pair.append(depth_path)
                                mask_pair.append(mask_path)

                                # 获取对应的位姿和内参
                                if poses is not None and idx < len(poses):
                                    pose_pair.append(poses[idx])
                                    intrinsic_pair.append(intrinsics[idx])
                                else:
                                    pose_pair.append(None)
                                    intrinsic_pair.append(None)
                            else:
                                valid_pair = False
                                break
                        else:
                            valid_pair = False
                            break

                    if not valid_pair:
                        continue

                    # 检查所有RGB文件是否存在
                    all_exist = True
                    for rgb_path in rgb_pair:
                        if not os.path.exists(rgb_path):
                            all_exist = False
                            break

                    if not all_exist:
                        continue

                    self.rgb_paths.append(rgb_pair)
                    self.depth_paths.append(depth_pair)
                    self.mask_paths.append(mask_pair)
                    self.full_idxs.append(full_idx)
                    self.sample_stride.append(stride)

                    # 存储位姿和内参
                    self.poses.append(pose_pair)
                    self.intrinsics.append(intrinsic_pair)

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
        self.mask_paths = [self.mask_paths[i] for i in resampled_idxs]
        self.full_idxs = [self.full_idxs[i] for i in resampled_idxs]
        self.sample_stride = [self.sample_stride[i] for i in resampled_idxs]
        self.poses = [self.poses[i] for i in resampled_idxs]
        self.intrinsics = [self.intrinsics[i] for i in resampled_idxs]

    def __len__(self):
        return len(self.rgb_paths)

    def _get_views(self, index, resolution, rng):
        """获取一对视图"""
        rgb_paths = self.rgb_paths[index]
        depth_paths = self.depth_paths[index]
        mask_paths = self.mask_paths[index]
        pose_list = self.poses[index] if index < len(self.poses) else [None] * self.S
        intrinsic_list = self.intrinsics[index] if index < len(self.intrinsics) else [None] * self.S

        views = []

        for i in range(self.S):
            rgb_path = rgb_paths[i]
            depth_path = depth_paths[i] if i < len(depth_paths) else None
            mask_path = mask_paths[i] if i < len(mask_paths) else None
            camera_pose = pose_list[i] if i < len(pose_list) else None
            K = intrinsic_list[i] if i < len(intrinsic_list) else None

            # 加载RGB图像
            try:
                rgb_image = imread_cv2(rgb_path)
                if rgb_image is None:
                    print(f"Error: Cannot read image: {rgb_path}")
                    # 创建一个占位符图像
                    rgb_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
            except Exception as e:
                print(f"Error reading image {rgb_path}: {e}")
                rgb_image = np.ones((480, 640, 3), dtype=np.uint8) * 128

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

            # 使用从poses_bounds.npy加载的内参或默认内参
            if K is not None:
                # 调整内参以适应裁剪后的图像
                # 假设原始内参是基于某种标准尺寸的
                scale_x = w / 640.0 if w > 0 else 1  # 假设原始宽度640
                scale_y = h / 480.0 if h > 0 else 1  # 假设原始高度480
                K = K.copy()
                K[0, 0] *= scale_x  # fx
                K[1, 1] *= scale_y  # fy
                K[0, 2] *= scale_x  # cx
                K[1, 2] *= scale_y  # cy
            else:
                # 如果没有内参，根据图像尺寸估计
                K = np.array([
                    [0.5 * w, 0, 0.5 * w],
                    [0, 0.5 * h, 0.5 * h],
                    [0, 0, 1]
                ], dtype=np.float32)

            original_height, original_width = h, w

            # 使用从poses_bounds.npy加载的位姿或固定位姿
            if camera_pose is not None:
                camera_pose = camera_pose.copy()
            elif self.fixed_pose:
                camera_pose = np.eye(4, dtype=np.float32)
            else:
                # 如果没有位姿，使用单位矩阵
                camera_pose = np.eye(4, dtype=np.float32)

            # 加载深度图（如果存在）
            depthmap = None
            if depth_path and os.path.exists(depth_path):
                try:
                    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                    if depth_img is not None:
                        # 裁剪深度图以匹配RGB图像
                        depthmap = depth_img[:h, :w]

                        if len(depthmap.shape) == 2:
                            # EndoNeRF深度图可能是16位或浮点数
                            if depthmap.dtype == np.uint16:
                                depthmap = depthmap.astype(np.float32) / 65535.0 * 10.0  # 假设最大深度10米
                            elif depthmap.dtype == np.uint8:
                                depthmap = depthmap.astype(np.float32) / 255.0 * 10.0  # 假设最大深度10米
                            else:
                                depthmap = depthmap.astype(np.float32)
                        elif len(depthmap.shape) == 3:
                            # 如果是3通道深度图，转换为灰度
                            depthmap = cv2.cvtColor(depthmap, cv2.COLOR_BGR2GRAY)
                            depthmap = depthmap.astype(np.float32) / 255.0 * 10.0

                        # 处理无效深度值
                        depthmap[depthmap <= 0] = 0
                except Exception as e:
                    print(f"Warning: Could not load depth map {depth_path}: {e}")

            # 加载mask（如果存在）
            mask = None
            if mask_path and os.path.exists(mask_path):
                try:
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask_img is not None:
                        mask = mask_img[:h, :w]
                        mask = (mask > 0).astype(np.float32)

                        # 应用mask到图像
                        if mask.shape[:2] == rgb_image.shape[:2]:
                            # 将mask扩展为3通道以匹配RGB图像
                            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                            # 将图像与mask相乘（背景变为黑色）
                            rgb_image = (rgb_image * mask_3d).astype(np.uint8)
                        else:
                            print(f"Warning: Mask shape {mask.shape} doesn't match image shape {rgb_image.shape[:2]}")
                except Exception as e:
                    print(f"Warning: Could not load mask {mask_path}: {e}")

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
                    if mask is not None:
                        mask = mask[:h_final, :w_final]

                    # 调整内参
                    scale_x = w_final / w if w > 0 else 1
                    scale_y = h_final / h if h > 0 else 1
                    K[0, 0] *= scale_x
                    K[1, 1] *= scale_y
                    K[0, 2] *= scale_x
                    K[1, 2] *= scale_y

            # 如果调整大小后mask存在且形状匹配，再次应用mask
            if mask is not None and resolution:
                if mask.shape[:2] == rgb_image.shape[:2]:
                    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                    rgb_image = (rgb_image * mask_3d).astype(np.uint8)
                else:
                    # 如果形状不匹配，调整mask大小
                    try:
                        mask_resized = cv2.resize(mask, (w_final, h_final), interpolation=cv2.INTER_NEAREST)
                        mask_3d = np.repeat(mask_resized[:, :, np.newaxis], 3, axis=2)
                        rgb_image = (rgb_image * mask_3d).astype(np.uint8)
                        mask = mask_resized  # 更新mask为调整大小后的版本
                    except Exception as e:
                        print(f"Warning: Failed to resize and apply mask: {e}")

            # 将NumPy数组转换为PIL图像
            pil_image = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))

            # 构建视图字典
            view_data = dict(
                img=pil_image,  # 现在返回PIL图像（已应用mask）
                camera_pose=camera_pose,
                camera_intrinsics=K,
                dataset=self.dataset_label,
                label=os.path.basename(os.path.dirname(rgb_path)),
                instance=os.path.basename(rgb_path),
                original_size=(original_height, original_width)
            )

            if depthmap is not None:
                view_data['depthmap'] = depthmap
            if mask is not None:
                view_data['mask'] = mask

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
    dataset_location = '/path/to/endonerf/dataset'

    print("Testing EndoNeRF dataset...")

    dataset = EndoNeRFDUSt3R(
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
        fixed_pose=True,  # EndoNeRF通常有固定位姿
        poses_bounds_file='poses_bounds.npy'
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

            if 'mask' in view:
                print(f"  Mask shape: {view['mask'].shape}")

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
        plt.savefig('./test_output/endonerf_sample.png')
        plt.show()
        print("Sample saved to ./test_output/endonerf_sample.png")