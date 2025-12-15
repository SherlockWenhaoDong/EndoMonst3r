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
from torchvision.transforms import ColorJitter, GaussianBlur
from functools import partial
import xml.etree.ElementTree as ET

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from dust3r.utils.misc import get_stride_distribution

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')


class EndoDUSt3R(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='data/endoscopic',
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
                 calib_xml_path='/home/bygpu/Downloads/results/2_NON_RIGID_DEFORMATION_PULL/Xi_left.xml',
                 *args,
                 **kwargs):

        print('loading endoscopic dataset...')
        super().__init__(*args, **kwargs)
        self.dataset_label = 'endoscopic'
        self.split = dset
        self.S = S  # stride
        self.N = N  # min num points (not used for endoscopic, kept for compatibility)
        self.verbose = verbose
        self.use_augs = use_augs
        self.dset = dset
        self.fixed_pose = fixed_pose
        self.calib_xml_path = calib_xml_path

        # 存储路径
        self.rgb_paths = []
        self.depth_paths = []
        self.mask_paths = []
        self.full_idxs = []
        self.sample_stride = []
        self.strides = strides

        self.sequences = []

        # 直接使用提供的frames_cam0目录
        if os.path.isdir(dataset_location):
            if 'frames_cam0' in dataset_location:
                # 直接使用frames_cam0目录
                self.sequences.append({
                    'path': dataset_location,
                    'name': os.path.basename(os.path.dirname(dataset_location))
                })
            else:
                # 查找frames_cam0子目录
                for seq in glob.glob(os.path.join(dataset_location, "*/frames_cam0")):
                    seq_name = seq.split('/')[-2]  # 获取序列名
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

        # 从XML文件读取固定内参
        self.fixed_intrinsics = self._parse_calibration_xml(calib_xml_path) if calib_xml_path else None
        if self.fixed_intrinsics:
            print(f"Loaded fixed intrinsics from {calib_xml_path}")
            print(f"  fx={self.fixed_intrinsics['fx']:.2f}, fy={self.fixed_intrinsics['fy']:.2f}")
            print(f"  cx={self.fixed_intrinsics['cx']:.2f}, cy={self.fixed_intrinsics['cy']:.2f}")
            print(f"  image size: {self.fixed_intrinsics['width']}x{self.fixed_intrinsics['height']}")
            if 'distortion' in self.fixed_intrinsics:
                print(f"  distortion coefficients: {self.fixed_intrinsics['distortion']}")
            if 'registration' in self.fixed_intrinsics:
                print(f"  registration matrix shape: {self.fixed_intrinsics['registration'].shape}")

        # 加载所有图像对
        self._load_image_pairs(strides, clip_step, clip_step_last_skip, quick)

        # 如果需要重新采样
        if len(strides) > 1 and dist_type is not None:
            self._resample_clips(strides, dist_type)

        print(f'collected {len(self.rgb_paths)} clips of length {self.S} in {dataset_location} (dset={dset})')

    def _parse_calibration_xml(self, xml_path):
        """
        解析相机标定XML文件
        文件格式：
        <?xml version="1.0" encoding="utf-8"?>
        <propertyfile version="1.1" name="">
            <param name="imageSize">1264 1008 </param>
            <param name="K">1045.87471656702 0 589.864029607369 0 1044.63816741387 506.467596734399 0 0 1 </param>
            <param name="distortion">-0.0219845811218631 0.0258416946521303 -0.000632101689148172 0.00221558249094752 0 </param>
            <param name="registration">1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 </param>
        </propertyfile>
        """
        if not os.path.exists(xml_path):
            print(f"Warning: Calibration XML file not found: {xml_path}")
            return None

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            calibration_data = {}

            # 解析图像尺寸
            image_size_elem = root.find("./param[@name='imageSize']")
            if image_size_elem is not None:
                size_values = list(map(float, image_size_elem.text.strip().split()))
                if len(size_values) >= 2:
                    calibration_data['width'] = int(size_values[0])
                    calibration_data['height'] = int(size_values[1])

            # 解析相机内参矩阵 K
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

            # 解析畸变系数
            distortion_elem = root.find("./param[@name='distortion']")
            if distortion_elem is not None:
                distortion_values = list(map(float, distortion_elem.text.strip().split()))
                calibration_data['distortion'] = np.array(distortion_values, dtype=np.float32)

            # 解析注册矩阵（可能是位姿或外参）
            registration_elem = root.find("./param[@name='registration']")
            if registration_elem is not None:
                reg_values = list(map(float, registration_elem.text.strip().split()))
                if len(reg_values) >= 16:
                    # 可能是4x4变换矩阵
                    registration = np.array(reg_values, dtype=np.float32).reshape(4, 4)
                    calibration_data['registration'] = registration

                    # 提取旋转和平移
                    calibration_data['R'] = registration[:3, :3]  # 旋转矩阵
                    calibration_data['t'] = registration[:3, 3]  # 平移向量
                    calibration_data['pose'] = registration  # 完整的4x4位姿矩阵

            return calibration_data

        except Exception as e:
            print(f"Error parsing calibration XML {xml_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_image_pairs(self, strides, clip_step, clip_step_last_skip, quick):
        """加载所有图像对"""
        for seq_info in self.sequences:
            seq_path = seq_info['path']
            seq_name = seq_info['name']

            if self.verbose:
                print(f'Processing sequence: {seq_name}')

            # 获取所有RGB图像，按文件名排序
            rgb_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            rgb_files = []
            for ext in rgb_extensions:
                rgb_files.extend(glob.glob(os.path.join(seq_path, ext)))

            # 按文件名排序（假设文件名包含数字）
            def extract_number(filename):
                # 从文件名中提取数字
                import re
                numbers = re.findall(r'\d+', os.path.basename(filename))
                return int(numbers[0]) if numbers else 0

            rgb_files = sorted(rgb_files, key=extract_number)

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
                    valid_pair = True

                    for idx in full_idx:
                        if idx < len(rgb_files):
                            rgb_pair.append(rgb_files[idx])
                        else:
                            valid_pair = False
                            break

                    if not valid_pair:
                        continue

                    # 检查所有图像文件是否存在
                    all_exist = True
                    for rgb_path in rgb_pair:
                        if not os.path.exists(rgb_path):
                            all_exist = False
                            break

                    if not all_exist:
                        continue

                    # 生成对应的深度图和mask路径
                    depth_pair = []
                    mask_pair = []

                    for rgb_path in rgb_pair:
                        rgb_dir = os.path.dirname(rgb_path)
                        rgb_filename = os.path.basename(rgb_path)
                        rgb_name, rgb_ext = os.path.splitext(rgb_filename)

                        # 深度图路径
                        depth_path = None
                        possible_depth_dirs = [
                            os.path.join(rgb_dir, 'depth'),
                            os.path.join(os.path.dirname(rgb_dir), 'depth'),
                            os.path.join(rgb_dir, 'depths'),
                            os.path.join(os.path.dirname(rgb_dir), 'depths'),
                            rgb_dir
                        ]

                        for depth_dir in possible_depth_dirs:
                            if os.path.exists(depth_dir):
                                possible_depth_files = [
                                    os.path.join(depth_dir, f'{rgb_name}.png'),
                                    os.path.join(depth_dir, f'{rgb_name}_depth.png'),
                                    os.path.join(depth_dir, f'{rgb_name}_depth.jpg'),
                                    os.path.join(depth_dir, f'depth_{rgb_name}.png'),
                                    os.path.join(depth_dir, f'depth_{rgb_name}.jpg'),
                                ]
                                for depth_file in possible_depth_files:
                                    if os.path.exists(depth_file):
                                        depth_path = depth_file
                                        break
                            if depth_path:
                                break

                        # Mask路径 - 根据您的描述：rgb路径后加_mask，文件名为rgb名_combined_masks.png
                        mask_path = None
                        mask_dir = rgb_dir + '_mask'
                        mask_filename = f'{rgb_name}_combined_masks.png'
                        mask_file = os.path.join(mask_dir, mask_filename)

                        if os.path.exists(mask_file):
                            mask_path = mask_file
                        else:
                            # 如果上述模式不存在，尝试其他可能的模式
                            possible_mask_files_in_dir = [
                                os.path.join(mask_dir, f'{rgb_name}_combined_masks_png.png'),
                                os.path.join(mask_dir, f'{rgb_name}_combined_masks.jpg'),
                                os.path.join(mask_dir, f'{rgb_name}_mask.png'),
                                os.path.join(mask_dir, f'{rgb_name}_mask.jpg'),
                                os.path.join(mask_dir, f'{rgb_name}.png'),
                                os.path.join(mask_dir, f'{rgb_name}.jpg'),
                            ]
                            for possible_file in possible_mask_files_in_dir:
                                if os.path.exists(possible_file):
                                    mask_path = possible_file
                                    break

                        # 如果标准mask目录不存在，尝试其他可能的目录
                        if mask_path is None:
                            possible_mask_dirs = [
                                os.path.join(os.path.dirname(rgb_dir), 'mask'),
                                os.path.join(os.path.dirname(rgb_dir), 'masks'),
                                os.path.join(rgb_dir, 'mask'),
                                os.path.join(rgb_dir, 'masks'),
                                rgb_dir
                            ]

                            for possible_mask_dir in possible_mask_dirs:
                                if os.path.exists(possible_mask_dir):
                                    possible_mask_files = [
                                        os.path.join(possible_mask_dir, f'{rgb_name}_combined_masks.png'),
                                        os.path.join(possible_mask_dir, f'{rgb_name}_combined_masks_png.png'),
                                        os.path.join(possible_mask_dir, f'{rgb_name}_combined_masks.jpg'),
                                        os.path.join(possible_mask_dir, f'{rgb_name}_mask.png'),
                                        os.path.join(possible_mask_dir, f'{rgb_name}_mask.jpg'),
                                        os.path.join(possible_mask_dir, f'{rgb_name}.png'),
                                        os.path.join(possible_mask_dir, f'{rgb_name}.jpg'),
                                    ]
                                    for possible_file in possible_mask_files:
                                        if os.path.exists(possible_file):
                                            mask_path = possible_file
                                            break
                                if mask_path:
                                    break

                        depth_pair.append(depth_path)
                        mask_pair.append(mask_path)

                    self.rgb_paths.append(rgb_pair)
                    self.depth_paths.append(depth_pair)
                    self.mask_paths.append(mask_pair)
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
        max_num_clips = self.stride_counts[strides[np.argmax(dist)]]
        num_clips_each_stride = [
            min(self.stride_counts[stride], int(dist[i] * max_num_clips))
            for i, stride in enumerate(strides)
        ]

        print('resampled_num_clips_each_stride:', num_clips_each_stride)

        resampled_idxs = []
        for i, stride in enumerate(strides):
            available_idxs = self.stride_idxs[stride]
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

    def __len__(self):
        return len(self.rgb_paths)

    def _get_views(self, index, resolution, rng):
        """获取一对视图"""
        rgb_paths = self.rgb_paths[index]
        depth_paths = self.depth_paths[index]
        mask_paths = self.mask_paths[index]

        views = []

        for i in range(self.S):
            rgb_path = rgb_paths[i]
            depth_path = depth_paths[i] if i < len(depth_paths) else None
            mask_path = mask_paths[i] if i < len(mask_paths) else None

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

            # 使用固定内参
            if self.fixed_intrinsics:
                K = self.fixed_intrinsics['K'].copy()
                original_height = self.fixed_intrinsics.get('height', h)
                original_width = self.fixed_intrinsics.get('width', w)

                # 调整内参以适应裁剪后的图像
                scale_x = w / original_width if original_width > 0 else 1
                scale_y = h / original_height if original_height > 0 else 1
                K[0, 0] *= scale_x  # fx
                K[1, 1] *= scale_y  # fy
                K[0, 2] *= scale_x  # cx
                K[1, 2] *= scale_y  # cy
            else:
                # 如果没有固定内参，根据图像尺寸估计
                K = np.array([
                    [0.5 * w, 0, 0.5 * w],
                    [0, 0.5 * h, 0.5 * h],
                    [0, 0, 1]
                ], dtype=np.float32)
                original_height, original_width = h, w

            # 使用固定位姿
            camera_pose = np.eye(4, dtype=np.float32)
            if self.fixed_intrinsics and 'pose' in self.fixed_intrinsics:
                camera_pose = self.fixed_intrinsics['pose'].copy()

            # 加载深度图（如果存在）
            depthmap = None
            if depth_path and os.path.exists(depth_path):
                try:
                    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                    if depth_img is not None:
                        # 裁剪深度图以匹配RGB图像
                        depthmap = depth_img[:h, :w]
                        if len(depthmap.shape) == 2:
                            if depthmap.dtype == np.uint16:
                                depthmap = depthmap.astype(np.float32) / 65535.0 * 1000.0
                            elif depthmap.dtype == np.uint8:
                                depthmap = depthmap.astype(np.float32) / 255.0 * 1000.0
                        elif len(depthmap.shape) == 3:
                            # 如果是3通道深度图，转换为灰度
                            depthmap = cv2.cvtColor(depthmap, cv2.COLOR_BGR2GRAY)
                            depthmap = depthmap.astype(np.float32) / 255.0 * 1000.0
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
            from PIL import Image
            pil_image = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))

            # 构建视图字典 - 返回PIL图像
            view_data = dict(
                img=pil_image,  # 现在返回PIL图像（已应用mask）
                camera_pose=camera_pose,
                camera_intrinsics=K,
                dataset=self.dataset_label,
                label=os.path.basename(
                    os.path.dirname(os.path.dirname(rgb_path))) if 'frames_cam0' in rgb_path else os.path.basename(
                    os.path.dirname(rgb_path)),
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
        from PIL import Image
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
    dataset_location = '/home/bygpu/Downloads/SurgicalRecon/2_NON_RIGID_DEFORMATION_PULL/SLOW_HORIZONTAL_PULL_CC/frames_cam0'
    calib_xml_path = '/path/to/your/calibration.xml'  # 您的标定XML文件路径

    print("Testing EndoDUSt3R dataset...")

    dataset = EndoDUSt3R(
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
        fixed_pose=True,
        calib_xml_path=calib_xml_path
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
            axes[i].imshow(cv2.cvtColor(view['img'], cv2.COLOR_BGR2RGB))
            axes[i].set_title(f'View {i}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig('./test_output/endo_sample.png')
        plt.show()
        print("Sample saved to ./test_output/endo_sample.png")