# --------------------------------------------------------
# Depth prediction and saving code for DUSt3R model
# (只保存PNG格式，所有深度图使用统一scale)
# --------------------------------------------------------
import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
import cv2
from PIL import Image
import glob
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
project_root = '/home/bygpu/Downloads/EndoMonst3r-main'
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'dust3r'))

# Import torchvision transforms
from torchvision import transforms

# Import dust3r modules
try:
    from dust3r.model import AsymmetricCroCo3DStereo, inf
except ImportError as e:
    print(f"Error importing dust3r modules: {e}")
    sys.exit(1)


class DepthPredictor:
    """Depth predictor for DUSt3R models"""

    def __init__(self, model_path, device='cuda', img_size=512):
        """
        Initialize depth predictor

        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            img_size: Input image size
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.global_max_depth = 0.0  # 用于记录全局最大深度

        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()

        print(f"Initialized DepthPredictor on {self.device}")
        print(f"Model input size: {img_size}")

    def load_model(self, model_path):
        """Load trained model from checkpoint"""
        print(f"Loading model from: {model_path}")

        try:
            # Load checkpoint
            if model_path.endswith('.pth'):
                checkpoint = torch.load(model_path, map_location=self.device)
            else:
                checkpoint = torch.load(model_path, map_location=self.device)

            # Create model with same configuration as training
            model = AsymmetricCroCo3DStereo(
                pos_embed='RoPE100',
                patch_embed_cls='ManyAR_PatchEmbed',
                img_size=(self.img_size, self.img_size),
                head_type='dpt',
                output_mode='pts3d',
                depth_mode=('exp', -inf, inf),
                conf_mode=('exp', 1, inf),
                enc_embed_dim=1024,
                enc_depth=24,
                enc_num_heads=16,
                dec_embed_dim=768,
                dec_depth=12,
                dec_num_heads=12
            )

            # Load state dict
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            # Load weights
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)

            print("✓ Model loaded successfully")

            # Check which parameters are trainable
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Trainable parameters: {trainable_params:,}/{total_params:,}")

            return model

        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        # Load image
        img = Image.open(image_path).convert('RGB')
        original_size = img.size  # (width, height)

        # Resize to model input size
        img_resized = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img_resized).unsqueeze(0).to(self.device)

        # Default camera parameters
        camera_pose = torch.eye(4).unsqueeze(0).to(self.device)

        # Default intrinsics
        intrinsics = torch.tensor([[
            [0.5 * self.img_size, 0, 0.5 * self.img_size],
            [0, 0.5 * self.img_size, 0.5 * self.img_size],
            [0, 0, 1]
        ]]).float().to(self.device)

        # Get filename for instance
        filename = os.path.basename(image_path)

        return {
            'img': img_tensor,
            'camera_pose': camera_pose,
            'camera_intrinsics': intrinsics,
            'original_size': torch.tensor([original_size[1], original_size[0]]).to(self.device),  # (height, width)
            'index': torch.tensor([0]).to(self.device),
            'dataset': 'predict',
            'label': filename.split('.')[0],
            'instance': filename,
            'resized_size': torch.tensor([self.img_size, self.img_size]).to(self.device)
        }, original_size

    def predict_depth(self, image_path):
        """Predict depth for single image"""
        # Preprocess image
        view1, original_size = self.preprocess_image(image_path)

        # Create second view with different index (required by model)
        view2 = view1.copy()
        view2['index'] = torch.tensor([1]).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(view1, view2)

        # Extract depth from output
        if isinstance(output, tuple) and len(output) == 2:
            pred1, pred2 = output

            # Extract 3D points and compute depth
            if 'pts3d' in pred1:
                pts3d = pred1['pts3d']  # Shape: [1, H, W, 3]
                depth = pts3d[0, :, :, 2]  # Z coordinate is depth
            elif 'depth' in pred1:
                depth = pred1['depth'][0] if isinstance(pred1['depth'], list) else pred1['depth']
                depth = depth.squeeze()
            else:
                # Try to find depth in other keys
                for key in pred1.keys():
                    if 'depth' in key.lower():
                        depth = pred1[key]
                        if isinstance(depth, list):
                            depth = depth[0]
                        depth = depth.squeeze()
                        break
                else:
                    raise ValueError("Cannot extract depth from prediction")
        elif isinstance(output, dict):
            # Handle dict output
            if 'pred1' in output and 'pts3d' in output['pred1']:
                pts3d = output['pred1']['pts3d']
                depth = pts3d[0, :, :, 2]
            elif 'depth' in output:
                depth = output['depth'][0] if isinstance(output['depth'], list) else output['depth']
                depth = depth.squeeze()
            else:
                # Try to find depth in other keys
                for key in output.keys():
                    if 'depth' in key.lower():
                        depth = output[key]
                        if isinstance(depth, list):
                            depth = depth[0]
                        depth = depth.squeeze()
                        break
                else:
                    raise ValueError("Cannot extract depth from dict output")
        else:
            raise ValueError(f"Unexpected output format: {type(output)}")

        # Resize depth to original image size if needed
        original_h, original_w = original_size[1], original_size[0]
        current_h, current_w = depth.shape[-2], depth.shape[-1]

        if current_h != original_h or current_w != original_w:
            depth_np = depth.cpu().numpy()
            depth_resized = cv2.resize(depth_np, (original_w, original_h),
                                       interpolation=cv2.INTER_LINEAR)
            depth = torch.from_numpy(depth_resized)

        return depth.cpu().numpy()

    def get_depth_statistics(self, depth_map):
        """获取深度图统计信息"""
        # 去掉负值和零值
        valid_depth = depth_map[depth_map > 0]

        if len(valid_depth) == 0:
            return 0.0, 0.0, 0.0

        # 计算统计信息
        mean_depth = np.mean(valid_depth)
        median_depth = np.median(valid_depth)
        max_depth = np.max(valid_depth)

        return mean_depth, median_depth, max_depth

    def save_depth_png(self, depth_map, output_path, global_max_depth):
        """Save depth map as 16-bit PNG with global scaling"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Clip negative values to 0
        depth_map = np.maximum(depth_map, 0)

        # Find valid depth range (positive values)
        valid_depth = depth_map[depth_map > 0]

        if len(valid_depth) > 0:
            # 使用全局最大深度值进行缩放
            if global_max_depth > 0:
                # 线性缩放到0-65535范围 (16-bit)
                # 使用全局最大深度值，确保所有图像使用相同的scale
                depth_scaled = np.clip(depth_map / global_max_depth, 0, 1) * 65535
            else:
                # 如果没有全局最大深度，使用当前图像的最大深度
                max_val = np.max(valid_depth)
                depth_scaled = np.clip(depth_map / max_val, 0, 1) * 65535
        else:
            depth_scaled = np.zeros_like(depth_map, dtype=np.uint16)

        # Convert to 16-bit unsigned integer
        depth_16bit = depth_scaled.astype(np.uint16)

        # Save as PNG
        cv2.imwrite(output_path, depth_16bit)

        print(f"  Saved depth map to: {output_path}")
        if global_max_depth > 0:
            print(f"    (Using global max depth: {global_max_depth:.3f}m)")

    def save_depth_visualization(self, depth_map, output_path, colormap=cv2.COLORMAP_PLASMA):
        """Save depth visualization as color image"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Clip negative values to 0
        depth_map = np.maximum(depth_map, 0)

        # Normalize depth for visualization (使用每张图自己的统计信息)
        valid_depth = depth_map[depth_map > 0]
        if len(valid_depth) > 0:
            min_val = np.percentile(valid_depth, 5)
            max_val = np.percentile(valid_depth, 95)
            if max_val > min_val:
                depth_normalized = (depth_map - min_val) / (max_val - min_val)
                depth_normalized = np.clip(depth_normalized, 0, 1)
            else:
                depth_normalized = np.zeros_like(depth_map)
        else:
            depth_normalized = np.zeros_like(depth_map)

        # Apply colormap
        depth_8bit = (depth_normalized * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_8bit, colormap)

        # Save
        cv2.imwrite(output_path, depth_colored)
        print(f"  Saved depth visualization to: {output_path}")

    def analyze_dataset_depth(self, input_dir):
        """分析数据集中的深度统计信息，确定全局最大深度"""
        print("\nAnalyzing dataset depth statistics...")

        # Find image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            # Also search in subdirectories
            image_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))

        if not image_files:
            print(f"✗ No images found in {input_dir}")
            return 0.0

        image_files = sorted(list(set(image_files)))  # Remove duplicates
        print(f"Found {len(image_files)} images for analysis")

        # 收集所有深度统计信息
        all_max_depths = []
        all_mean_depths = []
        all_median_depths = []

        # 分析部分图像来确定全局最大深度
        analyze_count = min(50, len(image_files))  # 最多分析50张图片
        print(f"Analyzing {analyze_count} images for depth statistics...")

        pbar = tqdm(image_files[:analyze_count], desc="Analyzing images")
        for img_path in pbar:
            try:
                # 预测深度
                depth_map = self.predict_depth(img_path)

                # 获取统计信息
                mean_depth, median_depth, max_depth = self.get_depth_statistics(depth_map)

                if max_depth > 0:
                    all_max_depths.append(max_depth)
                    all_mean_depths.append(mean_depth)
                    all_median_depths.append(median_depth)

            except Exception as e:
                print(f"  Error analyzing {img_path}: {str(e)}")
                continue

        if not all_max_depths:
            print("✗ No valid depth predictions for analysis")
            return 0.0

        # 确定全局最大深度
        # 使用所有最大深度的95百分位作为全局最大深度，避免异常值
        global_max_depth = np.percentile(all_max_depths, 95)

        print(f"\nDepth Statistics:")
        print(f"  Number of analyzed images: {len(all_max_depths)}")
        print(f"  Average max depth: {np.mean(all_max_depths):.3f}m")
        print(f"  Median max depth: {np.median(all_max_depths):.3f}m")
        print(f"  Min max depth: {np.min(all_max_depths):.3f}m")
        print(f"  Max max depth: {np.max(all_max_depths):.3f}m")
        print(f"  Global max depth (95th percentile): {global_max_depth:.3f}m")

        return global_max_depth

    def process_directory(self, input_dir, output_dir, save_viz=False, global_max_depth=None):
        """Process all images in a directory with unified scaling"""
        print(f"\n{'=' * 60}")
        print(f"Processing directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"{'=' * 60}")

        # 如果没有提供全局最大深度，先分析数据集
        if global_max_depth is None:
            global_max_depth = self.analyze_dataset_depth(input_dir)
            if global_max_depth <= 0:
                print("✗ Could not determine global max depth, using individual scaling")
                global_max_depth = 0.0
            else:
                print(f"\nUsing global max depth: {global_max_depth:.3f}m for all images")

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'depth_png'), exist_ok=True)
        if save_viz:
            os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)

        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            # Also search in subdirectories
            image_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))

        if not image_files:
            print(f"✗ No images found in {input_dir}")
            return 0

        image_files = sorted(list(set(image_files)))  # Remove duplicates
        print(f"\nFound {len(image_files)} images for processing")

        # Process each image with unified scaling
        success_count = 0
        depth_stats = []  # 收集每张图的统计信息

        pbar = tqdm(image_files, desc="Processing images")
        for img_path in pbar:
            try:
                # Get base filename
                filename = os.path.basename(img_path)
                basename = os.path.splitext(filename)[0]

                # Predict depth
                depth_map = self.predict_depth(img_path)

                # Get depth statistics
                mean_depth, median_depth, max_depth = self.get_depth_statistics(depth_map)
                if max_depth > 0:
                    depth_stats.append({
                        'filename': filename,
                        'mean': mean_depth,
                        'median': median_depth,
                        'max': max_depth
                    })

                # Save depth as PNG with global scaling
                output_path = os.path.join(output_dir, 'depth_png', f'{basename}_depth.png')
                self.save_depth_png(depth_map, output_path, global_max_depth)

                # Save visualization if requested
                if save_viz:
                    viz_path = os.path.join(output_dir, 'visualizations', f'{basename}_depth_viz.png')
                    self.save_depth_visualization(depth_map, viz_path)

                success_count += 1
                pbar.set_postfix_str(f"Success: {success_count}/{len(image_files)}")

            except Exception as e:
                print(f"\n  Error processing {img_path}: {str(e)}")
                continue

        # Print final statistics
        if depth_stats:
            print(f"\n{'=' * 60}")
            print("DEPTH PROCESSING SUMMARY")
            print(f"{'=' * 60}")
            print(f"Processed images: {success_count}/{len(image_files)}")
            print(f"Global max depth used: {global_max_depth:.3f}m")
            print(f"\nDepth Statistics:")

            # 计算总体统计
            all_means = [s['mean'] for s in depth_stats]
            all_medians = [s['median'] for s in depth_stats]
            all_maxes = [s['max'] for s in depth_stats]

            print(
                f"  Mean depth - Avg: {np.mean(all_means):.3f}m, Min: {np.min(all_means):.3f}m, Max: {np.max(all_means):.3f}m")
            print(
                f"  Median depth - Avg: {np.mean(all_medians):.3f}m, Min: {np.min(all_medians):.3f}m, Max: {np.max(all_medians):.3f}m")
            print(
                f"  Max depth - Avg: {np.mean(all_maxes):.3f}m, Min: {np.min(all_maxes):.3f}m, Max: {np.max(all_maxes):.3f}m")

            # 显示深度最大的前5张图片
            print(f"\nTop 5 images with maximum depth:")
            sorted_stats = sorted(depth_stats, key=lambda x: x['max'], reverse=True)[:5]
            for i, stat in enumerate(sorted_stats, 1):
                print(f"  {i}. {stat['filename']}: max={stat['max']:.3f}m, mean={stat['mean']:.3f}m")

        print(f"\n{'=' * 60}")
        print(f"PROCESSING COMPLETED")
        print(f"{'=' * 60}")
        print(f"✓ Successfully processed: {success_count}/{len(image_files)} images")
        print(f"✓ Depth maps saved to: {os.path.join(output_dir, 'depth_png')}")
        if save_viz:
            print(f"✓ Visualizations saved to: {os.path.join(output_dir, 'visualizations')}")
        if global_max_depth > 0:
            print(f"✓ All depth maps use unified scaling with max depth: {global_max_depth:.3f}m")
        print(f"{'=' * 60}")

        return success_count, global_max_depth


def main():
    parser = argparse.ArgumentParser(description='Depth prediction for DUSt3R model (统一scale的PNG格式)')

    # Model path
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')

    # Input options
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input images')

    # Output options
    parser.add_argument('--output_dir', type=str, default='./depth_results',
                        help='Output directory for depth maps')
    parser.add_argument('--save_viz', action='store_true', default=False,
                        help='Save depth visualizations (color images)')

    # Global scaling options
    parser.add_argument('--global_max_depth', type=float, default=None,
                        help='Fixed global maximum depth value for scaling (meters). '
                             'If not specified, automatically analyzes dataset to determine optimal value.')

    # Model parameters
    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size for model')

    args = parser.parse_args()

    # Create predictor
    predictor = DepthPredictor(
        model_path=args.model_path,
        device='cuda',
        img_size=args.img_size
    )

    # Process directory with unified scaling
    if not os.path.exists(args.input_dir):
        print(f"✗ Input directory not found: {args.input_dir}")
        return

    # 处理整个目录，使用统一scale
    success_count, used_global_max_depth = predictor.process_directory(
        args.input_dir,
        args.output_dir,
        save_viz=args.save_viz,
        global_max_depth=args.global_max_depth
    )

    if success_count > 0:
        print(f"\n所有深度图已使用统一scale保存完成！")
        print(f"全局最大深度: {used_global_max_depth:.3f}米")
        print(f"要还原原始深度值，使用公式: depth_meters = png_value * {used_global_max_depth / 65535:.6f}")
    else:
        print(f"\n处理失败，没有成功生成深度图")


if __name__ == "__main__":
    main()