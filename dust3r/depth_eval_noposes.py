import torch
import numpy as np
import cv2
import glob
import os
import json
import re
from tqdm import tqdm
import argparse
import sys
from PIL import Image
import torchvision.transforms as transforms

# ============================================================
# Set import paths
# ============================================================

project_root = '/home/bygpu/Downloads/EndoMonst3r-main'
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'dust3r'))


# ============================================================
# Model loading function
# ============================================================

def load_model(model_path, device):
    """Load trained model"""
    print(f"Loading model from: {model_path}")

    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Checkpoint keys: {list(checkpoint.keys())}")

        # Import model
        from dust3r.model_noposes import AsymmetricCroCo3DStereo

        # Create model with same configuration as training
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

        # Load weights
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.to(device)
        model.eval()
        print("✓ Model loaded successfully!")
        return model

    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# Data loading functions
# ============================================================

def parse_scared_pose(pose_path):
    """Parse SCARED pose JSON file"""
    if not os.path.exists(pose_path):
        return None, None

    try:
        with open(pose_path, 'r') as f:
            data = json.load(f)

        # Extract camera intrinsics
        if 'camera-calibration' in data and 'KL' in data['camera-calibration']:
            K_data = data['camera-calibration']['KL']
            K = np.array([
                [K_data[0][0], K_data[0][1], K_data[0][2]],
                [K_data[1][0], K_data[1][1], K_data[1][2]],
                [K_data[2][0], K_data[2][1], K_data[2][2]]
            ], dtype=np.float32)
        else:
            # Default intrinsics
            K = np.eye(3, dtype=np.float32)
            K[0, 0] = K[1, 1] = 500.0
            K[0, 2] = 320.0
            K[1, 2] = 240.0

        # Extract camera pose
        if 'camera-pose' in data:
            pose_data = data['camera-pose']
            pose = np.array(pose_data, dtype=np.float32)
        else:
            # Identity pose
            pose = np.eye(4, dtype=np.float32)

        return K, pose
    except Exception as e:
        print(f"Error parsing pose {pose_path}: {e}")
        return None, None


def load_npz_depth(npz_path):
    """Load NPZ format depth map"""
    if not os.path.exists(npz_path):
        return None

    try:
        data = np.load(npz_path)
        if 'arr_0' in data:
            depth = data['arr_0']
        else:
            depth = data[data.files[0]]

        if len(depth.shape) == 3:
            depth = depth[:, :, 0]

        # SCARED depth is in mm, convert to meters
        depth = depth.astype(np.float32) / 1000.0

        # Handle invalid values
        depth[depth <= 0] = 0

        return depth
    except Exception as e:
        print(f"Error loading NPZ depth {npz_path}: {e}")
        return None


def load_png_depth(png_path):
    """Load PNG format depth map"""
    if not os.path.exists(png_path):
        return None

    try:
        depth_img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            return None

        if depth_img.dtype == np.uint16:
            depth = depth_img.astype(np.float32) / 1000.0  # Convert mm to meters
        elif depth_img.dtype == np.uint8:
            depth = depth_img.astype(np.float32) / 255.0 * 10.0  # Assume max depth 10m
        else:
            depth = depth_img.astype(np.float32)

        # Handle invalid values
        depth[depth <= 0] = 0

        return depth
    except Exception as e:
        print(f"Error loading PNG depth {png_path}: {e}")
        return None


# ============================================================
# Image preprocessing - Fixed for patch size constraint
# ============================================================

def preprocess_image_for_model(image_path, camera_pose=None, intrinsics=None, target_size=512, patch_size=16):
    """
    Preprocess image ensuring dimensions are multiples of patch_size
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Resize while ensuring dimensions are multiples of patch_size
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Adjust dimensions to be multiples of patch_size
    new_h = new_h - (new_h % patch_size)
    new_w = new_w - (new_w % patch_size)

    # Ensure minimum size
    if new_h < patch_size:
        new_h = patch_size
    if new_w < patch_size:
        new_w = patch_size

    img_resized = cv2.resize(img, (new_w, new_h))

    # Convert to PIL Image
    pil_image = Image.fromarray(img_resized)

    # Default intrinsics if not provided
    if intrinsics is None:
        intrinsics = np.array([
            [0.5 * new_w, 0, 0.5 * new_w],
            [0, 0.5 * new_h, 0.5 * new_h],
            [0, 0, 1]
        ], dtype=np.float32)
    else:
        # Adjust intrinsics for resized image
        scale_x = new_w / w
        scale_y = new_h / h
        intrinsics = intrinsics.copy()
        intrinsics[0, 0] *= scale_x  # fx
        intrinsics[1, 1] *= scale_y  # fy
        intrinsics[0, 2] *= scale_x  # cx
        intrinsics[1, 2] *= scale_y  # cy

    # Default pose if not provided
    if camera_pose is None:
        camera_pose = np.eye(4, dtype=np.float32)

    # Create view dictionary
    view_dict = {
        'img': pil_image,
        'camera_pose': camera_pose,
        'camera_intrinsics': intrinsics,
        'dataset': 'scared' if 'scared' in image_path.lower() else 'endonerf',
        'label': os.path.basename(os.path.dirname(os.path.dirname(image_path))),
        'instance': os.path.basename(image_path),
        'original_size': (h, w),
        'resized_size': (new_h, new_w),
        'index': 0
    }

    return view_dict


# ============================================================
# Inference function - Fixed for tuple output
# ============================================================

def inference_with_correct_format(model, image_path, camera_pose=None, intrinsics=None, device='cuda', img_size=512):
    """
    Perform inference handling both dict and tuple outputs
    """
    # Preprocess image ensuring patch size constraint
    view1_data = preprocess_image_for_model(image_path, camera_pose, intrinsics, img_size)
    view2_data = preprocess_image_for_model(image_path, camera_pose, intrinsics, img_size)

    # Set different indices for the two views
    view2_data['index'] = 1

    # Image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare view1
    view1 = {
        'img': transform(view1_data['img']).unsqueeze(0).to(device),
        'camera_pose': torch.from_numpy(view1_data['camera_pose']).float().unsqueeze(0).to(device),
        'camera_intrinsics': torch.from_numpy(view1_data['camera_intrinsics']).float().unsqueeze(0).to(device),
        'dataset': view1_data['dataset'],
        'label': view1_data['label'],
        'instance': view1_data['instance'],
        'original_size': torch.tensor([view1_data['original_size']]).to(device),
        'index': torch.tensor([view1_data['index']]).to(device)
    }

    # Prepare view2 (same image, different index)
    view2 = {
        'img': transform(view2_data['img']).unsqueeze(0).to(device),
        'camera_pose': torch.from_numpy(view2_data['camera_pose']).float().unsqueeze(0).to(device),
        'camera_intrinsics': torch.from_numpy(view2_data['camera_intrinsics']).float().unsqueeze(0).to(device),
        'dataset': view2_data['dataset'],
        'label': view2_data['label'],
        'instance': view2_data['instance'],
        'original_size': torch.tensor([view2_data['original_size']]).to(device),
        'index': torch.tensor([view2_data['index']]).to(device)
    }

    # Inference
    with torch.no_grad():
        output = model(view1, view2)

    # Handle tuple output (pred1, pred2)
    if isinstance(output, tuple):
        if len(output) == 2:
            pred1, pred2 = output
            # Extract depth from pred1
            if 'pts3d' in pred1:
                pts3d = pred1['pts3d']  # Shape: [1, H, W, 3]
                depth = pts3d[0, :, :, 2].cpu()  # Depth channel
            elif 'depth' in pred1:
                depth = pred1['depth'][0].cpu() if isinstance(pred1['depth'], list) else pred1['depth'].cpu()
            else:
                raise ValueError("Cannot extract depth from tuple output")
        else:
            raise ValueError(f"Unexpected tuple length: {len(output)}")
    elif isinstance(output, dict):
        # Handle dict output
        if 'pred1' in output and 'pts3d' in output['pred1']:
            pts3d = output['pred1']['pts3d']
            depth = pts3d[0, :, :, 2].cpu()
        elif 'depth' in output:
            depth = output['depth'][0].cpu() if isinstance(output['depth'], list) else output['depth'].cpu()
        else:
            raise ValueError("Cannot extract depth from dict output")
    else:
        raise ValueError(f"Unexpected output type: {type(output)}")

    # Resize depth to original image size
    original_h, original_w = view1_data['original_size']
    if depth.shape != (original_h, original_w):
        depth_np = depth.numpy()
        depth_resized = cv2.resize(depth_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        depth = torch.from_numpy(depth_resized)

    return depth


# ============================================================
# Evaluation functions
# ============================================================

def evaluate_depth_metrics(pred_depth, gt_depth, max_depth=80.0):
    """Calculate depth evaluation metrics"""
    # Filter valid pixels
    mask = (gt_depth > 0) & (gt_depth < max_depth) & (~np.isnan(gt_depth))

    if np.sum(mask) == 0:
        return {
            'Abs Rel': 0.0, 'Sq Rel': 0.0, 'RMSE': 0.0, 'Log RMSE': 0.0,
            'δ < 1.25': 0.0, 'δ < 1.25^2': 0.0, 'δ < 1.25^3': 0.0,
            'valid_pixels': 0
        }

    pred_valid = pred_depth[mask]
    gt_valid = gt_depth[mask]

    # Scale alignment (median scaling)
    scale_factor = np.median(gt_valid) / np.median(pred_valid)
    pred_aligned = pred_valid * scale_factor

    # Calculate metrics
    abs_rel = np.mean(np.abs(pred_aligned - gt_valid) / gt_valid)
    sq_rel = np.mean(((pred_aligned - gt_valid) ** 2) / gt_valid)
    rmse = np.sqrt(np.mean((pred_aligned - gt_valid) ** 2))

    # Log RMSE
    pred_log = np.log(np.clip(pred_aligned, 1e-7, None))
    gt_log = np.log(np.clip(gt_valid, 1e-7, None))
    log_rmse = np.sqrt(np.mean((pred_log - gt_log) ** 2))

    # Accuracy thresholds
    ratio = np.maximum(pred_aligned / gt_valid, gt_valid / pred_aligned)
    delta1 = np.mean(ratio < 1.25)
    delta2 = np.mean(ratio < 1.25 ** 2)
    delta3 = np.mean(ratio < 1.25 ** 3)

    return {
        'Abs Rel': float(abs_rel),
        'Sq Rel': float(sq_rel),
        'RMSE': float(rmse),
        'Log RMSE': float(log_rmse),
        'δ < 1.25': float(delta1),
        'δ < 1.25^2': float(delta2),
        'δ < 1.25^3': float(delta3),
        'valid_pixels': int(np.sum(mask))
    }


# ============================================================
# Dataset-specific evaluation functions
# ============================================================

def evaluate_scared_dataset(model, data_dir, device='cuda', img_size=512):
    """Evaluate SCARED dataset"""
    print(f"\nEvaluating SCARED dataset from: {data_dir}")

    # Load dataset paths
    images_dir = os.path.join(data_dir, 'images')
    depth_dir = os.path.join(data_dir, 'depth')
    poses_dir = os.path.join(data_dir, 'poses')

    if not os.path.exists(images_dir):
        print(f"✗ Images directory not found: {images_dir}")
        return None

    # Get image files
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.png')))
    if not image_files:
        print(f"✗ No PNG images found in {images_dir}")
        return None

    print(f"Found {len(image_files)} images, evaluating {len(image_files)} samples")

    results = []

    for idx, img_file in enumerate(image_files):
        try:
            # Extract frame information from filename
            basename = os.path.basename(img_file)
            match = re.search(r'(\d+_\d+)_frame_data(\d{6})\.png', basename)

            if not match:
                print(f"  Skipping {basename}: filename pattern not recognized")
                continue

            seq_id, frame_num = match.groups()

            # Load depth ground truth
            depth_file = os.path.join(depth_dir, f"depth_{seq_id}_frame_data{frame_num}.npz")
            if not os.path.exists(depth_file):
                print(f"  Depth file not found: {depth_file}")
                continue

            gt_depth = load_npz_depth(depth_file)
            if gt_depth is None:
                print(f"  Failed to load depth: {depth_file}")
                continue

            # Load camera pose
            pose_file = os.path.join(poses_dir, f"{seq_id}_frame_data{frame_num}.json")
            camera_pose = None
            intrinsics = None

            if os.path.exists(pose_file):
                intrinsics, camera_pose = parse_scared_pose(pose_file)

            print(f"  Processing {idx + 1}/{len(image_files)}: {basename}")

            # Run inference
            pred_depth = inference_with_correct_format(
                model, img_file, camera_pose, intrinsics, device, img_size
            )

            # Convert to numpy for evaluation
            pred_depth_np = pred_depth.numpy()

            # Evaluate
            metrics = evaluate_depth_metrics(pred_depth_np, gt_depth, max_depth=80.0)
            metrics['filename'] = basename

            results.append(metrics)

            # Save visualization
            save_depth_visualization(img_file, pred_depth_np, gt_depth, './eval_results/scared')

        except Exception as e:
            print(f"  Error processing {img_file}: {str(e)}")
            continue

    if results:
        # Calculate summary statistics
        summary = calculate_summary(results)
        print(f"\n✓ SCARED evaluation completed: {len(results)} images evaluated")
        return {'per_image': results, 'summary': summary}
    else:
        print(f"\n✗ No valid results for SCARED dataset")
        return None


def evaluate_endonerf_dataset(model, data_dir, dataset_name, device='cuda', img_size=512):
    """Evaluate EndoNeRF dataset"""
    print(f"\nEvaluating {dataset_name} dataset from: {data_dir}")

    # Load dataset paths
    images_dir = os.path.join(data_dir, 'images')
    depth_dir = os.path.join(data_dir, 'depth')

    # Check if depth directory exists
    if not os.path.exists(images_dir):
        print(f"✗ Images directory not found: {images_dir}")
        return None

    if not os.path.exists(depth_dir):
        print(f"✗ Depth directory not found: {depth_dir}")
        # Try to find depth directory with different name
        alt_depth_dir = os.path.join(data_dir, 'depth_maps')
        if os.path.exists(alt_depth_dir):
            depth_dir = alt_depth_dir
            print(f"  Using alternative depth directory: {depth_dir}")
        else:
            return None

    # Get image files
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.png')))
    if not image_files:
        print(f"✗ No PNG images found in {images_dir}")
        return None

    print(f"Found {len(image_files)} images, evaluating {len(image_files)} samples")

    results = []

    for idx, img_file in enumerate(image_files):
        try:
            basename = os.path.basename(img_file)

            # Try to find corresponding depth file
            depth_file = None

            # First, try exact match with .depth.png extension
            depth_candidate = img_file.replace('.color.png', '.depth.png').replace('.png', '.depth.png')
            if os.path.exists(depth_candidate):
                depth_file = depth_candidate
            else:
                # Try to find by frame number
                frame_match = re.search(r'(\d{6})', basename)
                if frame_match:
                    frame_num = frame_match.group(1)
                    # Try different naming patterns
                    patterns = [
                        os.path.join(depth_dir, f'frame-{frame_num}.depth.png'),
                        os.path.join(depth_dir, f'{frame_num}_depth.png'),
                        os.path.join(depth_dir, f'depth_{frame_num}.png'),
                        os.path.join(depth_dir, f'{frame_num}.png'),
                    ]

                    for pattern in patterns:
                        if os.path.exists(pattern):
                            depth_file = pattern
                            break

            if depth_file is None or not os.path.exists(depth_file):
                print(f"  Depth file not found for {basename}")
                # List available depth files for debugging
                depth_files = glob.glob(os.path.join(depth_dir, '*.png'))
                if depth_files:
                    print(f"    Available depth files: {[os.path.basename(f) for f in depth_files[:3]]}")
                continue

            # Load depth ground truth
            gt_depth = load_png_depth(depth_file)
            if gt_depth is None:
                print(f"  Failed to load depth: {depth_file}")
                continue

            print(f"  Processing {idx + 1}/{len(image_files)}: {basename}")

            # Run inference (EndoNeRF uses fixed poses)
            pred_depth = inference_with_correct_format(
                model, img_file, None, None, device, img_size
            )

            # Convert to numpy for evaluation
            pred_depth_np = pred_depth.numpy()

            # Evaluate (smaller depth range for endoscopic scenes)
            metrics = evaluate_depth_metrics(pred_depth_np, gt_depth, max_depth=10.0)
            metrics['filename'] = basename

            results.append(metrics)

            # Save visualization
            save_depth_visualization(img_file, pred_depth_np, gt_depth, f'./eval_results/{dataset_name.lower()}')

        except Exception as e:
            print(f"  Error processing {img_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if results:
        # Calculate summary statistics
        summary = calculate_summary(results)
        print(f"\n✓ {dataset_name} evaluation completed: {len(results)} images evaluated")
        return {'per_image': results, 'summary': summary}
    else:
        print(f"\n✗ No valid results for {dataset_name} dataset")
        return None


def calculate_summary(results):
    """Calculate summary statistics from results"""
    if not results:
        return {}

    summary = {}
    metric_keys = ['Abs Rel', 'Sq Rel', 'RMSE', 'Log RMSE', 'δ < 1.25', 'δ < 1.25^2', 'δ < 1.25^3']

    for key in metric_keys:
        values = [r[key] for r in results]
        summary[f'avg_{key}'] = float(np.mean(values))
        summary[f'std_{key}'] = float(np.std(values))
        summary[f'min_{key}'] = float(np.min(values))
        summary[f'max_{key}'] = float(np.max(values))

    summary['total_images'] = len(results)

    return summary


def save_depth_visualization(img_path, pred_depth, gt_depth, output_dir):
    """Save depth visualization"""
    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(img_path))[0]

    # Normalize depth maps for visualization
    def normalize_depth(depth):
        valid_mask = depth > 0
        if np.any(valid_mask):
            valid_depth = depth[valid_mask]
            vmin, vmax = np.percentile(valid_depth, [5, 95])
            depth_norm = (depth - vmin) / (vmax - vmin + 1e-8)
            depth_norm = np.clip(depth_norm, 0, 1)
            return (depth_norm * 255).astype(np.uint8)
        return np.zeros_like(depth, dtype=np.uint8)

    # Save predicted depth
    pred_viz = normalize_depth(pred_depth)
    cv2.imwrite(os.path.join(output_dir, f'{basename}_pred.png'), pred_viz)

    # Save ground truth depth
    gt_viz = normalize_depth(gt_depth)
    cv2.imwrite(os.path.join(output_dir, f'{basename}_gt.png'), gt_viz)


# ============================================================
# Main function
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Depth Evaluation for SCARED and EndoNeRF datasets')

    # Dataset paths
    parser.add_argument('--scared_dir', type=str, default='/home/bygpu/Desktop/scared/test',
                        help='SCARED dataset directory')
    parser.add_argument('--cut_dir', type=str, default='/home/bygpu/Desktop/cut/test',
                        help='EndoNeRF Cut dataset directory')
    parser.add_argument('--pull_dir', type=str, default='/home/bygpu/Desktop/pull/test',
                        help='EndoNeRF Pull dataset directory')

    # Model path
    parser.add_argument('--model_path', type=str, default='./results/test/checkpoint-best.pth',
                        help='Path to model checkpoint')

    # Evaluation parameters
    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='Patch size for model (must divide image dimensions)')

    # Output
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='Output directory')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model = load_model(args.model_path, device)
    if model is None:
        print("Failed to load model. Exiting.")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    # Evaluate SCARED dataset
    if os.path.exists(args.scared_dir):
        scared_results = evaluate_scared_dataset(
            model, args.scared_dir, device, args.img_size
        )
        if scared_results:
            all_results['SCARED'] = scared_results

            # Print SCARED summary
            print("\n" + "=" * 60)
            print("SCARED DATASET RESULTS")
            print("=" * 60)
            for key, value in scared_results['summary'].items():
                if key.startswith('avg_'):
                    metric_name = key[4:]
                    print(f"{metric_name:15s}: {value:.6f}")

    # Evaluate EndoNeRF Cut dataset
    if os.path.exists(args.cut_dir):
        cut_results = evaluate_endonerf_dataset(
            model, args.cut_dir, 'EndoNeRF_Cut', device, args.img_size
        )
        if cut_results:
            all_results['EndoNeRF_Cut'] = cut_results

            # Print Cut summary
            print("\n" + "=" * 60)
            print("EndoNeRF CUT DATASET RESULTS")
            print("=" * 60)
            for key, value in cut_results['summary'].items():
                if key.startswith('avg_'):
                    metric_name = key[4:]
                    print(f"{metric_name:15s}: {value:.6f}")

    # Evaluate EndoNeRF Pull dataset
    if os.path.exists(args.pull_dir):
        pull_results = evaluate_endonerf_dataset(
            model, args.pull_dir, 'EndoNeRF_Pull', device, args.img_size
        )
        if pull_results:
            all_results['EndoNeRF_Pull'] = pull_results

            # Print Pull summary
            print("\n" + "=" * 60)
            print("EndoNeRF PULL DATASET RESULTS")
            print("=" * 60)
            for key, value in pull_results['summary'].items():
                if key.startswith('avg_'):
                    metric_name = key[4:]
                    print(f"{metric_name:15s}: {value:.6f}")

    # Save all results to JSON
    if all_results:
        output_file = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ All results saved to: {output_file}")
    else:
        print("\n✗ No results were generated")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()