# --------------------------------------------------------
# Inference code for DUSt3R with fixed camera poses support
# --------------------------------------------------------
import os
import torch
import tqdm
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf
from dust3r.viz import SceneViz, auto_cam_size
from dust3r.utils.image import rgb
from dust3r.model_noposes import AsymmetricCroCo3DStereo, inf


def _interleave_imgs(img1: Dict, img2: Dict) -> Dict:
    """
    Interleave data from two views for symmetric processing.

    Args:
        img1: First view data
        img2: Second view data

    Returns:
        Interleaved data
    """
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor) and value1.ndim == value2.ndim:
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(batch: Tuple[Dict, Dict]) -> Tuple[Dict, Dict]:
    """
    Make batch symmetric by interleaving views.

    Args:
        batch: Tuple of (view1, view2)

    Returns:
        Symmetric batch
    """
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2


def visualize_fixed_pose_results(view1: Dict, view2: Dict, pred1: Dict, pred2: Dict,
                                 save_dir: str = './tmp', save_name: Optional[str] = None,
                                 visualize_type: str = 'pred', use_known_poses: bool = True) -> str:
    """
    Visualize inference results with fixed camera poses.

    Args:
        view1: First view data
        view2: Second view data
        pred1: Predictions for first view
        pred2: Predictions for second view
        save_dir: Directory to save visualization
        save_name: Base name for saved file
        visualize_type: 'pred' for predictions or 'gt' for ground truth
        use_known_poses: Whether to use known camera poses

    Returns:
        Path to saved visualization file
    """
    viz = SceneViz()
    views = [view1, view2]

    # Get camera poses
    if use_known_poses:
        poses = [views[view_idx]['camera_pose'][0] for view_idx in [0, 1]]
    else:
        poses = [pred1['camera_pose'][0], pred2['camera_pose'][0]]

    cam_size = max(auto_cam_size(poses), 0.5)

    if visualize_type == 'pred':
        cam_size *= 0.1

    for view_idx in [0, 1]:
        if visualize_type == 'pred':
            if view_idx == 0:
                pts3d = pred1['pts3d'][0]
            else:
                pts3d = pred2['pts3d_in_other_view'][0]

            if use_known_poses:
                pts3d = geotrf(poses[view_idx], pts3d)
        else:
            if 'pts3d' in views[view_idx]:
                pts3d = views[view_idx]['pts3d'][0]
            else:
                pts3d = pred1['pts3d'][0] if view_idx == 0 else pred2['pts3d_in_other_view'][0]
                if use_known_poses:
                    pts3d = geotrf(poses[view_idx], pts3d)

        valid_mask = None
        if 'valid_mask' in views[view_idx]:
            valid_mask = views[view_idx]['valid_mask'][0]
        elif 'mask' in views[view_idx]:
            valid_mask = views[view_idx]['mask'][0]

        colors = rgb(views[view_idx]['img'][0])

        if valid_mask is not None:
            viz.add_pointcloud(pts3d, colors, valid_mask)
        else:
            viz.add_pointcloud(pts3d, colors)

        viz.add_camera(
            pose_c2w=poses[view_idx],
            focal=views[view_idx]['camera_intrinsics'][0, 0],
            color=(255, 0, 0),
            image=colors,
            cam_size=cam_size
        )

    os.makedirs(save_dir, exist_ok=True)

    if save_name is None:
        save_name = f'{views[0]["dataset"][0]}_{views[0]["label"][0]}_{views[0]["instance"][0]}_{views[1]["instance"][0]}_{visualize_type}_fixed_pose'

    save_path = os.path.join(save_dir, f'{save_name}.glb')
    print(f'Saving visualization to {save_path}')

    return viz.save_glb(save_path)


def loss_of_one_batch(batch: Tuple[Dict, Dict], model: torch.nn.Module, criterion: Optional[torch.nn.Module],
                      device: torch.device, symmetrize_batch: bool = False, use_amp: bool = False,
                      use_known_poses: bool = False, ret: Optional[str] = None) -> Dict:
    """
    Process one batch for inference or loss computation with known poses support.

    Args:
        batch: Input batch (view1, view2)
        model: Model to use for inference
        criterion: Loss criterion (optional)
        device: Target device
        symmetrize_batch: Whether to symmetrize the batch
        use_amp: Whether to use automatic mixed precision
        use_known_poses: Whether to use known poses mode
        ret: Key to return from result (if None, return full result)

    Returns:
        Inference results or loss
    """
    view1, view2 = batch
    ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'])

    # Move data to device
    for view in [view1, view2]:
        for name in view.keys():
            if name in ignore_keys:
                continue
            if isinstance(view[name], torch.Tensor):
                view[name] = view[name].to(device, non_blocking=True)

    if symmetrize_batch:
        view1, view2 = make_batch_symmetric((view1, view2))

    # Forward pass
    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        pred1, pred2 = model(view1, view2)

        # Compute loss if criterion is provided
        loss = None
        if criterion is not None:
            with torch.cuda.amp.autocast(enabled=False):
                loss = criterion(view1, view2, pred1, pred2) if criterion is not None else None

    result = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2, loss=loss)

    return result[ret] if ret else result


def inference_with_fixed_poses(model: torch.nn.Module, pairs: List[Tuple[Dict, Dict]],
                               device: torch.device, batch_size: int = 8, verbose: bool = True,
                               use_known_poses: bool = True, save_visualizations: bool = False,
                               save_dir: str = './results') -> Tuple[Dict, List[str]]:
    """
    Perform inference with fixed camera poses.

    Args:
        model: Trained model with fixed pose support
        pairs: List of image pairs [(view1, view2), ...] with camera poses
        device: Computation device
        batch_size: Batch size for inference
        verbose: Whether to show progress bar
        use_known_poses: Whether to use known camera poses
        save_visualizations: Whether to save 3D visualizations
        save_dir: Directory to save results

    Returns:
        tuple: (inference_results, visualization_paths)
    """
    if verbose:
        print(f'>> Fixed-pose inference on {len(pairs)} image pairs')

    result = []
    all_visualizations = []

    # Check if all images have the same size
    multiple_shapes = not check_if_same_size(pairs)
    if multiple_shapes:
        batch_size = 1
        if verbose:
            print('Warning: Multiple image shapes detected, forcing batch_size=1')

    # Process in batches
    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        batch = pairs[i:i + batch_size]

        # Collate batch
        view1_list, view2_list = zip(*batch)
        batch_collated = collate_with_cat([view1_list, view2_list])

        # Process batch
        batch_result = loss_of_one_batch(
            batch_collated, model, None, device,
            symmetrize_batch=False,
            use_amp=False,
            use_known_poses=use_known_poses
        )

        # Move results to CPU
        batch_result = {k: to_cpu(v) if isinstance(v, torch.Tensor) else v
                        for k, v in batch_result.items()}

        result.append(batch_result)

        # Save visualizations if requested
        if save_visualizations and i == 0:
            for j in range(min(batch_size, len(batch))):
                # Extract single sample
                single_view1 = {}
                single_view2 = {}
                single_pred1 = {}
                single_pred2 = {}

                view1_data = batch_result['view1']
                view2_data = batch_result['view2']
                pred1_data = batch_result['pred1']
                pred2_data = batch_result['pred2']

                for key in view1_data.keys():
                    if isinstance(view1_data[key], torch.Tensor) and view1_data[key].ndim > 0:
                        single_view1[key] = view1_data[key][j:j + 1]
                        single_view2[key] = view2_data[key][j:j + 1]

                for key in pred1_data.keys():
                    if isinstance(pred1_data[key], torch.Tensor) and pred1_data[key].ndim > 0:
                        single_pred1[key] = pred1_data[key][j:j + 1]
                        single_pred2[key] = pred2_data[key][j:j + 1]

                # Create visualization
                viz_path = visualize_fixed_pose_results(
                    single_view1, single_view2, single_pred1, single_pred2,
                    save_dir=save_dir,
                    save_name=f'sample_{i + j}',
                    visualize_type='pred',
                    use_known_poses=use_known_poses
                )
                all_visualizations.append(viz_path)

    # Combine results from all batches
    final_result = {}
    if result:
        for key in result[0].keys():
            if all(key in res for res in result):
                # Concatenate tensors
                if isinstance(result[0][key], torch.Tensor):
                    final_result[key] = torch.cat([res[key] for res in result], dim=0)

    return final_result, all_visualizations


def check_if_same_size(pairs: List[Tuple[Dict, Dict]]) -> bool:
    """
    Check if all image pairs have the same dimensions.

    Args:
        pairs: List of image pairs

    Returns:
        True if all images have same dimensions, False otherwise
    """
    if not pairs:
        return True

    shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]

    return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)


def get_fixed_pose_prediction(pred: Dict, camera_pose: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Extract 3D point cloud from prediction for fixed-pose setup.

    Args:
        pred: Prediction dictionary
        camera_pose: Optional camera pose for coordinate transformation

    Returns:
        3D point cloud
    """
    if 'pts3d' in pred:
        pts3d = pred['pts3d']
    elif 'pts3d_in_other_view' in pred:
        pts3d = pred['pts3d_in_other_view']
    elif 'depth' in pred and 'pseudo_focal' in pred:
        pts3d = depthmap_to_pts3d(pred['depth'], pred['pseudo_focal'])
    else:
        raise ValueError("No 3D point information available in prediction")

    if camera_pose is not None:
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


def load_model_for_inference(model_path: str, model_config: str, device: torch.device,
                             use_known_poses: bool = True) -> torch.nn.Module:
    """
    Load model for inference with fixed poses support.

    Args:
        model_path: Path to model checkpoint
        model_config: Model configuration string
        device: Target device
        use_known_poses: Whether to use known poses mode

    Returns:
        Loaded model
    """
    print(f'Loading model: {model_config}')

    # Modify model configuration for known poses mode
    if use_known_poses:
        if 'use_known_poses=' not in model_config:
            if ')' in model_config:
                model_config = model_config.replace(')', f', use_known_poses=True, pose_input_key="camera_pose")')
            else:
                model_config = f'{model_config[:-1]}, use_known_poses=True, pose_input_key="camera_pose")'

    print(f'Actual model configuration: {model_config}')

    # Instantiate model
    model = eval(model_config)
    model.to(device)

    # Load checkpoint
    if model_path and os.path.exists(model_path):
        print(f'Loading checkpoint: {model_path}')
        checkpoint = torch.load(model_path, map_location=device)

        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        print('Model loaded successfully')

    model.eval()
    return model


def prepare_pairs_for_inference(image_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
                                camera_poses: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                                camera_intrinsics: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> List[
    Tuple[Dict, Dict]]:
    """
    Prepare image pairs for inference with fixed poses.

    Args:
        image_pairs: List of (image1, image2) tensors
        camera_poses: Optional list of (pose1, pose2) tensors
        camera_intrinsics: Optional list of (intrinsics1, intrinsics2) tensors

    Returns:
        List of prepared pairs for inference
    """
    pairs = []

    for idx, (img1, img2) in enumerate(image_pairs):
        view1 = {'img': img1}
        view2 = {'img': img2}

        # Add camera poses if provided
        if camera_poses is not None:
            pose1, pose2 = camera_poses[idx]
            view1['camera_pose'] = pose1
            view2['camera_pose'] = pose2

        # Add camera intrinsics if provided
        if camera_intrinsics is not None:
            intrinsics1, intrinsics2 = camera_intrinsics[idx]
            view1['camera_intrinsics'] = intrinsics1
            view2['camera_intrinsics'] = intrinsics2

        # Add metadata
        view1['dataset'] = 'inference'
        view2['dataset'] = 'inference'
        view1['label'] = f'pair_{idx}'
        view2['label'] = f'pair_{idx}'
        view1['instance'] = f'image1_{idx}'
        view2['instance'] = f'image2_{idx}'

        pairs.append((view1, view2))

    return pairs


def main_inference(args):
    """
    Main inference function with fixed poses support.

    Args:
        args: Command line arguments or configuration dict
    """
    import argparse

    # Parse arguments if provided as dict
    if isinstance(args, dict):
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        args = Args(**args)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model
    model = load_model_for_inference(
        model_path=args.pretrained if hasattr(args, 'pretrained') else None,
        model_config=args.model if hasattr(args,
                                           'model') else "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')",
        device=device,
        use_known_poses=getattr(args, 'use_known_poses', True)
    )

    # Prepare data (example - you need to implement your own data loading)
    # image_pairs = load_your_image_pairs(args.input_dir)
    # camera_poses = load_your_camera_poses(args.pose_dir) if args.use_known_poses else None
    # camera_intrinsics = load_your_camera_intrinsics(args.calib_dir)
    #
    # pairs = prepare_pairs_for_inference(image_pairs, camera_poses, camera_intrinsics)

    # Perform inference
    # results, visualizations = inference_with_fixed_poses(
    #     model=model,
    #     pairs=pairs,
    #     device=device,
    #     batch_size=getattr(args, 'batch_size', 8),
    #     verbose=True,
    #     use_known_poses=getattr(args, 'use_known_poses', True),
    #     save_visualizations=getattr(args, 'save_visualizations', True),
    #     save_dir=getattr(args, 'output_dir', './inference_results')
    # )

    # Save results
    # save_results(results, args.output_dir)

    print('Inference completed')


# 在 training_noposes.py 的合适位置添加这个函数
def visualize_results(view1, view2, pred1, pred2, save_dir='./tmp',
                      save_name=None, visualize_type='gt'):
    """Simple visualization function for debugging."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    # 创建一个简单的文本文件表示我们保存了可视化
    if save_name is None:
        save_name = 'visualization'

    save_path = os.path.join(save_dir, f'{save_name}.txt')

    with open(save_path, 'w') as f:
        f.write(f'Visualization type: {visualize_type}\n')
        f.write(f'View1 shape: {view1["img"].shape}\n')
        f.write(f'View2 shape: {view2["img"].shape}\n')
        f.write('Visualization saved (placeholder)\n')

    print(f'Visualization placeholder saved to {save_path}')
    return save_path


def get_args_parser_for_inference():
    """
    Get argument parser for inference.

    Returns:
        Argument parser
    """
    parser = argparse.ArgumentParser('DUST3R inference with fixed poses', add_help=False)

    # Model parameters
    parser.add_argument('--model',
                        default="AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')",
                        type=str, help="Model configuration string")

    # Known poses parameters
    parser.add_argument('--use_known_poses', action='store_true', default=True,
                        help='Use known camera poses for inference')
    parser.add_argument('--pose_input_key', default='camera_pose', type=str,
                        help='Key name for pose in input data')

    # Input/output parameters
    parser.add_argument('--pretrained', default=None, type=str,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--pose_dir', type=str, default=None,
                        help='Directory containing camera poses (required if use_known_poses=True)')
    parser.add_argument('--calib_dir', type=str, default=None,
                        help='Directory containing camera calibration')
    parser.add_argument('--output_dir', default='./inference_results', type=str,
                        help="Directory to save output")

    # Inference parameters
    parser.add_argument('--batch_size', default=8, type=int,
                        help="Batch size for inference")
    parser.add_argument('--save_visualizations', action='store_true', default=True,
                        help='Save 3D visualizations')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of data loading workers')

    return parser


if __name__ == "__main__":
    # Example usage
    parser = get_args_parser_for_inference()
    args = parser.parse_args()

    main_inference(args)