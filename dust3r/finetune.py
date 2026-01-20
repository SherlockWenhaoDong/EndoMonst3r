# --------------------------------------------------------
# optimization code for DUSt3R with known poses (Fixed Version)
# --------------------------------------------------------
import os

os.environ['OMP_NUM_THREADS'] = '4'
import argparse
import datetime
import json
import numpy as np
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True

from dust3r.model import AsymmetricCroCo3DStereo, inf
from dust3r.datasets import get_data_loader
from dust3r.losses import *
from dust3r.inference import loss_of_one_batch, visualize_results

import dust3r.utils.path_to_croco
import croco.utils.misc as misc
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging to wandb disabled.")


def get_args_parser():
    parser = argparse.ArgumentParser('DUSt3R optimization with known poses', add_help=False)

    # model and checkpoint
    parser.add_argument('--model', default="AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', \
                        img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
                        enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)",
                        type=str, help="model architecture")
    parser.add_argument('--checkpoint', required=True, type=str, help='path to checkpoint for optimization')

    # dataset with known poses
    parser.add_argument('--dataset', default='[CO3D(split="train", max_len=1000, resolution=224, augment=False)]',
                        type=str, help="dataset with known poses")

    # optimization
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=8, type=int, help="Batch size per GPU")
    parser.add_argument('--accum_iter', default=1, type=int, help="Accumulate gradient iterations")
    parser.add_argument('--epochs', default=50, type=int, help="Number of optimization epochs")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="weight decay")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="minimum learning rate")
    parser.add_argument('--warmup_epochs', type=int, default=5, help="epochs to warmup LR")
    parser.add_argument('--amp', type=int, default=1, choices=[0, 1], help="Use Automatic Mixed Precision")
    parser.add_argument('--grad_clip', type=float, default=1.0, help="Gradient clipping value")

    # freeze settings
    parser.add_argument('--freeze_encoder', action='store_true', default=True,
                        help="freeze encoder parameters")
    parser.add_argument('--freeze_pos_embed', action='store_true', default=True,
                        help="freeze positional embeddings")
    parser.add_argument('--freeze_patch_embed', action='store_true', default=True,
                        help="freeze patch embedding")
    parser.add_argument('--train_only_decoder', action='store_true', default=False,
                        help="train only decoder parameters")
    parser.add_argument('--train_only_head', action='store_true', default=False,
                        help="train only head parameters")

    # loss settings
    parser.add_argument('--train_criterion', default="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)",
                        type=str, help="train criterion")
    parser.add_argument('--test_criterion', default=None, type=str, help="test criterion")

    # data settings
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--use_gt_intrinsics', action='store_true', default=True,
                        help="use ground truth camera intrinsics")
    parser.add_argument('--use_gt_poses', action='store_true', default=True,
                        help="use ground truth camera poses")
    parser.add_argument('--fixed_eval_set', action='store_true', default=False,
                        help="use fixed evaluation set (don't shuffle)")

    # memory management
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help="use mixed precision training")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="number of gradient accumulation steps")
    parser.add_argument('--skip_ooms', action='store_true', default=True,
                        help="skip batches that cause OOM errors")

    # distributed training
    parser.add_argument('--distributed', action='store_true', default=False,
                        help="use distributed training")
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # output
    parser.add_argument('--output_dir', required=True, type=str, help="path to save optimized model")
    parser.add_argument('--save_freq', default=5, type=int, help="save frequency in epochs")
    parser.add_argument('--eval_freq', default=1, type=int, help="evaluation frequency")
    parser.add_argument('--print_freq', default=20, type=int, help="print frequency")
    parser.add_argument('--wandb', action='store_true', default=False, help="use wandb for logging")
    parser.add_argument('--num_save_visual', default=2, type=int, help="number of visualizations to save")
    parser.add_argument('--no_wandb_offline', action='store_true', default=False,
                        help="disable wandb offline mode")

    return parser


def freeze_model_parameters(model, args):
    """Freeze specified parameters of the model"""
    total_params = 0
    trainable_params = 0

    print("\n=== Parameter Freezing ===")

    # 首先解冻所有参数
    for param in model.parameters():
        param.requires_grad = True

    # 然后根据参数冻结
    for name, param in model.named_parameters():
        total_params += param.numel()

        # 默认不解冻
        freeze = True  # 默认冻结所有参数

        # 根据条件判断哪些参数应该解冻
        if args.freeze_encoder:
            # 冻结encoder，但解冻decoder和head
            if 'encoder' not in name:
                freeze = False
        else:
            # 不解冻encoder
            if 'encoder' in name:
                freeze = False

        if args.freeze_pos_embed:
            if 'pos_embed' in name or 'rope' in name.lower():
                freeze = True
            else:
                # 如果不是pos_embed，保持原状
                pass

        if args.freeze_patch_embed:
            if 'patch_embed' in name:
                freeze = True
            else:
                # 如果不是patch_embed，保持原状
                pass

        if args.train_only_decoder:
            # 只训练decoder
            if 'decoder' in name:
                freeze = False
            else:
                freeze = True

        if args.train_only_head:
            # 只训练输出头
            if 'head' in name or 'proj' in name or 'pred' in name:
                freeze = False
            else:
                freeze = True

        # 如果所有冻结选项都是默认值（True），那么只训练头部
        if (args.freeze_encoder and args.freeze_pos_embed and args.freeze_patch_embed and
                not args.train_only_decoder and not args.train_only_head):
            # 默认设置：只训练头部
            if 'head' in name or 'proj' in name or 'pred' in name:
                freeze = False
            else:
                freeze = True

        # 应用冻结/解冻
        param.requires_grad = not freeze
        if param.requires_grad:
            trainable_params += param.numel()

    # 打印详细的可训练参数信息
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    # 打印可训练的参数名
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}")

    return model


def load_checkpoint(model, checkpoint_path, device):
    """Load checkpoint and handle compatibility"""
    print(f"\n=== Loading Checkpoint ===")
    print(f"Checkpoint path: {checkpoint_path}")

    try:
        if checkpoint_path.endswith('.pth'):
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("Found 'model' in checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("Found 'state_dict' in checkpoint")
        else:
            state_dict = checkpoint
            print("Using checkpoint directly as state_dict")

        # Remove 'module.' prefix if present (from DDP)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            print("Removed 'module.' prefix from state_dict")

        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"\nMissing keys ({len(missing_keys)}):")
            for key in missing_keys[:10]:  # Show first 10
                print(f"  - {key}")
            if len(missing_keys) > 10:
                print(f"  ... and {len(missing_keys) - 10} more")

        if unexpected_keys:
            print(f"\nUnexpected keys ({len(unexpected_keys)}):")
            for key in unexpected_keys[:10]:  # Show first 10
                print(f"  - {key}")
            if len(unexpected_keys) > 10:
                print(f"  ... and {len(unexpected_keys) - 10} more")

        print(f"\nCheckpoint loaded successfully")

        # Load epoch and other info if available
        epoch = 0
        best_so_far = float('inf')
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
            print(f"Checkpoint epoch: {epoch}")
        if 'best_so_far' in checkpoint:
            best_so_far = checkpoint['best_so_far']
            print(f"Best loss: {best_so_far}")

        return model, checkpoint, epoch, best_so_far

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise


def set_dataset_epoch(data_loader, epoch, fixed_eval_set=False):
    """Set epoch for dataset (required for ResizedDataset)"""
    if hasattr(data_loader, 'dataset'):
        if hasattr(data_loader.dataset, 'set_epoch'):
            if fixed_eval_set:
                data_loader.dataset.set_epoch(0)  # Fixed for evaluation
            else:
                data_loader.dataset.set_epoch(epoch)

    if hasattr(data_loader, 'sampler'):
        if hasattr(data_loader.sampler, 'set_epoch'):
            if fixed_eval_set:
                data_loader.sampler.set_epoch(0)  # Fixed for evaluation
            else:
                data_loader.sampler.set_epoch(epoch)


class SafeMetricLogger:
    """Custom MetricLogger that handles zero division errors"""

    def __init__(self, delimiter="  "):
        self.delimiter = delimiter
        self.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9 ** 9))

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __str__(self):
        if len(self.meters) == 0:
            return ""

        # Build safe string representation
        parts = []
        for name, meter in self.meters.items():
            # Safely get meter value
            try:
                if hasattr(meter, 'count') and meter.count > 0:
                    value = meter.global_avg
                    parts.append("{}: {:.4f}".format(name, value))
                else:
                    parts.append("{}: N/A".format(name))
            except ZeroDivisionError:
                parts.append("{}: N/A".format(name))
            except Exception:
                parts.append("{}: ERR".format(name))

        return self.delimiter.join(parts)

    def global_avg_safe(self, name):
        """Safely get global average"""
        if name not in self.meters:
            return 0.0

        meter = self.meters[name]
        try:
            if hasattr(meter, 'count') and meter.count > 0:
                return meter.global_avg
        except ZeroDivisionError:
            pass
        return 0.0

    def synchronize_between_processes(self):
        # For now, just pass. In distributed setting, implement proper sync
        pass


def optimize_one_epoch(model, criterion, data_loader, optimizer, device,
                       epoch, loss_scaler, args, log_writer=None):
    """Optimize for one epoch with known poses (fixes OOM issues)"""
    model.train()
    metric_logger = SafeMetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Optimize Epoch: [{epoch + 1}/{args.epochs}]'
    accum_iter = args.accum_iter

    # Set dataset epoch
    set_dataset_epoch(data_loader, epoch, args.fixed_eval_set)

    optimizer.zero_grad()

    # Counter to track successful batches
    successful_batches = 0
    total_batches = len(data_loader)

    # Create data loader iterator
    data_loader_iter = iter(data_loader)

    for data_iter_step in range(total_batches):
        try:
            # Get batch
            batch = next(data_loader_iter)
        except StopIteration:
            break

        epoch_f = epoch + data_iter_step / total_batches

        # Adjust learning rate
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)

        # Clear cache to reduce memory usage
        torch.cuda.empty_cache()

        # Forward pass using loss_of_one_batch (handles everything)
        try:
            batch_result = loss_of_one_batch(
                batch, model, criterion, device,
                symmetrize_batch=True,
                use_amp=bool(args.amp)
            )
        except RuntimeError as e:
            if "out of memory" in str(e) and args.skip_ooms:
                print(f"OOM at iteration {data_iter_step}, skipping batch")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        # Extract loss
        if 'loss' in batch_result:
            loss_tuple = batch_result['loss']
            if isinstance(loss_tuple, tuple) and len(loss_tuple) == 2:
                loss, loss_details = loss_tuple
            else:
                loss = loss_tuple
                loss_details = {}
        else:
            # Fallback to zero loss
            loss = torch.tensor(0.0, device=device)
            loss_details = {}

        loss_value = float(loss)

        # Save visualizations (optional)
        if (args.num_save_visual > 0 and
                (data_iter_step % max((total_batches // args.num_save_visual), 1) == 0) and
                misc.is_main_process()):
            try:
                save_dir = Path(args.output_dir) / f'epoch_{epoch + 1}'
                save_dir.mkdir(parents=True, exist_ok=True)

                # Extract views and predictions from batch_result
                if all(key in batch_result for key in ['view1', 'view2', 'pred1', 'pred2']):
                    view1 = batch_result['view1']
                    view2 = batch_result['view2']
                    pred1 = batch_result['pred1']
                    pred2 = batch_result['pred2']

                    # Visualize results
                    save_name = f'train_batch_{data_iter_step}'
                    try:
                        gt_visual_path = visualize_results(
                            view1, view2, pred1, pred2,
                            save_dir=str(save_dir), save_name=save_name + '_gt', visualize_type='gt')
                        pred_visual_path = visualize_results(
                            view1, view2, pred1, pred2,
                            save_dir=str(save_dir), save_name=save_name + '_pred', visualize_type='pred')

                        # Log to wandb if available and enabled
                        if args.wandb and WANDB_AVAILABLE:
                            try:
                                wandb.log({
                                    'epoch': epoch,
                                    'optim_visual_gt': wandb.Image(str(gt_visual_path)),
                                    'optim_visual_pred': wandb.Image(str(pred_visual_path))
                                })
                            except:
                                pass
                    except Exception as viz_error:
                        print(f"Error in visualization: {viz_error}")
            except Exception as e:
                print(f"Error saving visualizations: {e}")

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping optimization")
            sys.exit(1)

        # Backward pass with memory management
        try:
            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0,
                        clip_grad=args.grad_clip)
        except RuntimeError as e:
            if "out of memory" in str(e) and args.skip_ooms:
                print(f"OOM during backward at iteration {data_iter_step}, skipping")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # Logging - only update for successful batches
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value)
        successful_batches += 1

        # Add loss details
        if loss_details:
            for name, val in loss_details.items():
                if val is not None:
                    try:
                        metric_logger.update(**{f'{name}_loss': float(val)})
                    except:
                        pass

        # Tensorboard logging
        if log_writer is not None and misc.is_main_process():
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar('train/loss', loss_value, epoch_1000x)
            log_writer.add_scalar('train/lr', lr, epoch_1000x)
            if loss_details:
                for name, val in loss_details.items():
                    if val is not None:
                        try:
                            log_writer.add_scalar(f'train/{name}_loss', float(val), epoch_1000x)
                        except:
                            pass

        # Manual logging every print_freq
        if (data_iter_step + 1) % args.print_freq == 0 or data_iter_step == total_batches - 1:
            # Build log string manually to avoid MetricLogger.__str__ issues
            log_str = header + f' [{data_iter_step}/{total_batches}]'
            try:
                stats_str = str(metric_logger)
                if stats_str:
                    log_str += ' ' + stats_str
            except ZeroDivisionError:
                log_str += ' [skipped all batches due to OOM]'
            print(log_str)

    # Fix ZeroDivisionError: check if any batches were successful
    if successful_batches == 0:
        print(f"Warning: No successful batches processed in epoch {epoch + 1}")
        # Return default values
        stats = {'loss': float('inf'), 'lr': optimizer.param_groups[0]["lr"], 'epoch': epoch + 1}
        return stats

    print(f"\nEpoch {epoch + 1} optimization stats:")
    stats = {}
    for k in metric_logger.meters.keys():
        try:
            stats[k] = metric_logger.global_avg_safe(k)
            print(f"  {k}: {stats[k]:.6f}")
        except:
            stats[k] = 0.0
            print(f"  {k}: 0.000000 (no data)")

    return stats


@torch.no_grad()
def evaluate_one_epoch(model, criterion, data_loader, device, epoch, args):
    """Evaluate the model with known poses"""
    model.eval()
    metric_logger = SafeMetricLogger(delimiter="  ")
    header = f'Evaluate Epoch: [{epoch + 1}/{args.epochs}]'

    # Set dataset epoch (use fixed epoch for evaluation if specified)
    set_dataset_epoch(data_loader, 0 if args.fixed_eval_set else epoch, args.fixed_eval_set)

    # Add counter
    successful_batches = 0
    total_batches = len(data_loader)

    # Create data loader iterator
    data_loader_iter = iter(data_loader)

    for data_iter_step in range(total_batches):
        try:
            # Get batch
            batch = next(data_loader_iter)
        except StopIteration:
            break

        # Clear cache
        torch.cuda.empty_cache()

        # Evaluate using loss_of_one_batch
        try:
            batch_result = loss_of_one_batch(
                batch, model, criterion, device,
                symmetrize_batch=True,
                use_amp=bool(args.amp)
            )
        except RuntimeError as e:
            if "out of memory" in str(e) and args.skip_ooms:
                print(f"OOM at evaluation iteration {data_iter_step}, skipping batch")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        # Extract loss
        if 'loss' in batch_result:
            loss_tuple = batch_result['loss']
            if isinstance(loss_tuple, tuple) and len(loss_tuple) == 2:
                loss, loss_details = loss_tuple
            else:
                loss = loss_tuple
                loss_details = {}
        else:
            loss = torch.tensor(0.0, device=device)
            loss_details = {}

        metric_logger.update(loss=float(loss))
        successful_batches += 1

        # Add loss details
        if loss_details:
            for name, val in loss_details.items():
                if val is not None:
                    try:
                        metric_logger.update(**{f'{name}_loss': float(val)})
                    except:
                        pass

        # Manual logging every print_freq
        if (data_iter_step + 1) % args.print_freq == 0 or data_iter_step == total_batches - 1:
            # Build log string manually to avoid MetricLogger.__str__ issues
            log_str = header + f' [{data_iter_step}/{total_batches}]'
            try:
                stats_str = str(metric_logger)
                if stats_str:
                    log_str += ' ' + stats_str
            except ZeroDivisionError:
                log_str += ' [skipped all batches due to OOM]'
            print(log_str)

    # Check if any batches were successful
    if successful_batches == 0:
        print(f"Warning: No successful batches evaluated in epoch {epoch + 1}")
        return {'loss': float('inf')}

    stats = {}
    for k in metric_logger.meters.keys():
        try:
            stats[k] = metric_logger.global_avg_safe(k)
        except:
            stats[k] = float('inf')

    return stats


def main(args):
    """Main optimization function"""
    # Initialize distributed mode if needed
    if args.distributed:
        misc.init_distributed_mode(args)

    global_rank = misc.get_rank()
    misc.is_main_process = lambda: global_rank == 0

    # Initialize wandb if requested and available
    if args.wandb and WANDB_AVAILABLE and misc.is_main_process():
        try:
            wandb.init(
                name=f"optim_{Path(args.output_dir).name}",
                project="dust3r_optimization",
                config=args,
                dir=args.output_dir,
                resume="allow"
            )
            print("WandB initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            args.wandb = False
    elif args.wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available")
        args.wandb = False

    # Create output directory
    if args.output_dir and misc.is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Save arguments
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Device Info ===")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        # Print GPU memory info
        print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")

    # Fix seed
    seed = args.seed + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = True

    # Load model
    print(f"\n=== Building Model ===")
    print(f"Model config: {args.model[:100]}...")
    model = eval(args.model)
    model.to(device)

    # Load checkpoint
    model, checkpoint, checkpoint_epoch, best_so_far = load_checkpoint(model, args.checkpoint, device)

    # Freeze specified parameters - FIXED!
    model = freeze_model_parameters(model, args)

    # Prepare model for distributed training
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu] if torch.cuda.is_available() else None,
            find_unused_parameters=True)
        model_without_ddp = model.module

    # Create dataset
    print(f"\n=== Building Dataset ===")
    print(f"Dataset config: {args.dataset}")
    data_loader = get_data_loader(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_mem=torch.cuda.is_available(),
        shuffle=True,
        drop_last=True
    )

    print(f"Dataset length: {len(data_loader)}")
    if len(data_loader) == 0:
        print("Warning: Dataset is empty!")
        return

    # Create criterion
    print(f"\n=== Creating Criterion ===")
    criterion_str = args.train_criterion
    print(f"Using criterion: {criterion_str}")
    criterion = eval(criterion_str)
    criterion.to(device)

    # Create optimizer (only for trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        print("Warning: No trainable parameters! Check freezing settings.")

    print(f"\n=== Creating Optimizer ===")
    print(f"Learning rate: {args.lr}")
    print(f"Trainable parameters: {len(trainable_params)}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    loss_scaler = NativeScaler()

    # Resume from checkpoint if available
    start_epoch = checkpoint_epoch

    # Load optimizer state if available
    if 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded optimizer state from checkpoint")
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")

    # Load loss scaler state if available
    if 'loss_scaler' in checkpoint:
        try:
            loss_scaler.load_state_dict(checkpoint['loss_scaler'])
            print("Loaded loss scaler state from checkpoint")
        except Exception as e:
            print(f"Warning: Could not load loss scaler state: {e}")

    print(f"\nResuming from epoch {start_epoch}")
    print(f"Previous best loss: {best_so_far}")

    # Create tensorboard writer
    if misc.is_main_process() and args.output_dir:
        log_writer = SummaryWriter(log_dir=args.output_dir)
        print(f"\nTensorBoard logs at: {args.output_dir}")
    else:
        log_writer = None

    # Start optimization
    print(f"\n=== Starting Optimization ===")
    print(f"Total epochs: {args.epochs}")
    print(f"Start epoch: {start_epoch}")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'=' * 50}")

        # Evaluate
        if (epoch - start_epoch) % args.eval_freq == 0:
            print(f"\n[Evaluation]")
            eval_stats = evaluate_one_epoch(
                model, criterion, data_loader, device, epoch, args
            )

            if misc.is_main_process():
                print(f"Evaluation results:")
                for k, v in eval_stats.items():
                    print(f"  {k}: {v:.6f}")

                # Log to wandb
                if args.wandb and WANDB_AVAILABLE:
                    try:
                        wandb_log = {f"eval/{k}": v for k, v in eval_stats.items()}
                        wandb_log['epoch'] = epoch
                        wandb.log(wandb_log)
                    except:
                        pass

                # Save best model
                if 'loss' in eval_stats and eval_stats['loss'] < best_so_far:
                    best_so_far = eval_stats['loss']
                    print(f"\nNew best loss: {best_so_far:.6f}")
                    save_checkpoint(
                        args, epoch, model_without_ddp, optimizer,
                        loss_scaler, best_so_far, "best"
                    )

        # Optimize
        print(f"\n[Optimization]")
        train_stats = optimize_one_epoch(
            model, criterion, data_loader, optimizer,
            device, epoch, loss_scaler, args, log_writer
        )

        # Save checkpoint
        if misc.is_main_process() and ((epoch - start_epoch) % args.save_freq == 0 or epoch == args.epochs - 1):
            save_checkpoint(
                args, epoch, model_without_ddp, optimizer,
                loss_scaler, best_so_far, f"checkpoint_{epoch + 1:04d}"
            )

        # Log to wandb
        if args.wandb and WANDB_AVAILABLE and misc.is_main_process():
            try:
                wandb_log = {f"train/{k}": v for k, v in train_stats.items()}
                wandb_log['epoch'] = epoch
                wandb.log(wandb_log)
            except:
                pass

    # Save final model
    if misc.is_main_process():
        print(f"\n=== Saving Final Model ===")
        save_checkpoint(
            args, args.epochs, model_without_ddp, optimizer,
            loss_scaler, best_so_far, "final"
        )

        # Also save just the model weights
        model_weights_path = Path(args.output_dir) / "model_final.pth"
        torch.save(model_without_ddp.state_dict(), model_weights_path)
        print(f"Model weights saved to: {model_weights_path}")

    total_time = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"Optimization completed!")
    print(f"Total time: {datetime.timedelta(seconds=int(total_time))}")
    print(f"Best loss: {best_so_far:.6f}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'=' * 50}")

    if log_writer is not None:
        log_writer.close()


def save_checkpoint(args, epoch, model, optimizer, loss_scaler, best_loss, name):
    """Save checkpoint"""
    checkpoint_path = Path(args.output_dir) / f"{name}.pth"

    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_scaler': loss_scaler.state_dict(),
        'args': args,
        'best_so_far': best_loss,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DUSt3R optimization with known poses',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("DUST3R OPTIMIZATION (Fixed Version)")
    print("=" * 60)

    # Print memory optimization settings
    print("\n=== Memory Optimization Settings ===")
    print(f"Skip OOM batches: {args.skip_ooms}")
    print(f"Gradient clipping: {args.grad_clip}")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")

    main(args)