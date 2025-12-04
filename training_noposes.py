# --------------------------------------------------------
# Training code for DUSt3R with known camera poses support
# --------------------------------------------------------
import os

os.environ['OMP_NUM_THREADS'] = '4'  # will affect the performance of pairwise prediction
import argparse
import datetime
import json
import numpy as np
import sys
import time
import math
import wandb
from collections import defaultdict
from pathlib import Path
from typing import Sized

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from dust3r.model_noposes import AsymmetricCroCo3DStereo, inf  # noqa: F401, needed when loading the model
from dust3r.datasets import get_data_loader  # noqa
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference_noposes import loss_of_one_batch, visualize_results  # noqa

# from demo import get_3D_model_from_scene
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa


def get_args_parser():
    """Argument parser for DUSt3R training with known poses support"""
    parser = argparse.ArgumentParser('DUST3R training', add_help=False)
    # model and criterion
    parser.add_argument('--model',
                        default="AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', \
                        img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
                        enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')",
                        type=str, help="string containing the model to build")

    # Known poses related parameters
    parser.add_argument('--use_known_poses', action='store_true', default=False,
                        help='Use known camera poses instead of predicting them')
    parser.add_argument('--pose_input_key', default='camera_pose', type=str,
                        help='Key name for pose in batch data')
    parser.add_argument('--disable_pose_loss', action='store_true', default=False,
                        help='Disable pose loss calculation')
    parser.add_argument('--fixed_pose_loss', action='store_true', default=False,
                        help='Use fixed pose loss function')

    parser.add_argument('--pretrained', default=None, help='path of a starting checkpoint')
    parser.add_argument('--train_criterion', default="Regr3D(L21, norm_mode='avg_dis')",
                        type=str, help="train criterion")
    parser.add_argument('--test_criterion', default="Regr3D(L21, norm_mode='avg_dis')", type=str, help="test criterion")

    # dataset
    parser.add_argument('--train_dataset', default='[None]', type=str, help="training set")
    parser.add_argument('--test_dataset', default='[None]', type=str, help="testing set")

    # training
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=64, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--epochs', default=800, type=int, help="Maximum number of epochs for the scheduler")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--amp', type=int, default=0,
                        choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")
    parser.add_argument("--cudnn_benchmark", action='store_true', default=False,
                        help="set cudnn.benchmark = False")
    parser.add_argument("--eval_only", action='store_true', default=False)
    parser.add_argument("--fixed_eval_set", action='store_true', default=False)
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint (default: none)')

    # others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--eval_freq', type=int, default=1, help='Test loss evaluation frequency')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=5, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='frequence (number of iterations) to print infos while training')
    parser.add_argument('--wandb', action='store_true', default=False, help='use wandb for logging')
    parser.add_argument('--num_save_visual', default=1, type=int, help='number of visualizations to save')

    # switch mode for train / eval pose / eval depth
    parser.add_argument('--mode', default='train', type=str, help='train / eval_pose / eval_depth')

    # for monocular depth eval
    parser.add_argument('--no_crop', action='store_true', default=False,
                        help='do not crop the image for monocular depth evaluation')

    # output dir
    parser.add_argument('--output_dir', default='./results/tmp', type=str, help="path where to save the output")
    return parser


def load_model_with_known_poses(args, device):
    """
    Load model with support for known poses mode

    Args:
        args: Command line arguments
        device: Target device

    Returns:
        model: Loaded model
        model_without_ddp: Non-distributed model
    """
    print('Loading model: {:s}'.format(args.model))

    # Modify model configuration for known poses mode
    model_str = args.model

    if args.use_known_poses:
        # Add or modify use_known_poses parameter
        if 'use_known_poses=' not in model_str:
            # Add use_known_poses=True to model configuration
            if ')' in model_str:
                # Insert parameters before closing parenthesis
                model_str = model_str.replace(')', f', use_known_poses=True, pose_input_key="{args.pose_input_key}")')
            else:
                model_str = f'{model_str[:-1]}, use_known_poses=True, pose_input_key="{args.pose_input_key}")'
        else:
            # Replace existing use_known_poses parameter
            model_str = model_str.replace('use_known_poses=False', 'use_known_poses=True')

    print(f'Actual model configuration: {model_str}')

    # Instantiate model
    model = eval(model_str)
    model.to(device)

    # Set model attributes
    model_without_ddp = model

    # Load pretrained weights if specified
    if args.pretrained and not args.resume:
        print('Loading pretrained: ', args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device)
        print(model.load_state_dict(ckpt['model'], strict=False))
        del ckpt  # in case it occupies memory

    # Distributed training setup
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
        model_without_ddp = model.module

    return model, model_without_ddp


def create_criterion_with_fixed_poses(args):
    """
    Create loss criterion with support for fixed poses mode

    Args:
        args: Command line arguments

    Returns:
        train_criterion: Training loss function
        test_criterion: Test loss function
    """
    # Try to import the fixed pose criterion creator
    try:
        from dust3r.losses import create_fixed_pose_criterion
        has_fixed_pose_criterion = True
    except ImportError:
        has_fixed_pose_criterion = False
        print('Warning: create_fixed_pose_criterion not found in dust3r.losses, using standard criterion')

    print(f'>> Creating train criterion = {args.train_criterion}')

    # Create training criterion
    if args.use_known_poses and args.fixed_pose_loss and has_fixed_pose_criterion:
        train_criterion = create_fixed_pose_criterion(args.train_criterion, use_fixed_poses=True)
        print('>> Created fixed pose loss criterion')
    else:
        train_criterion = eval(args.train_criterion)

    # Create test criterion
    if args.test_criterion:
        if args.use_known_poses and args.fixed_pose_loss and has_fixed_pose_criterion:
            test_criterion = create_fixed_pose_criterion(args.test_criterion, use_fixed_poses=True)
        else:
            test_criterion = eval(args.test_criterion)
    else:
        test_criterion = train_criterion

    return train_criterion, test_criterion


def train(args):
    """Main training function with known poses support"""
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()
    # if main process, init wandb
    if args.wandb and misc.is_main_process():
        wandb.init(name=args.output_dir.split('/')[-1],
                   project='dust3r',
                   config=args,
                   sync_tensorboard=True,
                   dir=args.output_dir)

    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # auto resume if not specified
    if args.resume is None:
        last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
        if os.path.isfile(last_ckpt_fname) and (not args.eval_only): args.resume = last_ckpt_fname

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = args.cudnn_benchmark

    # Load model with known poses support
    model, model_without_ddp = load_model_with_known_poses(args, device)

    # training dataset and loader
    print('Building train dataset {:s}'.format(args.train_dataset))
    #  dataset and loader
    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)
    print('Building test dataset {:s}'.format(args.test_dataset))
    data_loader_test = {}
    for dataset in args.test_dataset.split('+'):
        testset = build_dataset(dataset, args.batch_size, args.num_workers, test=True)
        name_testset = dataset.split('(')[0]
        if getattr(testset.dataset.dataset, 'strides', None) is not None:
            name_testset += f'_stride{testset.dataset.dataset.strides}'
        data_loader_test[name_testset] = testset

    # Create loss criteria with fixed poses support
    train_criterion, test_criterion = create_criterion_with_fixed_poses(args)
    train_criterion = train_criterion.to(device)
    test_criterion = test_criterion.to(device)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            gathered_test_stats = {}
            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in train_stats.items()})

            for test_name, testset in data_loader_test.items():

                if test_name not in test_stats:
                    continue

                if getattr(testset.dataset.dataset, 'strides', None) is not None:
                    original_test_name = test_name.split('_stride')[0]
                    if original_test_name not in gathered_test_stats.keys():
                        gathered_test_stats[original_test_name] = []
                    gathered_test_stats[original_test_name].append(test_stats[test_name])

                log_stats.update({test_name + '_' + k: v for k, v in test_stats[test_name].items()})

            if len(gathered_test_stats) > 0:
                for original_test_name, stride_stats in gathered_test_stats.items():
                    if len(stride_stats) > 1:
                        stride_stats = {k: np.mean([x[k] for x in stride_stats]) for k in stride_stats[0]}
                        log_stats.update({original_test_name + '_stride_mean_' + k: v for k, v in stride_stats.items()})
                        if args.wandb:
                            log_dict = {original_test_name + '_stride_mean_' + k: v for k, v in stride_stats.items()}
                            log_dict.update({'epoch': epoch})
                            wandb.log(log_dict)

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far):
        """Save model checkpoint"""
        # Ensure best_so_far is float for saving
        if isinstance(best_so_far, tuple):
            # Extract the first element if it's a float
            if isinstance(best_so_far[0], (int, float)):
                best_so_far_to_save = best_so_far[0]
            else:
                best_so_far_to_save = float('inf')
        else:
            best_so_far_to_save = best_so_far

        misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, fname=fname, best_so_far=best_so_far_to_save)

    # Load checkpoint
    loaded_best = misc.load_model(args=args, model_without_ddp=model_without_ddp,
                                  optimizer=optimizer, loss_scaler=loss_scaler)

    # DEBUG: Print what load_model returns
    print(f"DEBUG: Type of loaded_best: {type(loaded_best)}, Value: {loaded_best}")

    # Initialize best_so_far as float
    best_so_far = float('inf')

    # Handle different return types from load_model
    if loaded_best is not None:
        if isinstance(loaded_best, tuple):
            # If it's a tuple, try to extract float value
            for item in loaded_best:
                if isinstance(item, (int, float)):
                    best_so_far = item
                    break
                elif hasattr(item, 'item'):  # Could be a tensor
                    try:
                        best_so_far = float(item.item())
                        break
                    except:
                        continue
        elif isinstance(loaded_best, (int, float)):
            best_so_far = loaded_best
        elif hasattr(loaded_best, 'item'):  # Could be a tensor
            try:
                best_so_far = float(loaded_best.item())
            except:
                best_so_far = float('inf')

    print(f"DEBUG: Final best_so_far: {best_so_far}, Type: {type(best_so_far)}")

    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # Print known poses mode information
    if args.use_known_poses:
        print("=" * 60)
        print("KNOWN POSES MODE ENABLED")
        print(f"  Pose input key: {args.pose_input_key}")
        print(f"  Disable pose loss: {args.disable_pose_loss}")
        print(f"  Fixed pose loss: {args.fixed_pose_loss}")
        print("=" * 60)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}

    # Set start epoch from checkpoint if resuming
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint.get('epoch', 0) + 1

    for epoch in range(start_epoch, args.epochs + 1):

        # Test on multiple datasets
        new_best = False
        already_saved = False
        if (epoch > start_epoch and args.eval_freq > 0 and epoch % args.eval_freq == 0) or args.eval_only:
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                # Use updated test function with known poses support
                stats = test_one_epoch_with_known_poses(model, test_criterion, testset,
                                                        device, epoch, args=args,
                                                        log_writer=log_writer, prefix=test_name)
                test_stats[test_name] = stats

                # DEBUG: Print stats
                print(f"DEBUG: Stats for {test_name}: {stats}")

                # Extract loss value properly
                if 'loss_med' in stats:
                    # DEBUG: Print loss_med type
                    print(f"DEBUG: Type of stats['loss_med']: {type(stats['loss_med'])}, Value: {stats['loss_med']}")

                    # Extract float value from stats
                    current_loss = stats['loss_med']
                    if isinstance(current_loss, tuple):
                        # Try to extract float from tuple
                        for item in current_loss:
                            if isinstance(item, (int, float)):
                                loss_value = item
                                break
                            elif hasattr(item, 'item'):  # Could be a tensor
                                try:
                                    loss_value = float(item.item())
                                    break
                                except:
                                    continue
                        else:
                            # If no float found in tuple, use first element
                            loss_value = float(current_loss[0]) if len(current_loss) > 0 else float('inf')
                    elif isinstance(current_loss, (int, float)):
                        loss_value = current_loss
                    elif hasattr(current_loss, 'item'):  # Could be a tensor
                        try:
                            loss_value = float(current_loss.item())
                        except:
                            loss_value = float('inf')
                    else:
                        # Try to convert to float
                        try:
                            loss_value = float(current_loss)
                        except:
                            print(f"Warning: Cannot convert loss_med to float: {current_loss}")
                            loss_value = float('inf')

                    # DEBUG: Print extracted loss value
                    print(f"DEBUG: Extracted loss_value: {loss_value}, Type: {type(loss_value)}")
                    print(f"DEBUG: Current best_so_far: {best_so_far}, Type: {type(best_so_far)}")

                    # Compare with float value and update best_so_far with float
                    if loss_value < best_so_far:
                        best_so_far = loss_value  # Store float, not tuple
                        new_best = True
                        print(f"New best loss: {best_so_far:.6f}")
                else:
                    print(f"Warning: 'loss_med' not found in stats: {stats.keys()}")

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if args.eval_only and args.epochs <= 1:
            exit(0)

        # Save checkpoints
        if epoch > start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far)
                already_saved = True
            if new_best:
                save_model(epoch - 1, 'best', best_so_far)
                already_saved = True

        # Save immediately the last checkpoint
        if epoch > start_epoch:
            if args.save_freq and epoch % args.save_freq == 0 or epoch == args.epochs and not already_saved:
                save_model(epoch - 1, 'last', best_so_far)

        if epoch >= args.epochs:
            break  # exit after writing last test to disk

        # Train with known poses support
        train_stats = train_one_epoch_with_known_poses(
            model, train_criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    save_final_model(args, args.epochs, model_without_ddp, best_so_far=best_so_far)


def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    """Save final model checkpoint"""
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / 'checkpoint-final.pth'
    to_save = {
        'args': args,
        'model': model_without_ddp if isinstance(model_without_ddp, dict) else model_without_ddp.cpu().state_dict(),
        'epoch': epoch
    }
    if best_so_far is not None:
        to_save['best_so_far'] = best_so_far
    print(f'>> Saving model to {checkpoint_path} ...')
    misc.save_on_master(to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, test=False):
    """Build dataset loader"""
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader


def train_one_epoch_with_known_poses(model: torch.nn.Module, criterion: torch.nn.Module,
                                     data_loader: Sized, optimizer: torch.optim.Optimizer,
                                     device: torch.device, epoch: int, loss_scaler,
                                     args,
                                     log_writer=None):
    """
    Train for one epoch with support for known poses

    Args:
        model: Model to train
        criterion: Loss function
        data_loader: Data loader
        optimizer: Optimizer
        device: Target device
        epoch: Current epoch number
        loss_scaler: Loss scaler for mixed precision
        args: Command line arguments
        log_writer: TensorBoard writer

    Returns:
        Dictionary of training statistics
    """
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + data_iter_step / len(data_loader)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)

        # Process batch with known poses support
        batch_result = loss_of_one_batch(batch, model, criterion, device,
                                         symmetrize_batch=True,
                                         use_amp=bool(args.amp),
                                         use_known_poses=args.use_known_poses)
        loss, loss_details = batch_result['loss']  # criterion returns two values
        loss_value = float(loss)

        if (data_iter_step % max((len(data_loader) // args.num_save_visual), 1) == 0) and misc.is_main_process():
            save_dir = f'{args.output_dir}/{epoch}'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            view1, view2, pred1, pred2 = batch_result['view1'], batch_result['view2'], batch_result['pred1'], \
            batch_result['pred2']
            gt_visual = visualize_results(view1, view2, pred1, pred2, save_dir=save_dir, visualize_type='gt')
            pred_visual = visualize_results(view1, view2, pred1, pred2, save_dir=save_dir, visualize_type='pred')

            if args.wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_visual_gt': wandb.Object3D(open(gt_visual)),
                    'train_visual_pred': wandb.Object3D(open(pred_visual))
                })

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss
        del batch

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value, **loss_details)

        if (data_iter_step + 1) % accum_iter == 0 and ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0:
            loss_value_reduce = misc.all_reduce_mean(loss_value)  # MUST BE EXECUTED BY ALL NODES
            if log_writer is None:
                continue
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_lr', lr, epoch_1000x)
            log_writer.add_scalar('train_iter', epoch_1000x, epoch_1000x)
            for name, val in loss_details.items():
                log_writer.add_scalar('train_' + name, val, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch_with_known_poses(model: torch.nn.Module, criterion: torch.nn.Module,
                                    data_loader: Sized, device: torch.device, epoch: int,
                                    args, log_writer=None, prefix='test'):
    """
    Test for one epoch with support for known poses

    Args:
        model: Model to test
        criterion: Loss function
        data_loader: Data loader
        device: Target device
        epoch: Current epoch number
        args: Command line arguments
        log_writer: TensorBoard writer
        prefix: Prefix for logging

    Returns:
        Dictionary of test statistics
    """

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9 ** 9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch) if not args.fixed_eval_set else data_loader.dataset.set_epoch(0)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch) if not args.fixed_eval_set else data_loader.sampler.set_epoch(0)

    for idx, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):

        # Process batch with known poses support
        batch_result = loss_of_one_batch(batch, model, criterion, device,
                                         symmetrize_batch=True,
                                         use_amp=bool(args.amp),
                                         use_known_poses=args.use_known_poses)
        loss_tuple = batch_result['loss']
        loss_value, loss_details = loss_tuple  # criterion returns two values
        metric_logger.update(loss=float(loss_value), **loss_details)

        if args.num_save_visual > 0 and (idx % max((len(data_loader) // args.num_save_visual),
                                                   1) == 0) and misc.is_main_process():  # save visualizations
            save_dir = f'{args.output_dir}/{epoch}'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            view1, view2, pred1, pred2 = batch_result['view1'], batch_result['view2'], batch_result['pred1'], \
            batch_result['pred2']
            gt_visual = visualize_results(view1, view2, pred1, pred2, save_dir=save_dir, visualize_type='gt')
            pred_visual = visualize_results(view1, view2, pred1, pred2, save_dir=save_dir, visualize_type='pred')

            if args.wandb:
                wandb.log({
                    'epoch': epoch,
                    'test_visual_gt': wandb.Object3D(open(gt_visual)),
                    'test_visual_pred': wandb.Object3D(open(pred_visual))
                })

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix + '_' + name, val, 1000 * epoch)

    return results


# Keep original functions for backward compatibility
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args,
                    log_writer=None):
    """Original training function (for backward compatibility)"""
    # Call the new function without known poses
    return train_one_epoch_with_known_poses(model, criterion, data_loader, optimizer,
                                            device, epoch, loss_scaler, args, log_writer)


@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test'):
    """Original test function (for backward compatibility)"""
    # Call the new function without known poses
    return test_one_epoch_with_known_poses(model, criterion, data_loader, device,
                                           epoch, args, log_writer, prefix)


def load_model(args, device):
    """Original model loading function (for backward compatibility)"""
    return load_model_with_known_poses(args, device)


def main():
    """Main function"""
    parser = argparse.ArgumentParser('DUST3R training', parents=[get_args_parser()])
    args = parser.parse_args()

    # Start training
    train(args)


if __name__ == '__main__':
    main()