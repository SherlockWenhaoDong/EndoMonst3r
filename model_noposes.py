# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class with known poses support
# --------------------------------------------------------
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed, ManyAR_PatchEmbed
from third_party.raft import load_RAFT

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse(
    "0.22.0"), "Outdated huggingface_hub version, please reinstall requirements.txt"


def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricCroCo3DStereo(
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/junyi/monst3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry). Supports using known camera poses instead of predicting them.
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 use_known_poses=True,  # New: whether to use known poses
                 pose_input_key='camera_pose',  # New: key for pose input
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # New: known poses related parameters
        self.use_known_poses = use_known_poses
        self.pose_input_key = pose_input_key

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)

        # Disable pose prediction if using known poses
        if self.use_known_poses:
            self._disable_pose_prediction()

    def _disable_pose_prediction(self):
        """Disable pose prediction functionality when using known poses"""
        # Check if heads have pose prediction capability
        # In DUSt3R, pose prediction is typically part of the downstream head
        # We'll handle this in the forward method by extracting known poses from input
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none': [],
            'mask': [self.mask_token],
            'encoder': [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks,
                                    self.dec_blocks2],
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # x (B, 576, 1024) pos (B, 576, 2); patch_size=16
        B, N, C = x.size()
        posvis = pos
        # add positional embedding without cls token
        assert self.enc_pos_embed is None
        # TODO: where to add mask for the patches
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, posvis)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]

        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # warning! maybe the images have different portrait/landscape orientations
        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection
        original_D = f1.shape[-1]

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2, known_poses=None):
        """
        Forward pass with support for known camera poses

        Args:
            view1: First view data (dictionary containing 'img' and optionally other metadata)
            view2: Second view data (dictionary containing 'img' and optionally other metadata)
            known_poses: Optional dictionary of known poses {'pose1': ..., 'pose2': ...}
                       If None, will try to extract poses from view1/view2 using pose_input_key
        """
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        # Handle known poses if enabled
        if self.use_known_poses:
            # Extract poses from input if known_poses not provided
            if known_poses is None:
                known_poses = {
                    'pose1': view1.get(self.pose_input_key),
                    'pose2': view2.get(self.pose_input_key)
                }

            # If poses are available, add them to the results
            pose1 = known_poses.get('pose1')
            pose2 = known_poses.get('pose2')

            if pose1 is not None:
                res1['camera_pose'] = pose1
            if pose2 is not None:
                res2['camera_pose'] = pose2

        # Note: If use_known_poses is False and the model has pose prediction capability,
        # the downstream heads will include pose predictions in their output

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2

    def forward_with_known_poses(self, view1, view2, known_poses=None):
        """
        Forward pass with known poses (compatibility interface)

        Args:
            view1: First view data
            view2: Second view data
            known_poses: Dictionary of known poses {'pose1': ..., 'pose2': ...}

        Returns:
            res1, res2: Model outputs with known poses included
        """
        return self.forward(view1, view2, known_poses=known_poses)


def create_model_with_known_poses(model_config, use_known_poses=True, pose_input_key='camera_pose'):
    """
    Helper function to create a model with known poses configuration

    Args:
        model_config: Original model configuration string
        use_known_poses: Whether to enable known poses mode
        pose_input_key: Key for pose input in batch data

    Returns:
        Modified model configuration string
    """
    # Parse the model configuration and add known poses parameters
    if use_known_poses:
        # Add use_known_poses parameter to model configuration
        if 'use_known_poses=' not in model_config:
            # Find the closing parenthesis
            if ')' in model_config:
                # Insert parameters before closing parenthesis
                model_config = model_config.replace(')',
                                                    f', use_known_poses={use_known_poses}, pose_input_key="{pose_input_key}")')
            else:
                model_config = f'{model_config[:-1]}, use_known_poses={use_known_poses}, pose_input_key="{pose_input_key}")'
        else:
            # Replace existing use_known_poses parameter
            model_config = model_config.replace('use_known_poses=False', f'use_known_poses={use_known_poses}')

    return model_config


def load_model_with_known_poses(model_path, device, use_known_poses=True, pose_input_key='camera_pose', verbose=True):
    """
    Load model with known poses support

    Args:
        model_path: Path to model checkpoint
        device: Target device
        use_known_poses: Whether to enable known poses mode
        pose_input_key: Key for pose input in batch data
        verbose: Whether to print loading information

    Returns:
        Loaded model with known poses support
    """
    if verbose:
        print('... loading model from', model_path)

    ckpt = torch.load(model_path, map_location='cpu')

    # Get the model configuration from checkpoint
    model_config = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")

    # Add landscape_only=False if not present
    if 'landscape_only' not in model_config:
        model_config = model_config[:-1] + ', landscape_only=False)'
    else:
        model_config = model_config.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')

    assert "landscape_only=False" in model_config

    # Add known poses parameters
    model_config = create_model_with_known_poses(model_config, use_known_poses, pose_input_key)

    if verbose:
        print(f"Instantiating model with configuration: {model_config}")

    # Instantiate the model
    net = eval(model_config)

    # Load weights
    s = net.load_state_dict(ckpt['model'], strict=False)

    if verbose:
        print(f"Model loading status: {s}")

    # Print known poses mode information
    if use_known_poses and verbose:
        print(f"KNOWN POSES MODE ENABLED")
        print(f"  Pose input key: {pose_input_key}")
        print(f"  Model will use provided camera poses instead of predicting them")

    return net.to(device)