#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Multi-camera RGB + Mask + Optical Flow Preprocessing
# Author: Wenhao Dong
# Updated: 2025-11-07
# --------------------------------------------------------

import argparse
import os
import os.path as osp
import random
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from tqdm.auto import tqdm
from PIL import Image


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing camera folders")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--num_frames", type=int, default=100, help="Number of frames to sample per camera")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera_param", type=str, default="camera.xml", help="Path to XML camera parameter file")
    return parser


def load_camera_params(xml_path):
    """Load camera intrinsics and distortion from XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    K_values = [float(x) for x in root.find(".//param[@name='K']").text.strip().split()]
    K = np.array(K_values).reshape(3, 3)

    dist_values = [float(x) for x in root.find(".//param[@name='distortion']").text.strip().split()]
    distortion = np.array(dist_values)

    img_size = [int(x) for x in root.find(".//param[@name='imageSize']").text.strip().split()]

    return K, distortion, img_size


def undistort_image(image, K, distortion):
    """Undistort image using camera intrinsics."""
    h, w = image.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, distortion, (w, h), 1, (w, h))
    undistorted = cv2.undistort(image, K, distortion, None, new_K)
    return undistorted


def process_camera_folder(camera_dir, output_dir, K, distortion, img_size, num_frames, seed):
    """Process one camera folder containing rgb/, rgb_masks/, and flow/."""
    random.seed(seed)

    rgb_dir = osp.join(camera_dir, "rgb")
    mask_dir = osp.join(camera_dir, "rgb_masks")
    flow_dir = osp.join(camera_dir, "flow")

    if not osp.exists(rgb_dir):
        print(f"[Warning] No RGB directory found in {camera_dir}")
        return

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.startswith("frame_") and f.endswith(".png")])
    total_frames = len(rgb_files)
    if total_frames == 0:
        print(f"[Warning] No RGB frames found in {camera_dir}")
        return

    # Select evenly spaced frames
    selected_indices = np.round(np.linspace(0, total_frames - 1, num_frames)).astype(int).tolist()

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    rgb_out = osp.join(output_dir, "rgb")
    mask_out = osp.join(output_dir, "masks")
    flow_out = osp.join(output_dir, "flow")
    meta_out = osp.join(output_dir, "metadata")
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)
    os.makedirs(flow_out, exist_ok=True)
    os.makedirs(meta_out, exist_ok=True)

    for idx in tqdm(selected_indices, desc=f"Processing {osp.basename(camera_dir)}"):
        rgb_name = rgb_files[idx]
        frame_id = int(rgb_name.split("_")[1].split(".")[0])  # e.g. frame_000123.png -> 123
        frame_str = f"frame_{frame_id:06d}"

        # --- Load RGB ---
        rgb_path = osp.join(rgb_dir, rgb_name)
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            continue
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = undistort_image(rgb_img, K, distortion)

        # --- Load mask ---
        mask_path = osp.join(mask_dir, f"{frame_str}.png")
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if osp.exists(mask_path) else np.zeros(rgb_img.shape[:2])

        # --- Load optical flow (fwd & bwd) ---
        flow_fwd_path = osp.join(flow_dir, f"{frame_str}_fwd.npz")
        flow_bwd_path = osp.join(flow_dir, f"{frame_str}_bwd.npz")
        flow_fwd = np.load(flow_fwd_path)["flow"] if osp.exists(flow_fwd_path) else None
        flow_bwd = np.load(flow_bwd_path)["flow"] if osp.exists(flow_bwd_path) else None

        # --- Save processed data ---
        Image.fromarray(rgb_img).save(osp.join(rgb_out, f"{frame_str}.jpg"))
        cv2.imwrite(osp.join(mask_out, f"{frame_str}.png"), mask_img.astype(np.uint8))

        if flow_fwd is not None:
            np.savez_compressed(osp.join(flow_out, f"{frame_str}_fwd.npz"), flow=flow_fwd)
        if flow_bwd is not None:
            np.savez_compressed(osp.join(flow_out, f"{frame_str}_bwd.npz"), flow=flow_bwd)

        # --- Metadata (camera intrinsics, distortion, constant pose) ---
        metadata = {
            "K": K.tolist(),
            "distortion": distortion.tolist(),
            "imageSize": img_size,
            "pose": np.eye(4).tolist(),  # fixed pose
            "frame_id": frame_id
        }
        np.savez(osp.join(meta_out, f"{frame_str}.npz"), **metadata)

    print(f"✅ Finished processing camera: {osp.basename(camera_dir)}")


def main():
    parser = get_parser()
    args = parser.parse_args()

    K, distortion, img_size = load_camera_params(args.camera_param)
    print("Loaded camera intrinsics:")
    print(K)

    camera_folders = [osp.join(args.root_dir, d) for d in os.listdir(args.root_dir)
                      if osp.isdir(osp.join(args.root_dir, d))]

    os.makedirs(args.output_dir, exist_ok=True)

    for cam_folder in camera_folders:
        cam_name = osp.basename(cam_folder)
        out_dir = osp.join(args.output_dir, cam_name)
        process_camera_folder(cam_folder, out_dir, K, distortion, img_size,
                              args.num_frames, args.seed)


if __name__ == "__main__":
    main()
