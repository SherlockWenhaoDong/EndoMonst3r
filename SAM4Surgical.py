import torch
import os
from PIL import Image
import numpy as np
import json
from pathlib import Path

# Add the sam3 module path to system path
import sys

sys.path.append('/home/bygpu/sam3')

# 🔧 强制完全离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 🔧 只调整这个置信度阈值 - 范围 0-1，值越大要求越严格
CONFIDENCE_THRESHOLD = 0.05


def load_sam3_from_local():
    """
    Load SAM3 model from local models directory
    """
    model_dir = Path('/home/bygpu/sam3/models')
    model_path = model_dir / 'sam3.pt'
    config_path = model_dir / 'config.json'

    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return None

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return None

    print("✅ Found local SAM3 model files")

    try:
        print("🔄 Loading SAM3 model...")
        model = build_sam3_image_model(checkpoint_path=model_path, load_from_HF=False)
        model.eval()

        print("✅ SAM3 model loaded successfully from local files")
        return model

    except Exception as e:
        print(f"❌ Error loading model from local files: {e}")
        return None


def process_folder_for_non_human_tissues(folder_path, mask_output_dir=None, visualization_dir=None):
    """
    Process all images in a folder to segment non-human tissues and save masks
    """
    # Load the SAM3 model from local files
    model = load_sam3_from_local()

    if model is None:
        print("❌ Failed to load SAM3 model.")
        return {}

    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)

    # Supported image formats
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # Check if input folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Input folder does not exist: {folder_path}")
        return {}

    # Get all image files in the folder
    image_files = []
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(fmt) for fmt in supported_formats):
            image_files.append(os.path.join(folder_path, file))

    if not image_files:
        print(f"❌ No supported image files found in folder: {folder_path}")
        return {}

    # Sort image files for consistent processing
    image_files.sort()

    # Create output directories if they don't exist
    if mask_output_dir and not os.path.exists(mask_output_dir):
        os.makedirs(mask_output_dir)
        print(f"📁 Created mask output directory: {mask_output_dir}")

    if visualization_dir and not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)
        print(f"📁 Created visualization directory: {visualization_dir}")

    results = {}

    print(f"🎯 Starting to process {len(image_files)} images...")
    print(f"🔧 Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"💬 Using prompt: 'tools'")

    # Process each image in the folder
    for i, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        print(f"\n📸 Processing image {i + 1}/{len(image_files)}: {filename}")

        try:
            # Load image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_array = np.array(image)

            print("  🔄 Setting image for processing...")
            # Set image for processing
            inference_state = processor.set_image(image)

            all_masks = []
            all_boxes = []
            all_scores = []

            # 🔧 只使用 "tools" 提示词
            try:
                print("  🔍 Using prompt: 'tools'")
                output = processor.set_text_prompt(state=inference_state, prompt="tools")

                if output["masks"] is not None and len(output["masks"]) > 0:
                    # 🔧 应用置信度阈值过滤
                    initial_objects = len(output["masks"])
                    passed_objects = 0

                    for j in range(len(output["masks"])):
                        all_masks.append(output["masks"][j])
                        all_boxes.append(output["boxes"][j])
                        all_scores.append(output["scores"][j])
                        passed_objects += 1

                    print(
                        f"    ✅ Found {initial_objects} objects, {passed_objects} passed confidence threshold (>{CONFIDENCE_THRESHOLD})")
                else:
                    print("    ℹ️ No objects detected with prompt 'tools'")

            except Exception as e:
                print(f"    ❌ Prompt 'tools' failed: {e}")

            # Store results for current image
            results[filename] = {
                'masks': all_masks,
                'boxes': all_boxes,
                'scores': all_scores,
                'image_size': image.size
            }

            # Save combined mask only
            if mask_output_dir and all_masks:
                save_combined_mask(image_array, all_masks, mask_output_dir, filename)
            else:
                print("  ℹ️ No masks found to save")

            # Save visualization results (optional)
            if visualization_dir and all_masks:
                save_visualization(image, all_masks, all_scores, visualization_dir, filename)

            print(f"  ✅ Completed {filename}, found {len(all_masks)} non-human tissue regions")

        except Exception as e:
            print(f"  ❌ Error processing image {image_path}: {e}")
            continue

    # Save metadata for all results
    if mask_output_dir and results:
        save_results_metadata(results, mask_output_dir)

    return results


def save_combined_mask(original_image, masks, output_dir, filename):
    """
    Save only the combined mask file to specified directory
    """
    try:
        base_name = os.path.splitext(filename)[0]

        # Save combined mask (all detected objects merged)
        if masks:
            # Initialize combined mask
            if len(original_image.shape) == 3:
                combined_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
            else:
                combined_mask = np.zeros_like(original_image)

            # Merge all masks
            for mask in masks:
                if mask is None:
                    continue

                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)

                if len(mask_np.shape) > 2:
                    mask_np = mask_np.squeeze()

                # Combine using logical OR
                combined_mask = np.logical_or(combined_mask, mask_np > 0.5)

            # Convert to binary image
            combined_mask = combined_mask.astype(np.uint8) * 255
            combined_mask_image = Image.fromarray(combined_mask)
            combined_mask_path = os.path.join(output_dir, f"{base_name}_combined_mask.png")
            combined_mask_image.save(combined_mask_path, 'PNG')
            print(f"    💾 Saved combined mask: {base_name}_combined_mask.png")
        else:
            print("    ℹ️ No masks to combine")

    except Exception as e:
        print(f"  ❌ Error saving combined mask: {e}")


def save_visualization(original_image, masks, scores, output_dir, filename):
    """
    Save visualization with mask overlays (optional)
    """
    try:
        # Create overlay image
        overlay = original_image.copy().convert('RGBA')

        # Create colored overlay for each mask
        for i, (mask, score) in enumerate(zip(masks, scores)):
            if mask is None:
                continue

            # Convert mask to numpy array
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)

            # Ensure mask is 2D
            if len(mask_np.shape) > 2:
                mask_np = mask_np.squeeze()

            # Create colored mask image
            mask_image = Image.new('RGBA', original_image.size, (0, 0, 0, 0))

            # Create color based on confidence score
            if score > 0.7:
                color = (0, 255, 0, 128)  # Green for high confidence
            elif score > 0.4:
                color = (255, 255, 0, 128)  # Yellow for medium confidence
            else:
                color = (255, 0, 0, 128)  # Red for low confidence

            # Apply mask using numpy for better performance
            mask_array = np.array(mask_image)
            mask_binary = (mask_np > 0.5)
            mask_array[mask_binary, 0] = color[0]  # Red channel
            mask_array[mask_binary, 1] = color[1]  # Green channel
            mask_array[mask_binary, 2] = color[2]  # Blue channel
            mask_array[mask_binary, 3] = color[3]  # Alpha channel

            mask_image = Image.fromarray(mask_array, 'RGBA')

            # Composite with overlay
            overlay = Image.alpha_composite(overlay, mask_image)

        # Save visualization result
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}_visualization.png")
        overlay.save(output_path, 'PNG')
        print(f"    🎨 Saved visualization: {base_name}_visualization.png")

    except Exception as e:
        print(f"  ❌ Error saving visualization: {e}")


def save_results_metadata(results, output_dir):
    """
    Save detection metadata as JSON file
    """
    try:
        metadata = {}

        # Process results for each image
        for filename, data in results.items():
            masks_info = []
            for i, (mask, box, score) in enumerate(zip(data['masks'], data['boxes'], data['scores'])):
                # Calculate mask area
                mask_area = 0
                if mask is not None:
                    if isinstance(mask, torch.Tensor):
                        mask_np = mask.cpu().numpy()
                    else:
                        mask_np = np.array(mask)
                    mask_area = int(np.sum(mask_np > 0.5))

                mask_info = {
                    'mask_id': i,
                    'confidence_score': float(score),
                    'bounding_box': box.tolist() if hasattr(box, 'tolist') else box,
                    'mask_area': mask_area
                }
                masks_info.append(mask_info)

            metadata[filename] = {
                'image_size': data['image_size'],
                'num_detections': len(data['masks']),
                'detections': masks_info,
                'confidence_threshold': CONFIDENCE_THRESHOLD
            }

        # Save metadata as JSON file
        metadata_path = os.path.join(output_dir, "detection_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"📄 Metadata saved to: {metadata_path}")

    except Exception as e:
        print(f"❌ Error saving metadata: {e}")


def print_summary(results):
    """
    Print processing summary
    """
    if not results:
        print("❌ No results to display")
        return

    total_images = len(results)
    total_detections = sum(len(data['masks']) for data in results.values())
    images_with_detections = sum(1 for data in results.values() if len(data['masks']) > 0)

    print("\n" + "=" * 50)
    print("📊 PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total images processed: {total_images}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total regions detected: {total_detections}")
    print(f"Detection rate: {images_with_detections / total_images * 100:.1f}%")
    print(f"Confidence threshold used: {CONFIDENCE_THRESHOLD}")


# Main execution
if __name__ == "__main__":
    # Set input folder path
    input_folder = "/home/bygpu/Downloads/SurgicalRecon/3_PANORAMIC_STATIC/frames_cam3"

    # Set output directories
    mask_output_folder = "/home/bygpu/Downloads/SurgicalRecon/3_PANORAMIC_STATIC/frames_cam3_masks"
    visualization_folder = "/home/bygpu/Downloads/SurgicalRecon/3_PANORAMIC_STATIC/frames_cam0_visualizations"

    print("🚀 Starting SAM3 Surgical Tool Segmentation")
    print("💾 Using local model files from /home/bygpu/sam3/models/")
    print(f"🔧 Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"💬 Prompt: 'tools'")
    print("=" * 50)

    # Process all images in the folder
    results = process_folder_for_non_human_tissues(
        input_folder,
        mask_output_dir=mask_output_folder,
        visualization_dir=visualization_folder
    )

    # Print summary
    print_summary(results)
    print("\n🎉 Processing completed!")