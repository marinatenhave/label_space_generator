"""
generate_label_space_from_dir.py

🧠 Description:
This script processes a directory of images and generates a scene-wide semantic label space
(using the LabelGenerator class). It prints the results and saves a final label space JSON
and YAML file in the appropriate location.

📦 Usage:
    python generate_label_space_from_dir.py --image-dir images/ --dataset mp3d --scene scene_001

🗂️ Expected structure:
    - label_generator.py (same folder)
    - images/<image files>

💡 Notes:
    - Requires models to download from HuggingFace on first run.
    - Automatically creates a per-scene label YAML and JSON.
"""
import os
import argparse
import cv2
from datetime import datetime
from label_generator import LabelGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="Path to image directory")
    parser.add_argument("--dataset", default=None, help="Optional dataset name")
    parser.add_argument("--scene", default=None, help="Optional scene name")
    parser.add_argument("--fixed-labelspace", default=None, help="Optional path to fixed label space YAML")
    args = parser.parse_args()

    dataset_name = args.dataset if args.dataset else None
    scene_name = args.scene if args.scene else None

    if not dataset_name and not scene_name:
        print("🆕 No dataset or scene specified — using timestamp fallback inside LabelGenerator.")
    elif dataset_name and not scene_name:
        print(f"🆕 Only dataset specified: '{dataset_name}' → scene fallback handled inside LabelGenerator.")
    elif scene_name and not dataset_name:
        print(f"🆕 Only scene specified: '{scene_name}' → dataset fallback handled inside LabelGenerator.")
    else:
        print(f"🧭 Using dataset = '{dataset_name}', scene = '{scene_name}'")

    # Initialize LabelGenerator
    generator = LabelGenerator(fixed_labelspace_path=args.fixed_labelspace)
    generator.set_run_info(dataset_name=dataset_name, scene_name=scene_name)

    # Process all images
    image_filenames = sorted(f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    if not image_filenames:
        print("❌ No valid images found in:", args.image_dir)
        return

    for filename in image_filenames:
        image_path = os.path.join(args.image_dir, filename)
        print(f"\n🖼️ Processing {filename}...")

        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Could not read {filename}, skipping.")
            continue

        generator.step(image)

if __name__ == "__main__":
    main()
