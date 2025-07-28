# --- START OF FILE run_augmentation.py ---

import argparse
import sys
from pathlib import Path

# Add project root to Python path to enable imports from src/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.kinetic_augment.utils.data_formats import load_and_reshape_json, save_to_json
from src.kinetic_augment.pipeline import AugmentationPipeline

def main():
    parser = argparse.ArgumentParser(
        description="Apply geometry-aware augmentations to a single landmark data file."
    )
    parser.add_argument("--input_json", required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_json", required=True, help="Path where the augmented JSON file will be saved.")
    parser.add_argument("--profile", required=True, help="Path to the .yaml augmentation profile.")
    parser.add_argument(
        "--output_space", 
        choices=['canonical', 'original'], 
        default='original', 
        help="Coordinate space for the output file. 'original' for visualization, 'canonical' for training."
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    profile_path = Path(args.profile)

    print(f"Loading data from: {input_path}")
    try:
        raw_data = load_and_reshape_json(input_path)
    except Exception:
        return

    print(f"\nInitializing augmentation pipeline with profile: {profile_path}")
    try:
        pipeline = AugmentationPipeline(profile_path=profile_path)
        # Pass the desired output space to the process method
        augmented_data = pipeline.process(raw_data, output_space=args.output_space)
    except Exception as e:
        print(f"A critical error occurred during pipeline execution: {e}")
        return

    print(f"\nSaving augmented data to: {output_path} (Space: {args.output_space})")
    try:
        save_to_json(augmented_data, output_path)
        print("✅ Successfully saved augmented file.")
    except Exception:
        return

if __name__ == "__main__":
    main()