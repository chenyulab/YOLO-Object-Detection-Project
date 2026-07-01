#!/usr/bin/env python

# Import necessary modules or scripts
import argparse
import obj_detector as op
# import obj_predictor.data_processing.detect_blur as blur_module

def main():
    parser = argparse.ArgumentParser(description="Get a txt file of blurry levels of all images in dir")
    parser.add_argument("--input_dir", required=True, help="Input directory path")
    parser.add_argument("--output_dir", help="Path and name of output txt file. Default is inside input_dir.")
    parser.add_argument("--threshold", type=int, help="Threshold for output statistics. 50 is good more or less and default.")

    args = parser.parse_args()
    input_dir = args.input_dir
    threshold = args.threshold
    output_dir = args.output_dir

    # Call your main function or execution logic
    op.data_processing.batch_gen_blur_levels(input_dir, output_dir, threshold)
    
    
    # data_processing. blur_module.batch_gen_blur_levels(input_dir, output_dir, threshold)

if __name__ == "__main__":
    main()