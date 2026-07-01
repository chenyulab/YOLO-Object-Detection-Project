#!/usr/bin/env python

# Import necessary modules or scripts
import argparse
import obj_detector.data_processing.blur as blur

def main():
    parser = argparse.ArgumentParser(description="Get the blur value of a single image")
    parser.add_argument("--input_img", required=True, help="Input image source")

    args = parser.parse_args()
    input_img = args.input_img

    # Call your main function or execution logic
    blur_val = blur.get_blur_level(input_img)

    print(f"Blur value of {input_img}: {blur_val}")

    return

if __name__ == "__main__":
    main()