#!/usr/bin/env python


'''
Created by Jacob Rivera
Fall 2023

Last edit: 01/03/2024

Description:
This script will generate text files containing the blur levels of each image in the object level directories.
Given a top level directory containing object level directories, this script will iterate through each object level directory
and create a text file containing the blur levels of each image in the object level directory and produce ratio of blurry images to total images.

'''



# Import necessary modules or scripts
import argparse
import obj_detector.data_processing.blur as blur_module

def main():
    parser = argparse.ArgumentParser(description="Get the blur values for each object")
    parser.add_argument("--input_dir", required=True, help="Parent directory containing object lvl dirs")
    parser.add_argument("--num_obj", required=True, help="Number of object level directories")
    parser.add_argument("--threshold", required=False, default=50, help="Threshold for blurry level")
    parser.add_argument("--output_dir", required=False, default='', help="Directory to put obj blur levels text files, if none given, will put in src dir for each object")

    args = parser.parse_args()

    input_dir = args.input_dir
    num_obj = args.num_obj
    threshold = args.threshold
    output_dir = args.output_dir

    # Call your main function or execution logic
    blur_module.get_obj_blur_levels(input_dir, num_obj, threshold, output_dir)

    return

if __name__ == "__main__":
    main()