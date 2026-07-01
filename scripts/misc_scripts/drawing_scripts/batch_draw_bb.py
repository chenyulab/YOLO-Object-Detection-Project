#!/usr/bin/env python

import argparse
import os
import obj_detector as op
from tqdm import tqdm

'''
Created by Jacob Rivera
Spring 2024

Last edit: 02/29/2024

Description:

'''


def get_text_file(input_img):
    path, img_name = os.path.split(input_img)
    text_name = img_name[:-3] + "txt"
    return text_name



def main():
    parser = argparse.ArgumentParser(description="draw bounding boxes on all images from corresponding txt file")
    parser.add_argument("--images_dir", required=True, help="")
    parser.add_argument("--labels_dir", required=True, help="")
    parser.add_argument("--output_dir", required=True, help="")
    
    args = parser.parse_args()

    images_dir = args.images_dir
    labels_dir = args.labels_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    all_images = op.data_processing.smooth.list_files_in_directory(images_dir, '.jpg')
    all_annots = op.data_processing.smooth.list_files_in_directory(labels_dir, ".txt")

    for i in tqdm(range(len(all_images)), desc="Drawing annotations on images..."):
        img_path = os.path.join(images_dir, all_images[i])

        text_file = os.path.join(labels_dir, all_annots[i])

        path, img_root = os.path.split(all_images[i])
        drawn_img = os.path.join(output_dir, img_root[:-4] + "_drawn.jpg")

        op.data_processing.draw.draw_single_frame(img_path, text_file, drawn_img, True )

    return



if __name__ == "__main__":
    main()