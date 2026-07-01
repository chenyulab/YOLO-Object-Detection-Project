#!/usr/bin/env python

import json
import argparse
import obj_detector.data.generate_mirror_vars as gen_img_vars
import obj_detector.data_processing.smooth as smooth
import os
import shutil
from tqdm import tqdm
from PIL import Image, ImageOps, ImageFilter

##########
# NEED TO MAKE IT SO THE ORIG AND VARS ARE ALL TOGETHER IN ONE DIR
#####


'''
Created by Jacob Rivera
Spring 2024

Last edit: 01/30/2024

Description:
    Generate all image variations for all images given an images/ and labels/ directories
'''


### NOT TESTED YET ###

def get_parent_directory(directory):
    return os.path.abspath(os.path.join(directory, os.pardir))


def grayscale_image(input_image_path, output_image_path):
    with Image.open(input_image_path) as img:
        grayscale_img = ImageOps.grayscale(img).convert("RGB")
        grayscale_img.save(output_image_path)


def blur_image(input_image_path, output_image_path):
    with Image.open(input_image_path) as img:
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=2.0))
        blurred_img.save(output_image_path)



def main():
    parser = argparse.ArgumentParser(description="Mirror an image and annotation file across x-axis")
    parser.add_argument("--images_dir", required=True, help="directory containing all images")
    parser.add_argument("--labels_dir", required=True, help="directory containing all labels")
    parser.add_argument("--output_dir", required=False, help="top level directory to place output image and label dirs")
    parser.add_argument("--add_directly", type=bool, required=False, default=False, help="Flag if image variations should be placed in input directory, could create something messy")
    args = parser.parse_args()

    images_dir = args.images_dir
    labels_dir = args.labels_dir
    output_dir = args.output_dir
    add_directly = args.add_directly

    if output_dir is None:
        output_dir = 'data_vars'
        output_dir = os.path.join(get_parent_directory(images_dir), output_dir)
        # print(f"output_dir_path, {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)

    # print(output_dir)
    # create paths and vars for the child output directories
    output_images = os.path.join(output_dir, 'images')
    os.makedirs(output_images, exist_ok=True)
    output_labels = os.path.join(output_dir, 'labels')
    os.makedirs(output_labels, exist_ok=True)

    # print(f"output_images: {output_images}")
    # print(f"output_labels: {output_labels}")
    # print()


    images = smooth.list_files_in_directory(images_dir, '.jpg')

    for img in tqdm(images, desc="Processing images"):
        # print(img)
        img = os.path.join(images_dir, img)
        width, height = gen_img_vars.get_image_resolution(img)

        img_name = img.split('\\')[-1]
        text_name = img.split('\\')[-1].replace('.jpg', '.txt')


        lbl = os.path.join(labels_dir, text_name)



        ##########################
        
        
        img_name_out = img_name.replace(".jpg", "_mirror_acr_x.jpg")
        output_img = os.path.join(output_images, img_name_out)

        lbl_name_out = text_name.replace(".txt", "_mirror_acr_x.txt")
        output_lbl = os.path.join(output_labels, lbl_name_out)

        # print(f"img_name_out: {img_name_out}")
        # print(f"output_img: {output_img}")
        # print(f"lbl_name_out: {lbl_name_out}")
        # print(f"output_lbl: {output_lbl}")

        # input()
        # print(lbl)
        # print(output_lbl)
        # input()
        gen_img_vars.flip_image_vertically(img, output_img)
        gen_img_vars.process_text_file(lbl, output_lbl, width, height, 'x')

        ##########################

        img_name_out = img_name.replace(".jpg", "_mirror_acr_y.jpg")
        output_img = os.path.join(output_images, img_name_out)
        
        lbl_name_out = text_name.replace(".txt", "_mirror_acr_y.txt")
        output_lbl = os.path.join(output_labels, lbl_name_out)


        gen_img_vars.mirror_image_horizontally(img, output_img)
        gen_img_vars.process_text_file(lbl, output_lbl, width, height, 'y')

        ###########################

        img_name_out = img_name.replace(".jpg", "_mirror_acr_xy.jpg")
        output_img = os.path.join(output_images, img_name_out)
        
        lbl_name_out = text_name.replace(".txt", "_mirror_acr_xy.txt")
        output_lbl = os.path.join(output_labels, lbl_name_out)

        gen_img_vars.flip_and_mirror_image(img, output_img)
        gen_img_vars.process_text_file(lbl, output_lbl, width, height, 'xy')

        ###########################

        img_name_out = img_name.replace(".jpg", "_black_white.jpg")
        output_img = os.path.join(output_images, img_name_out)

        lbl_name_out = text_name.replace(".txt", "_black_white.txt")
        output_lbl = os.path.join(output_labels, lbl_name_out)

        grayscale_image(img, output_img)
        shutil.copy(lbl, output_lbl)

        ###########################

        img_name_out = img_name.replace(".jpg", "_blurred.jpg")
        output_img = os.path.join(output_images, img_name_out)

        lbl_name_out = text_name.replace(".txt", "_blurred.txt")
        output_lbl = os.path.join(output_labels, lbl_name_out)

        blur_image(img, output_img)
        shutil.copy(lbl, output_lbl)





    return

if __name__ == "__main__":
    main()
