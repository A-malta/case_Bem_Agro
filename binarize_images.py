import os
import cv2
import numpy as np
import argparse


def compute_excess_green(bgr_image):
    b, g, r = cv2.split(bgr_image.astype(np.float32))
    return 2 * g - r - b


def binarize_exg_image(exg_image):
    exg_image = cv2.normalize(exg_image, None, 0, 255, cv2.NORM_MINMAX)
    exg_image = exg_image.astype(np.uint8)
    _, mask = cv2.threshold(exg_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask01 = (mask // 255).astype(np.uint8)  
    return mask01, mask 


def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png')):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        image = cv2.imread(input_path)

        exg = compute_excess_green(image)
        mask01, mask255 = binarize_exg_image(exg)
        cv2.imwrite(output_path, mask255)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_directory(args.input, args.output)
