import os
import sys
import argparse

import cv2
import rasterio
from rasterio.windows import Window


def generate_windows(width, height, size):
    for y in range(0, height, size):
        for x in range(0, width, size):
            win_width = min(size, width - x)
            win_height = min(size, height - y)
            yield x, y, Window(x, y, win_width, win_height)


def crop_image(src, output_dir, size=512):
    os.makedirs(output_dir, exist_ok=True)
    for count, (_, _, window) in enumerate(generate_windows(src.width, src.height, size)):
        crop = src.read(window=window)  
        img = crop.transpose(1, 2, 0)   
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
        path = os.path.join(output_dir, f"crop_{count:04}.png")
        cv2.imwrite(path, img)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with rasterio.open(args.input) as src:
        crop_image(src, args.output, size=args.size)

