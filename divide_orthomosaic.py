import os
import sys

import cv2
import rasterio
from rasterio.windows import Window


def to_opencv_bgr(array):
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def generate_windows(width, height, size):
    for y in range(0, height, size):
        for x in range(0, width, size):
            if x + size <= width and y + size <= height:
                yield x, y, Window(x, y, size, size)


def crop_image(src, output_dir, size=512):
    os.makedirs(output_dir, exist_ok=True)
    for count, (_, _, window) in enumerate(generate_windows(src.width, src.height, size)):
        crop = src.read(window=window)  
        img = crop.transpose(1, 2, 0)   
        bgr = to_opencv_bgr(img)
        path = os.path.join(output_dir, f"crop_{count:04}.png")
        cv2.imwrite(path, bgr)


def main():
    if len(sys.argv) != 3:
        print("Uso: python divide_orthomosaic.py <input.tif> <output_folder>")
        sys.exit(1)

    input_path, output_dir = sys.argv[1], sys.argv[2]

    with rasterio.open(input_path) as src:
        crop_image(src, output_dir)


if __name__ == "__main__":
    main()