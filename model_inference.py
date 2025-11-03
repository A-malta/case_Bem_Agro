import argparse
import os
import glob
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--modelpath", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def collect_images(path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    p = Path(path)
    if p.is_file() and p.suffix.lower() in exts:
        return [str(p)]
    if p.is_dir():
        return [str(f) for f in p.iterdir() if f.suffix.lower() in exts]
    return []


def load_rgb_image(path, size=(512, 512)):
    with Image.open(path) as img:
        img = img.convert("RGB")
        original = np.array(img)
        resized = np.array(img.resize(size)) / 255.0
        return resized[np.newaxis, ...], original


def predict(model, image, shape):
    pred = model.predict(image, verbose=0)[0].squeeze()
    mask = (pred > 0.5).astype(np.uint8)
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.3):
    overlay = np.zeros_like(image)
    overlay[mask == 1] = color
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)


def write_image(output_path, image):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)


def run_inference(model_path, input_path, output_path):
    model = load_model(model_path, compile=False)
    images = collect_images(input_path)
    single_output = len(images) == 1 and not os.path.isdir(output_path)
    if not single_output:
        os.makedirs(output_path, exist_ok=True)
    for img_path in images:
        try:
            image, original = load_rgb_image(img_path)
            mask = predict(model, image, original.shape)
            result = overlay_mask(original, mask)
            out_path = output_path if single_output else os.path.join(output_path, os.path.basename(img_path))
            write_image(out_path, result)
        except Exception:
            continue


def main():
    args = parse_args()
    run_inference(args.modelpath, args.rgb, args.output)


if __name__ == "__main__":
    main()

