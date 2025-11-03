import argparse
import os
import glob
import cv2
import numpy as np
from tensorflow.keras.models import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--modelpath", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def get_image_paths(image_dir):
    return glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))


def load_image(path, size=(512, 512)):
    img = cv2.imread(path)
    original = img.copy()
    img = cv2.resize(img, size) / 255.0
    return img[np.newaxis, ...], original


def predict_mask(model, image, original_shape):
    pred = model.predict(image, verbose=0)[0].squeeze()
    mask = (pred > 0.5).astype(np.uint8)
    return cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)


def apply_color_overlay(original, mask, color=(0, 255, 0), alpha=0.3):
    overlay = np.zeros_like(original)
    overlay[mask == 1] = color
    return cv2.addWeighted(original, 1 - alpha, overlay, alpha, 0)


def save_image(image, output_path):
    cv2.imwrite(output_path, image)


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    model = load_model(args.modelpath, compile=False)
    image_paths = get_image_paths(args.rgb)

    for img_path in image_paths:
        image, original = load_image(img_path)
        mask = predict_mask(model, image, original.shape)
        result = apply_color_overlay(original, mask)
        output_path = os.path.join(args.output, os.path.basename(img_path))
        save_image(result, output_path)
        print(f"Imagem segmentada salva: {os.path.basename(img_path)}")


if __name__ == "__main__":
    main()

