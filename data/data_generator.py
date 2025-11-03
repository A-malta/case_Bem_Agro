import os
import cv2
import numpy as np
import random
from tensorflow.keras.utils import Sequence

class DataGen(Sequence):
    def __init__(self, rgb_dir, mask_dir, batch=4, shuffle=True, augment=True, **kwargs):
        super().__init__(**kwargs)
        self.x = sorted(os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir))
        self.y = sorted(os.path.join(mask_dir, f) for f in os.listdir(mask_dir))
        self.batch = batch
        self.shuffle = shuffle
        self.augment = augment
        self.indices = list(range(len(self.x)))
        if shuffle:
            random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices) // self.batch

    def __getitem__(self, i):
        xb, yb = [], []
        for j in range(self.batch):
            idx = self.indices[i * self.batch + j]
            img = cv2.imread(self.x[idx]) / 255.0
            mask = cv2.imread(self.y[idx], cv2.IMREAD_GRAYSCALE) / 255.0
            if self.augment:
                img, mask = self._augment(img, mask)
            xb.append(img)
            yb.append(mask[..., None])
        return np.array(xb), np.array(yb)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.indices)

    def _augment(self, img, mask):
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        if random.random() < 0.5:
            img *= random.uniform(0.7, 1.3)
            img = np.clip(img, 0, 1)
        return img, mask

