import os
import cv2
import numpy as np
import random
from tensorflow.keras.utils import Sequence

class DataGen(Sequence):
    def __init__(self, rgb_dir, mask_dir, batch=4, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.x = sorted(os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir))
        self.y = sorted(os.path.join(mask_dir, f) for f in os.listdir(mask_dir))
        self.batch = batch
        self.shuffle = shuffle
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
            xb.append(img)
            yb.append(mask[..., None])
        return np.array(xb), np.array(yb)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.indices)

