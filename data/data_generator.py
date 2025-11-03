import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

class DataGen(Sequence):
    def __init__(self, rgb_dir, mask_dir, batch=4, **kwargs):
        super().__init__(**kwargs)
        self.x = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir)])
        self.y = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        self.batch = batch


    def __len__(self):
        return len(self.x) // self.batch
        

    def __getitem__(self, i):
        xb, yb = [], []
        for j in range(self.batch):
            k = i * self.batch + j
            img = cv2.imread(self.x[k]) / 255.
            mask = cv2.imread(self.y[k], cv2.IMREAD_GRAYSCALE) / 255.
            xb.append(img)
            yb.append(mask[..., None])
        return np.array(xb), np.array(yb)
