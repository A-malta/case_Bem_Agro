from models.unet import unet
from data.data_generator import DataGen
from metrics.metrics import dice_coef, iou_coef
from utils.callbacks import get_callbacks


def train(rgb_path, gt_path, model_out_path, input_shape=(512, 512, 3), batch=4, epochs=50):
    model = unet(input_shape=input_shape)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', dice_coef, iou_coef]
    )

    gen = DataGen(rgb_path, gt_path, batch=batch)

    model.fit(
        gen,
        epochs=epochs,
        callbacks=get_callbacks(),
       # verbose=2
    )

    model.save(model_out_path)
