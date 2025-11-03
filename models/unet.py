from tensorflow.keras import layers, models

def unet(input_shape=(512, 512, 3)):
    f = [64, 128, 256, 512, 1024]
    inputs = layers.Input(input_shape)

    c1 = layers.Conv2D(f[0], 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(f[0], 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPool2D((2, 2))(c1)

    c2 = layers.Conv2D(f[1], 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(f[1], 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPool2D((2, 2))(c2)

    c3 = layers.Conv2D(f[2], 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(f[2], 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPool2D((2, 2))(c3)

    c4 = layers.Conv2D(f[3], 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(f[3], 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPool2D((2, 2))(c4)

    c5 = layers.Conv2D(f[4], 3, activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(f[4], 3, activation='relu', padding='same')(c5)

    u6 = layers.Conv2DTranspose(f[3], 2, strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(f[3], 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(f[3], 3, activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(f[2], 2, strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(f[2], 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(f[2], 3, activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(f[1], 2, strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(f[1], 3, activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(f[1], 3, activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(f[0], 2, strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(f[0], 3, activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(f[0], 3, activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)
    return models.Model(inputs, outputs)
