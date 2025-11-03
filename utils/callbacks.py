from tensorflow.keras.callbacks import EarlyStopping

def get_callbacks():
    return [
        EarlyStopping(
            monitor='loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
