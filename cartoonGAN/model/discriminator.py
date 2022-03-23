from tensorflow import keras as K
from tensorflow.keras import layers as L
from cartoonGAN import vars as v
def build_discriminator():
    model = K.Sequential()

    model.add(L.Conv2D(32, kernel_size=3, strides=2, input_shape=v.INPUT_SHAPE, padding="same"))
    model.add(L.LeakyReLU(alpha=0.2))

    model.add(L.Dropout(0.25))
    model.add(L.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(L.ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(L.BatchNormalization(momentum=0.8))
    model.add(L.LeakyReLU(alpha=0.2))

    model.add(L.Dropout(0.25))
    model.add(L.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(L.BatchNormalization(momentum=0.8))
    model.add(L.LeakyReLU(alpha=0.2))

    model.add(L.Dropout(0.25))
    model.add(L.Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(L.BatchNormalization(momentum=0.8))
    model.add(L.LeakyReLU(alpha=0.2))

    model.add(L.Dropout(0.25))
    model.add(L.Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(L.BatchNormalization(momentum=0.8))
    model.add(L.LeakyReLU(alpha=0.2))

    model.add(L.Dropout(0.25))
    model.add(L.Flatten())
    model.add(L.Dense(1, activation="sigmoid"))

    return model