from tensorflow import keras as K
from cartoonGAN import vars as v
def build_discriminator_model():
    model = K.Sequential()

    model.add(K.Conv2D(32, kernel_size=3, strides=2, input_shape=v.INPUT_SHAPE, padding="same"))
    model.add(K.LeakyReLU(alpha=0.2))

    model.add(K.Dropout(0.25))
    model.add(K.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(K.ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(K.BatchNormalization(momentum=0.8))
    model.add(K.LeakyReLU(alpha=0.2))

    model.add(K.Dropout(0.25))
    model.add(K.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(K.BatchNormalization(momentum=0.8))
    model.add(K.LeakyReLU(alpha=0.2))

    model.add(K.Dropout(0.25))
    model.add(K.Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(K.BatchNormalization(momentum=0.8))
    model.add(K.LeakyReLU(alpha=0.2))

    model.add(K.Dropout(0.25))
    model.add(K.Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(K.BatchNormalization(momentum=0.8))
    model.add(K.LeakyReLU(alpha=0.2))

    model.add(K.Dropout(0.25))
    model.add(K.Flatten())
    model.add(K.Dense(1, activation="sigmoid"))

    return model