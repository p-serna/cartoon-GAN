from tensorflow import keras as K
from tensorflow.keras import layers as L

def build_generator(seed_size, channels = 4):
    model = K.Sequential()
    
    model.add(L.Dense(4*4*256,activation="relu",input_dim=seed_size))
    model.add(L.BatchNormalization())
    model.add(L.LeakyReLU())
    model.add(L.Reshape((4, 4, 256)))
    
    model.add(L.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 256)
    model.add(L.BatchNormalization())
    model.add(L.LeakyReLU())
    
    model.add(L.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 256)
    model.add(L.BatchNormalization())
    model.add(L.LeakyReLU())
    
    
    model.add(L.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 32, 32, 256)
    model.add(L.BatchNormalization())
    model.add(L.LeakyReLU())
    
    
    model.add(L.Conv2DTranspose(64, (5, 5), strides=(3, 3), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 96, 96, 64)
    model.add(L.BatchNormalization())
    model.add(L.LeakyReLU())
    
    model.add(L.Conv2DTranspose(channels, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 96, 96, channels)
    
    return model