from tensorflow import keras as K

def build_generator(seed_size, channels):
    model = K.Sequential()
    
    model.add(K.Dense(4*4*256,activation="relu",input_dim=seed_size))
    model.add(K.BatchNormalization())
    model.add(K.LeakyReLU())
    model.add(K.Reshape((4, 4, 256)))
    
    model.add(K.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 256)
    model.add(K.BatchNormalization())
    model.add(K.LeakyReLU())
    
    model.add(K.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 256)
    model.add(K.BatchNormalization())
    model.add(K.LeakyReLU())
    
    
    model.add(K.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 32, 32, 256)
    model.add(K.BatchNormalization())
    model.add(K.LeakyReLU())
    
    
    model.add(K.Conv2DTranspose(64, (5, 5), strides=(3, 3), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 96, 96, 64)
    model.add(K.BatchNormalization())
    model.add(K.LeakyReLU())
    
    model.add(K.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 96, 96, 3)
    
    return model