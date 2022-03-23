from cartoonGAN.model.discriminator import build_discriminator
from cartoonGAN.model.generator import build_generator
from cartoonGAN import vars as v
from tensorflow import keras as K
import tensorflow as tf

generator = build_generator(v.SEEDSIZE)
discriminator = build_discriminator()

generator_optimizer = K.optimizers.Adam(v.GENR_LR, 0.5)
discriminator_optimizer = K.optimizers.Adam(v.DISC_LR, 0.5)

cross_entropy = K.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)