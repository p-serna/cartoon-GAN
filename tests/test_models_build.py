import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import matplotlib.pyplot as plt
import cartoonGAN.vars as v
from cartoonGAN.model.discriminator import build_discriminator
from cartoonGAN.model.generator import build_generator

if __name__=="__main__":
  gen = build_generator(10)
  noise = tf.random.normal([1, v.SEEDSIZE])
  generated_image = gen(noise, training=False)
  plt.ion()
  plt.imshow(generated_image[0,:,:,0])
  image_shape = v.INPUT_SHAPE
  discriminator = build_discriminator()
  decision = discriminator(generated_image)
  print(decision)