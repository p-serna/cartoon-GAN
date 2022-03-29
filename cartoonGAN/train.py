import tensorflow as tf
from cartoonGAN import vars as v
from cartoonGAN.model import model
from cartoonGAN.preprocess.visualize import plot_imgs
import time
import numpy as np 
from matplotlib.pyplot import close
from datetime import timedelta

@tf.function
def train_step(images):
  seed = tf.random.normal([v.BATCH_SIZE, v.SEEDSIZE])
  
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = model.generator(seed, training=True)
  
    real_output = model.discriminator(images, training=True)
    fake_output = model.discriminator(generated_images, training=True)

    gen_loss = model.generator_loss(fake_output)
    disc_loss = model.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, model.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, model.discriminator.trainable_variables)
      
    model.generator_optimizer.apply_gradients(zip(gradients_of_generator, model.generator.trainable_variables))
    model.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, model.discriminator.trainable_variables))
  
  return gen_loss, disc_loss

def save_images(epoch,fixed_seed):
  imgs = model.generator(fixed_seed, training=False)
  imgs = np.clip(imgs,0,1)
  fig, axs = plot_imgs(imgs)
  fig.savefig(f"training/epoch{epoch:03d}.png")
  close(fig)
  return

def train(datasetgenerator, epochs,nbatchesperepoch=10):
  fixed_seed = np.random.normal(0, 1, (v.PREVIEW_ROWS * v.PREVIEW_COLS, v.SEEDSIZE))
  start = time.time()
  datagen = datasetgenerator()

  for epoch in range(epochs):
    epoch_start = time.time()

    gen_loss_list = []
    disc_loss_list = []

    for ib,image_batch in enumerate(datagen):
      t = train_step(image_batch)
      gen_loss_list.append(t[0])
      disc_loss_list.append(t[1])
      if ib>nbatchesperepoch:
        break
    if len(gen_loss_list)==0:
      datagen = datasetgenerator()
      continue
    g_loss = np.mean(gen_loss_list) #/ len(gen_loss_list)
    d_loss = np.mean(disc_loss_list) #/ len(disc_loss_list)

    epoch_elapsed = (time.time() - epoch_start)
    print("Epoch {}, gen loss={}, disc loss={}, {}".format(epoch+1, g_loss, d_loss, str(epoch_elapsed)))
    save_images(epoch, fixed_seed)

  elapsed = (time.time() - start)
  print("Training time: {} minutes".format(str(elapsed/60)))