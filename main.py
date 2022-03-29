import cartoonGAN as cgan
from cartoonGAN.preprocess import transform2numpy as t2np
import cartoonGAN.vars as v
import cartoonGAN.train as train
import sys, os
import numpy as np 


def datasetgenerator(files, batch_size=v.BATCH_SIZE):
  X, Y = None,None
  for file in files:
    X0 = np.load(file)
    #X0, Y0 = t["X"],t["Y"]
    if X is None:
      #X,Y = X0,Y0
      X = X0
    else:
      X = np.concatenate((X,X0))
      #Y = np.concatenate((Y,Y0))
    while X.shape[0]>batch_size:
      yield X[:batch_size,:]#,Y[:batch_size,:]
      #X,Y = X[batch_size:,:],Y[batch_size:,:]
      X = X[batch_size:,:]


if __name__ =="__main__":
  if sys.argv[1] == "preprocess":
    print("Starting preprocessing data:")
    t2np.preprocess_data()

  if sys.argv[1] == "train":
    files = [f"{v.DATAST}/{f}" for f in os.listdir(v.DATAST) if f.split(".")[-1]=="npy" and f.split("/")[-1][:4]=="data"]
    #datagen = datasetgenerator(files)
    train.train(lambda batch_size=v.BATCH_SIZE: datasetgenerator(files,batch_size),200)
    train.model.generator.save(os.path.join(v.MODELCHKPNT, "face_generator.h5"))
    train.model.discriminator.save(os.path.join(v.MODELCHKPNT, "face_discriminator.h5"))