import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from cartoonGAN.train import *
import numpy as np

files = [f"{v.DATAST}/{f}" for f in os.listdir(v.DATAST) if f.split(".")[-1]=="npy" and f.split("/")[-1][:4]=="data"]
def datasetgenerator(batch_size=v.BATCH_SIZE):
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

datagen = datasetgenerator()
train(datasetgenerator,100)

#generator.save(os.path.join(DATA_PATH, "face_generator.h5"))