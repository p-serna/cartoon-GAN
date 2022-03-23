import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from cartoonGAN.train import *
import numpy as np
t = np.load("data/processeddata/data_000.npz")
X, Y = t["X"], t["Y"]
X.shape

files = [f"{v.DATAST}/{f}" for f in os.listdir(v.DATAST) if f.split(".")[-1]=="npz"]
def datasetgenerator(batch_size=v.BATCH_SIZE):
  X, Y = None,None
  for file in files:
    temp = np.load(file)
    X0, Y0 = t["X"],t["Y"]
    if X is None:
      X,Y = X0,Y0
    else:
      X = np.concatenate((X,X0))
      Y = np.concatenate((Y,Y0))
    while X.shape[0]>batch_size:
      yield X[:batch_size,:]#,Y[:batch_size,:]
      X,Y = X[batch_size:,:],Y[batch_size:,:]

datagen = datasetgenerator()
train(datagen,20)