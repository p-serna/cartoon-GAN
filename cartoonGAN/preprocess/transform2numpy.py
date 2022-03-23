import numpy as np 
import os,sys
from cartoonGAN import vars as v
import imageio 
import cv2
import pandas as pd

img_names = [v.get_par(f) for f in os.listdir(v.TRAINFOLDER) if f.split(".")[-1]=="png"]
def csv2array(csv):
  csvd = csv.sort_values(by=[0])
  d = [csvd.loc[i,1]/csvd.loc[i,2] for i, f in enumerate(csvd.loc[:,0].values)]
  return np.array(d)

def read_files(img_names, nmax=1000):

  X,Y = [],[]
  for fname in img_names:
    fimg,fsv = f"{v.TRAINFOLDER}/cs{fname}.png",f"{v.TRAINFOLDER}/cs{fname}.csv"
    img = imageio.imread(fimg)
    img = cv2.resize(img, v.AVATARSIZE)
    csv = pd.read_csv(fsv,header=None)
    csvd = csv2array(csv)
    if csvd.shape[0]!=18: 
      raise Exception("No todos tienen 17")
    Y.append(csvd)
    X.append(img)
    if len(Y)>=nmax:
      yield np.stack(X),np.stack(Y)
      X,Y = [],[]

def get_pars():
  if v.RELOADMEANNSTD:
    gen = read_files(img_names)
    m,sd = [],[]
    for i, (X,Y) in enumerate(gen):
      m.append(X.mean())
      sd.append(X.std())
    m = np.array(m)
    sd = np.array(sd)
    np.savez_compressed(f"data/pars0.npz",mean=m,std=sd)
  else:
    d = np.load("data/pars0.npz")
    m = d["mean"]
    sd = d["std"]
  return np.array(m),np.array(sd)



def preprocess_data(m=None,sd=None):
  if m is None:
    m,sd = get_pars()
    m = m.mean()
    sd = sd.mean()

  gen = read_files(img_names)
  for i, (X,Y) in enumerate(gen):
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[%-10s] %d%%" % ('='*i, 10*i))
    sys.stdout.flush()
    X = (X-m)/sd

    np.savez_compressed(f"{v.DATAST}/data_{i:03d}.npz",X=X,Y=Y)


if __name__=="__main__":
  pass