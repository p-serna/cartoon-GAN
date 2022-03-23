import matplotlib.pyplot as plt
import os
from cartoonGAN import vars as v
import imageio,random 

img_names = None
#img_names = [v.get_par(f) for f in os.listdir(v.TRAINFOLDER) if f.split(".")[-1]=="png"]
def get_images():
  img_names = [v.get_par(f) for f in os.listdir(v.TRAINFOLDER) if f.split(".")[-1]=="png"]
  return img_names

def show_n_random(img_names=None,n=3,max_x=5,**kwargs):
  if img_names is None:
    img_names = get_images()
  nrows = (n-1)//max_x+1
  ncols = min(n,max_x)
  fig,axs = plt.subplots(nrows,ncols,**kwargs)
  axs = axs.flatten()
  for fname, ax in zip(random.sample(img_names,n),axs):
    img = imageio.imread(f"{v.TRAINFOLDER}/cs{fname}.png")
    ax.imshow(img)
    ax.set_axis_off()
  return fig,axs


if __name__=="__main__":
  f,axs = show_n_random(n=10,max_x=5,figsize=(6,3)); 
  plt.tight_layout()
  f.savefig("figs/sample.png")