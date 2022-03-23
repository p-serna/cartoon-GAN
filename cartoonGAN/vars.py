
DEV=True

TRAINFOLDER1="data/cartoonset10k"
TRAINFOLDER2="data/cartoonset100k"
if DEV:
  TRAINFOLDER = TRAINFOLDER1
else:
  TRAINFOLDER = TRAINFOLDER2
DATAST = f"data/processeddata"
get_par = lambda fname : fname.split(".")[0][2:]

AVATARSIZE = (96,96)
RELOADMEANNSTD = False