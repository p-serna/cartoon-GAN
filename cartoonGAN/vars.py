
DEV=True

TRAINFOLDER1="data/cartoonset10k"
TRAINFOLDER2="data/cartoonset100k"
if DEV:
  TRAINFOLDER = TRAINFOLDER1
else:
  TRAINFOLDER = TRAINFOLDER2
DATAST = f"data/processeddata"
get_par = lambda fname : fname.split(".")[0][2:]

AVATARSIZE = (128,128)
INPUT_SHAPE = (*AVATARSIZE,4)
RELOADMEANNSTD = False
SEEDSIZE= 10
BATCH_SIZE = 128
GENR_LR = 1.5e-4
DISC_LR = 1.5e-4

MODELCHKPNT = "training/checkpoints"

PREVIEW_ROWS=2 
PREVIEW_COLS=5