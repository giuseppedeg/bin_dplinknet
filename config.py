from networks.dplinknet import LinkNet34, DLinkNet34, DPLinkNet34

# DIRECTORIES
IMG_INDIR = "uploaded_images"
IMG_OUTDIR = "binarized_images"


# CONFIGS
VIEW_PREVIEW_IMAGES = False   # Preview of the binarized images at running time

BATCHSIZE_PER_CARD = 2
TILE_SIZE = int(256)     # Dimension of the paches
BIN_THRESHOLD = 6#5.0    # Threshold for last binarization (TO BE LEARNED)

DATA_NAME = "DIBCO"      # Sataset for training. chose between:  BickleyDiary, DIBCO
DEEP_NETWORK_NAME = DPLinkNet34      # Model to use. Choose between: LinkNet34, DLinkNet34, DPLinkNet34