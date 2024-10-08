
# bin_dplinknet

This repository contains the code for image binarization of handwritten papyrus documents.

The project extends and optimizes the following repositories:
- [ImageBinarization by Agni Ramadani](https://github.com/agniramadani/ImageBinarization)
- [DP-LinkNet by beargolden](https://github.com/beargolden/DP-LinkNet)

The model is based on an optimized implementation of the DP-LinkNet network and introduces a patch-based method for binarization, where the document is processed in patches and then reconstructed from the binarized versions of all patches.

## Setup Environment

To set up the working environment, you'll need the [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) environment manager.

First, initialize your environment with:
```bash
conda env create -f environment.yml
```

Then, activate it using:
```bash
conda activate bin_dlnet
```

## Running the Binarization

Ensure that the images you want to binarize are uploaded into the `uploaded_images` folder.

Next, run the following command:
```bash
python main.py
```

The binarized images will be saved in the `binarized_images` folder.


---
---
----
## Recent Updates
- Resolved GPU usage issues.
- Image patches of size (256,256) are now applied (`#TILE_SIZE`).
- The size of the output binary image matches the original image size.
- The binarization threshold (`BIN_THRESHOLD`) can now be customized.

## TODO
- Make the `BIN_THRESHOLD` parameter trainable (this may require implementing a specific loss function).

---
---
---
