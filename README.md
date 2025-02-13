# TURBO
This repository contains the code for our state-of-the-art sarcasm explanation model, `TURBO`, as proposed in our paper titled "Target-Augmented Shared Fusion-based Multimodal Sarcasm Explanation Generation". 

## Dataset
This model is trained on the `MORE+` dataset which is an extension of the `MORE` dataset proposed by Desai et al. (2022). All of the data files for `MORE+` are available in the [`Dataset/`](/Dataset) directory except for the image files, which can be downloaded from the [original Github repo for the `MORE` dataset](https://github.com/LCS2-IIITD/Multimodal-Sarcasm-Explanation-MuSE). Put all of the images into a directory named `images/` inside the [`Dataset/`](/Dataset) directory of this repository.

## Training
`TURBO` can be trained by running the following in a terminal:
```bash
python train.py
```
