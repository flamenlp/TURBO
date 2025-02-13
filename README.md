# TURBO
This repository contains the code for our state-of-the-art sarcasm explanation model, `TURBO`, as proposed in our paper titled "Target-Augmented Shared Fusion-based Multimodal Sarcasm Explanation Generation". 

This model is trained on the `MORE+` dataset which is an extension of the `MORE` dataset proposed by Desai et al. (2022). All of the data files for `MORE+` are available in the `Dataset` directory except for the image files, which can be downloaded from the [original Github repo for the `MORE` dataset](https://github.com/LCS2-IIITD/Multimodal-Sarcasm-Explanation-MuSE). In order to train `TURBO`, download all of the image files and put them into a directory named `images\` in the root directory of this repository.

Once the above has been done, `TURBO` can be trained by running the following in a terminal:
```bash
python train.py
```
