import os
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import pickle

class MOREPlusDataset(Dataset):
    def __init__(
        self,
        data_file,
        CC_file,
        D_file,
        DC_file,
        O_file,
        OC_file,
        adj_file,
        path_to_images,
        tokenizer,
        image_transform,
        CFG,
    ):

        self.data = pd.read_csv(data_file, sep="\t", encoding="utf-8")

        with open(CC_file, "rb") as f:
            self.CC = pickle.load(f)

        with open(D_file, "rb") as f:
            self.D = pickle.load(f)

        with open(DC_file, "rb") as f:
            self.DC = pickle.load(f)

        with open(O_file, "rb") as f:
            self.O = pickle.load(f)

        with open(OC_file, "rb") as f:
            self.OC = pickle.load(f)

        with open(adj_file, "rb") as f:
            self.adj = pickle.load(f)

        self.path_to_images = path_to_images
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.CFG = CFG

    def __getitem__(self, idx):

        row = self.data.iloc[idx, :]

        pid = row["pid"]
        caption = row["text"]
        text = caption.split(" ")
        
        if pid in self.adj:
            adj = self.adj[pid]
        else:
            adj = np.float32(np.diag([1.0] *256))
        
        #[Section 4.3] Concat caption concept tokens
        if pid in self.CC:
            for key in list(self.CC[pid].keys()):
                text.append(self.CC[pid][key]["concept"])

        #[Section 4.3] Concat image description tokens
        if pid in self.D:
            for word in self.D[pid].split(" "):
                text.append(word)

        #[Section 4.3] Concat image description concepts
        if pid in self.DC:
            for key in list(self.DC[pid].keys()):
                text.append(self.DC[pid][key]["concept"])

        #[Section 4.3] Concat img objects
        if pid in self.O:
            for obj in self.O[pid]["classes"]:
                text.append(obj)

        #[Section 4.3] Concat img object concepts
        if pid in self.OC:
            for key in list(self.OC[pid].keys()):
                text.append(self.OC[pid][key]["concept"])

        caption = " ".join(text) #Section 4.3
        explanation = row["explanation"]
        target_of_sarcasm = row["target_of_sarcasm"]

        max_length = self.CFG.max_len
        
        encoded_dict = self.tokenizer(
            caption + "</s>" + target_of_sarcasm, #[Section 4.5] Incorporating Target
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_prefix_space=True,
        )
        input_ids = encoded_dict["input_ids"][0]
        attention_mask = encoded_dict["attention_mask"][0]

        image_path = os.path.join(self.path_to_images, pid + ".jpg")
        img = np.array(Image.open(image_path).convert("RGB"))
        img_inp = self.image_transform(img)

        encoded_dict = self.tokenizer(
            explanation,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_prefix_space=True,
        )

        explanation_ids = encoded_dict["input_ids"][0]

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_image": img_inp,
            "target_ids": explanation_ids,
            "graph": adj
        }

        return sample

    def __len__(self):
        return self.data.shape[0]
