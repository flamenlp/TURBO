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

        #Returns adjacency matrix of KG for sample with index = i
    def construct_kg(self, idx):
        row = self.data.iloc[idx, :]

        #Used to index all other info
        pid = row['pid']
        caption = row['text']
        caption = caption.split(' ')
        target = row['target_of_sarcasm']

        #C + CC + D + DC + O + OC
        adj = np.diag([1.0] * 256)
        N = 0  #Track current number of tokens

        #C - Caption
        for j in range(len(caption) - 1):
          adj[j][j+1] = 1
          adj[j+1][j] = 1
        N += len(caption)

        #CC - Caption concepts 
        if (pid in self.CC):
          concept_dict = self.CC[pid]
          concept_keys = list(concept_dict.keys())

          for j in range(len(caption)):
            word = caption[j]
            if (word in concept_keys):
              adj[j][N + concept_keys.index(word)] = concept_dict[word]['weight'] 
              adj[N + concept_keys.index(word)][j] = concept_dict[word]['weight']

          N += len(concept_keys)


        #D - Image description
        if (pid in self.D):
          img_caption_start = N
          img_caption_tokens = self.D[pid].split(' ')

          #Connecting adjacent tokens - WORKS
          for j in range(N, N + len(img_caption_tokens)-1):
            adj[j][j+1] = 1
            adj[j+1][j] = 1
          N += len(img_caption_tokens)

          #Connecting img desc tokens to concept tokens - WORKS
          if (pid in self.DC):
            ic_concept_dict = self.DC[pid]
            ic_concept_keys = list(ic_concept_dict.keys())
            for j in range(len(img_caption_tokens)):
              word = img_caption_tokens[j]
              if (word in ic_concept_keys):

                adj[img_caption_start + j][N + ic_concept_keys.index(word)] = ic_concept_dict[word]['weight'] 
                adj[N + ic_concept_keys.index(word)][img_caption_start + j] = ic_concept_dict[word]['weight']
            N += len(ic_concept_keys)

        #O - Image obj
        if (pid in self.O):
          img_obj_start = N
          img_obj_tokens = self.O[pid]['classes']
          img_obj_conf_scores = self.O[pid]['confidence_scores']

          N += len(img_obj_tokens)
            
        #OC - Image obj concepts
          if (pid in self.OC):
            io_concept_dict = self.OC[pid]
            io_concept_keys = list(io_concept_dict.keys())

            for j in range(len(img_obj_tokens)):
              word = img_obj_tokens[j]

              if (word in io_concept_keys):
                adj[img_obj_start + j][N + io_concept_keys.index(word)] = io_concept_dict[word]['weight']
                adj[N + io_concept_keys.index(word)][img_obj_start + j] = io_concept_dict[word]['weight']

            N += len(io_concept_keys)

        return np.float32(adj)
    
    def __getitem__(self, idx):

        row = self.data.iloc[idx, :]

        pid = row["pid"]
        caption = row["text"]
        text = caption.split(" ")
        adj = self.construct_kg(idx)
        
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
