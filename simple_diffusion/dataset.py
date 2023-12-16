import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import webdataset as wds
import datasets


class CustomDataset(Dataset):

    def __init__(self, data_df, transforms):
        image_paths = []
        for idx, row in data_df.iterrows():
            image_path = row["image_path"]

            image_paths.append(image_path)
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        return {"image": image}
    
# coco + textcap + sbucaptions
class CombinedDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data,
                 transforms,
                 shuffle_captions=True):

        self.data = data
        self.transforms = transforms
        self.shuffle_captions = shuffle_captions

    def __getitem__(self, idx):
        row = self.data[idx]
        image = row["image"]

        instance = {"image": self.transforms(image)}

        return instance

    def __len__(self):
        return len(self.data)
    
def get_dataset(dataset_name,
                dataset_path,
                transforms):
    if dataset_name == "combined":
        data_coco = datasets.load_from_disk(f"{dataset_path}/coco")["train"]
        data_textcap = datasets.load_from_disk(f"{dataset_path}/textcap")["train"]
        data_sbu = datasets.load_from_disk(f"{dataset_path}/sbu_captions_images")
        data_sbu = data_sbu.map(lambda example: {"caption": [example["caption"]]})
        data = datasets.concatenate_datasets([data_coco, data_textcap, data_sbu])
        return CombinedDataset(data,
                               transforms=transforms)
    elif dataset_name == "yfcc7m":
        def prepare_data(x):
            image = transforms(x["jpg"])
            instance = {"image": image}            
            return instance
        
        dataset = wds.WebDataset([f"{dataset_path}/yfcc7m/{i:05d}.tar" for i in range(1538)], shardshuffle=True, cache_dir=f"{dataset_path}/yfcc7m_training_cache")
        dataset = dataset.shuffle(1000, initial=100).decode("pil").map(prepare_data)
        return dataset
    raise Exception(f"Invalid dataset name {dataset_name} - options are [coco, sbucaptions, combined, yfcc7m]")
