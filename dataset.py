import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# 讀取物件列表（例如 objects.json 是一個 list）
def load_object_list(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

# 多標籤轉 multihot 向量

def labels_to_multihot(labels, obj2idx):
    vec = torch.zeros(len(obj2idx), dtype=torch.float32)
    for label in labels:
        if label in obj2idx:
            vec[obj2idx[label]] = 1.0
    return vec

class ICLEVRDataset(Dataset):
    def __init__(self, root_dir, json_path, object_json="objects.json", image_size=64):
        super().__init__()
        self.root_dir = root_dir

        # JSON 是 list 格式，每筆含 filename 和 attributes
        with open(json_path, "r") as f:
            self.annotations = json.load(f)

        # 讀取物件名 → index 對應表
        self.object_list = load_object_list(object_json)
        self.obj2idx = {name: idx for idx, name in enumerate(self.object_list)}

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        image_path = os.path.join(self.root_dir, item["filename"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        labels = item["attributes"]
        cond_vec = labels_to_multihot(labels, self.obj2idx)

        return image, cond_vec

class TestConditionDataset(Dataset):
    def __init__(self, json_path, obj2idx):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.obj2idx = obj2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        labels = self.data[idx]
        return labels_to_multihot(labels, self.obj2idx)

# 轉換函式方便使用

def load_obj2idx(json_path):
    obj_list = load_object_list(json_path)
    return {name: idx for idx, name in enumerate(obj_list)}, {idx: name for idx, name in enumerate(obj_list)}
