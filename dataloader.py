from torch.utils import data
import torch
import os
from PIL import Image
import numpy as np

class EvalDataset(data.Dataset):
    def __init__(self, img_root, label_root):
        lst_label = sorted(os.listdir(label_root))
        lst_pred = sorted(os.listdir(img_root))
        lst = []
        for name in lst_label:
            if name in lst_pred:
                lst.append(name)

        self.image_path = list(map(lambda x: os.path.join(img_root, x), lst))
        self.label_path = list(map(lambda x: os.path.join(label_root, x), lst))
        # print(self.image_path)
        # print(self.image_path.sort())
    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        pred_np = np.array(pred)
        pred_np = ((pred_np - pred_np.min()) / (pred_np.max() - pred_np.min()) * 255).astype(np.uint8)
        pred = Image.fromarray(pred_np)

        return pred, gt

    def __len__(self):
        return len(self.image_path)
    
