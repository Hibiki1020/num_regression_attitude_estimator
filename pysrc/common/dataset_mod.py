import torch.utils.data as data
from PIL import Image
import numpy as np

class Originaldataset(data.Dataset):
    def __init__(self, data_list, transform, phase):
        self.data_list = data_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index][0]

        roll_str = self.data_list[index][4]
        pitch_str = self.data_list[index][5]
        yaw_str = self.data_list[index][6]

        deg_list = [float(roll_str), float(pitch_str), float(yaw_str)]

        img_pil = Image.open(img_path)
        img_pil = img_pil.convert("RGB")
        deg_numpy = np.array(deg_list)

        img_trans, deg_trans = self.transform(img_pil, deg_numpy, phase=self.phase)
        return img_trans, deg_trans