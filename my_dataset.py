from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    # def __init__(self, images_path: list, images_class: list, transform=None):
    def __init__(self, images_path: list, images_class: list, images_sheet: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.images_sheet = images_sheet
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        path = self.images_path[item]
        img = Image.open(path)
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]
        sheet = self.images_sheet[item]
        if self.transform is not None:
            img = self.transform(img)
        # return img, label
        return img, label, sheet

    @staticmethod
    def collate_fn(batch):
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        # images, labels = tuple(zip(*batch))
        images, labels, sheet = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        sheet = torch.as_tensor(sheet)

        # return images, labels
        return images, labels, sheet
