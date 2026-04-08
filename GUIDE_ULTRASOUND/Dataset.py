import os
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image


class DatasetGaze(Dataset):
    def __init__(self, data_dir, csv_path, phase, img_size):
        self.csv = pd.read_csv(csv_path, encoding='gbk', dtype={'image_id': str})
        if phase == 'train':
            self.data = self.csv[self.csv['train_test'] == 'train']
        elif phase == 'test':
            self.data = self.csv[self.csv['train_test'] == 'test']
        else:
            raise ValueError("phase must be 'train' or 'test'")

        img_folder = os.path.join(data_dir, 'img')
        gaze_folder = os.path.join(data_dir, 'gaze')

        self.img_paths = [os.path.join(img_folder, f"{img_id}.png") for img_id in self.data['image_id']]
        self.gaze_paths = [os.path.join(gaze_folder, f"{img_id}.png") for img_id in self.data['image_id']]
        self.image_to_class = dict(zip(self.data['image_id'], self.data['class_id']))
        self.phase = phase
        self.transform = get_transform(phase, img_size)

    def __getitem__(self, index):
        img_path, gaze_path = self.img_paths[index], self.gaze_paths[index]
        img = np.array(Image.open(img_path).convert('L'))
        gaze = np.array(Image.open(gaze_path).convert('L'))

        if img.shape != gaze.shape:
            print(f"Warning: Image and mask size mismatch at index {index}")
            print(f"Image shape: {img.shape}, Mask shape: {gaze.shape}")
            print(f"Image path: {img_path}, Mask path: {gaze_path}")

        transformed = self.transform(image=img, mask=gaze)
        img = transformed["image"].float()
        gaze = transformed["mask"].float()

        img = img / 255.0
        gaze = gaze / 255.0

        img = (img - img.mean()) / (img.std() + 1e-8)
        gaze = (gaze - gaze.mean()) / (gaze.std() + 1e-8)

        gaze = gaze.unsqueeze(0)

        name = os.path.basename(img_path).split('.png')[0]
        class_id = self.image_to_class[name]
        cls = 0 if class_id == 0 else 1

        return img, gaze, cls, img_path

    def __len__(self):
        return len(self.img_paths)

class Dataset_nogaze(Dataset):
    def __init__(self, data_dir, csv_path, phase, img_size):
        self.csv = pd.read_csv(csv_path, encoding='gbk', dtype={'image_id': str})

        if phase == 'train':
            self.data = self.csv[self.csv['train_test'] == 'train']
        elif phase == 'test':
            self.data = self.csv[self.csv['train_test'] == 'test']
        else:
            raise ValueError("phase must be 'train' or 'test'")

        img_folder = os.path.join(data_dir, 'img')
        self.img_paths = [os.path.join(img_folder, f"{img_id}.png") for img_id in self.data['image_id']]
        self.image_to_class = dict(zip(self.data['image_id'], self.data['class_id']))

        self.phase = phase
        self.transform = get_transform(phase, img_size)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = np.array(Image.open(img_path).convert('L'))

        transformed = self.transform(image=img)
        img = transformed["image"].float()
        img = img / 255.0
        img = (img - img.mean()) / (img.std() + 1e-8)

        name = os.path.basename(img_path).split('.png')[0]
        class_id = self.image_to_class[name]
        cls = 0 if class_id == 0 else 1

        return img, cls, img_path

    def __len__(self):
        return len(self.img_paths)


def get_transform(phase, img_size):
    if phase == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(),
            A.ShiftScaleRotate(rotate_limit=15),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            ToTensorV2(),
        ])
