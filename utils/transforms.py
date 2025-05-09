# transforms.py

import torchvision.transforms as T
from torchvision.transforms import functional as F
import random

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

def get_transform(train=True):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        # Optional: Add data augmentations like horizontal flips
        transforms.append(T.RandomHorizontalFlip(0.5))
    return Compose(transforms)
