import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms


class CustomDataset(Dataset):

    def __init__(self, image_dir, mask_dir, image_height=256, image_width=256, transform=None):
        self.img_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.ids = os.listdir(image_dir)
        self.image_height = image_height  # 1024 originally
        self.image_width = image_width  # 512 originally

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.ids[idx])
        mask_path = os.path.join(self.mask_dir, self.ids[idx].replace(".png", "_label.png"))
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 2] = 1.0
        if self.transform is not None:  # In this project, no augmentation is used.
            augmentations = self.transform(img=img, mask=mask)
            img = augmentations['image']
            mask = augmentations['mask']
        else:
            img.resize([self.image_height, self.image_width, img.shape[2]])
            mask.resize([self.image_height, self.image_width])
            tensor_creator = transforms.Compose([transforms.ToTensor()])
            img = tensor_creator(img)
            mask = mask.astype(np.float32)

        return img, mask

    def __len__(self):
        return len(self.ids)


def test():
    script_dir = os.path.dirname(__file__)
    train_img_dir = os.path.join(script_dir, "Data/train")
    train_mask_dir = os.path.join(script_dir, "Data/train_label")
    val_img_dir = os.path.join(script_dir, "Data/val")
    val_mask_dir = os.path.join(script_dir, "Data/val_label")
    sample_dataset = CustomDataset(train_img_dir, train_mask_dir)
    image, mask = sample_dataset[0]
    print(image.shape, mask.shape)
    print(len(mask[mask!=0]))


if __name__ == "__main__":
    test()
