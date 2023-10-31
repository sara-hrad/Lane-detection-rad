import torch
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from UNET_Model import UNET
from utils_segementation import (
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Dataset Directories
script_dir = os.path.dirname(__file__)
train_img_dir = os.path.join(script_dir, "Data/train")
train_mask_dir = os.path.join(script_dir, "Data/train_label")
val_img_dir = os.path.join(script_dir, "Data/val")
val_mask_dir = os.path.join(script_dir, "Data/val_label")

# Hyperparameters etc.
learning_rate = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
num_epoch = 15
num_workers = 2
img_height = 256  # 1024 originally, Please select a number that is divisible by 16
img_width = 256  # 516 originally, Please select a number that is divisible by 16
pin_memory = True
load_model = False  # For training, use False, and for testing, use True


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader = get_loaders(
        train_img_dir,
        train_mask_dir,
        val_img_dir,
        val_mask_dir,
        img_height,
        img_width,
        batch_size,
        num_workers,
        pin_memory
    )
    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epoch):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint_to_save = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        print("=> Saving checkpoint")
        torch.save(checkpoint_to_save, "my_checkpoint.pth.tar")

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )


def test():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    batch_size_loading = 32
    train_loader, val_loader = get_loaders(
        train_img_dir,
        train_mask_dir,
        val_img_dir,
        val_mask_dir,
        img_height,
        img_width,
        batch_size_loading,
        num_workers,
        pin_memory
    )
    checkpoint_to_load = torch.load("my_checkpoint.pth.tar")
    model.load_state_dict(checkpoint_to_load["state_dict"])
    check_accuracy(val_loader, model, device=DEVICE)
    save_predictions_as_imgs(
        val_loader, model, folder="Testing/", device=DEVICE
    )


if __name__ == "__main__":
    if load_model:
        test()
    else:
        main()

