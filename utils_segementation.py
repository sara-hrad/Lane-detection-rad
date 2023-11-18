import torch
import torchvision
from torch.utils.data import DataLoader

from dataset import CustomDataset


def get_loaders(
    train_dir,
    train_mask_dir,
    val_dir,
    val_mask_dir,
    img_height,
    img_width,
    batch_size,
    num_workers=4,
    pin_memory=True,
):
    train_dataset = CustomDataset(
        image_dir=train_dir,
        mask_dir=train_mask_dir,
        image_height=img_height,
        image_width=img_width
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_dataset = CustomDataset(
        image_dir=val_dir,
        mask_dir=val_mask_dir,
        image_height=img_height,
        image_width=img_width
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}mask_{idx}.png")
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}pred_{idx}.png"
        )

    model.train()

