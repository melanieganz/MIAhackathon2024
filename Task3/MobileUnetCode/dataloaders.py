import os
import random
import sys
from glob import glob
from monai.data import decollate_batch, DataLoader, Dataset
from monai.metrics import DiceMetric
import monai.transforms as tr
from utils import read_yaml_file

def get_dataloaders(config):
    transformations = tr.Compose(
        [
            tr.LoadImaged(keys=["image", "label"]),
            tr.EnsureChannelFirstd(keys=["image", "label"]),
            tr.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, -1.0), mode=("bilinear", "nearest")),
            tr.SqueezeDimd(keys=["image", "label"], dim=-1, update_meta=True),
            tr.NormalizeIntensityd(keys=["image"], nonzero=True),
            tr.Resized(keys=["image", "label"], mode=("bilinear", "nearest"), spatial_size=(config["img_size"], config["img_size"])),
        ]
    )

    # load train data
    train_images = sorted(glob(os.path.join(config["images_dir"], "train_slice", f"images/{config['modality_type']}*.nii.gz")))
    train_labels = sorted(glob(os.path.join(config["images_dir"], "train_slice", f"masks/{config['modality_type']}*.nii.gz")))
    train_files = [{"image": image_name, "label": label_name} for
                    image_name, label_name in zip(train_images, train_labels)]
    train_ds = Dataset(data=train_files, transform=transformations)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=8)

    # load validation data
    validation_images = sorted(glob(os.path.join(config["images_dir"], "validation_slice", f"images/{config['modality_type']}*.nii.gz")))
    validation_labels = sorted(glob(os.path.join(config["images_dir"], "validation_slice", f"masks/{config['modality_type']}*.nii.gz")))
    validation_files = [{"image": image_name, "label": label_name} for
                        image_name, label_name in zip(validation_images, validation_labels)]
    val_ds = Dataset(data=validation_files, transform=transformations)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], num_workers=4)

    # define metrics
    # dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    # post_pred = tr.Compose([tr.AsDiscrete(argmax=True, to_onehot=config["num_classes"])])
    # post_label = tr.Compose([tr.AsDiscrete(to_onehot=config["num_classes"])])

    return train_loader, val_loader

if __name__ == "__main__":
    config = read_yaml_file("config.yaml")
    train_loader, val_loader = get_dataloaders(config)
    print(f"len train loader: {len(train_loader)}, val loader: {len(val_loader)}")
    batch = next(iter(val_loader))
    imgs, labels = batch['image'], batch['label']
    print(f"images => shape: {imgs.shape}, min: {imgs.min()}, max: {imgs.max()}")
    print(f"images => shape: {labels.shape}, min: {labels.min()}, max: {labels.max()}")
    # x = labels>0  labels<1
    x = labels[labels<1]
    print(x[x>0])
    # print(labels[x])
    # if (labels>0).any() or (labels<1).any():
    #     print('shit')
    #     print(labels)