import os
from glob import glob
from monai.data import DataLoader, Dataset
import monai.transforms as tr
from monai.transforms import MapTransform
from utils import read_yaml_file
import nibabel as nib
import numpy as np
import monai
import torch


def get_raw_data_nibabel(nifti_file):
    img = nib.load(nifti_file)
    # Get the image data as a NumPy array
    data = img.get_fdata()
    return data


def check_for_2d(nifti_file_dir):
    nifti_sample = get_raw_data_nibabel(nifti_file_dir)
    if len(np.squeeze(nifti_sample).shape) == 2:
        return True
    else:
        return False
    

def dataloader_input_list_of_dicts(image_dirs, label_dirs):
    return [{"image": image_name, "label": label_name} for
                        image_name, label_name in zip(image_dirs, label_dirs)]


class SliceWiseNormalizeIntensityd(MapTransform):
    def __init__(self, keys, subtrahend=0.0, divisor=None, nonzero=True):
        super().__init__(keys)
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.nonzero = nonzero

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            for i in range(image.shape[-1]):
                slice_ = image[..., i]
                if self.nonzero:
                    mask = slice_ > 0
                    if np.any(mask):
                        if self.subtrahend is None:
                            slice_[mask] = slice_[mask] - slice_[mask].mean()
                        else:
                            slice_[mask] = slice_[mask] - self.subtrahend

                        if self.divisor is None:
                            slice_[mask] /= slice_[mask].std()
                        else:
                            slice_[mask] /= self.divisor

                else:
                    if self.subtrahend is None:
                        slice_ = slice_ - slice_.mean()
                    else:
                        slice_ = slice_ - self.subtrahend

                    if self.divisor is None:
                        slice_ /= slice_.std()
                    else:
                        slice_ /= self.divisor

                image[..., i] = slice_
            d[key] = image
        return d

def get_2d_dataloader_train(images_dir, labels_dir, config):
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

    _files = dataloader_input_list_of_dicts(images_dir, labels_dir)
    _ds = Dataset(data=_files, transform=transformations)
    data_loader = DataLoader(_ds, batch_size=config["batch_size"], shuffle=True, num_workers=2)
    return data_loader


def get_3d_dataloader_train(images_dir, labels_dir, config):
    volumetric_transforms = tr.Compose(
        [
            tr.LoadImaged(keys=["image", "label"]),
            tr.EnsureChannelFirstd(keys=["image", "label"]),
            tr.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, -1.0), mode=["bilinear", "nearest"]),
            SliceWiseNormalizeIntensityd(keys=["image"], nonzero=True),
            tr.ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(config["img_size"], config["img_size"], -1)),
            # Split 3D image into 2D slices along the z-axis (or another axis if needed)
            tr.SplitDimd(keys=["image", "label"], dim=-1),  # Splits the last dimension (z-axis) into separate slices        
            # Ensure the channel dimension is there for 2D network compatibility
            # tr.AddChanneld(keys=["image"]),
        ]
    )

    patch_func = monai.data.PatchIterd(
        keys=["image", "label"],
        patch_size=(None, None, 1),  # dynamic first two dimensions
        start_pos=(0, 0, 0)
    )
    patch_transform = tr.Compose(
        [
            tr.SqueezeDimd(keys=["image", "label"], dim=-1),  # squeeze the last dim
            # tr.Resized(keys=["image", "label"], spatial_size=[config["img_size"], config["img_size"]]),
        ]
    )
    _files = dataloader_input_list_of_dicts(images_dir, labels_dir)
    volume_ds = monai.data.CacheDataset(data=_files, transform=volumetric_transforms)
    patch_ds = monai.data.GridPatchDataset(
        data=volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
    slicewise_dataloader = DataLoader(
        patch_ds,
        batch_size=config['batch_size'],
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return slicewise_dataloader


def get_dataloaders(config):
    # load train data
    train_images = sorted(glob(os.path.join(config["images_dir"], "train_slice", f"images/{config['modality_type']}*.nii.gz")))
    train_labels = sorted(glob(os.path.join(config["images_dir"], "train_slice", f"masks/{config['modality_type']}*.nii.gz")))
    if check_for_2d(train_images[0]):
        train_loader = get_2d_dataloader_train(train_images, train_labels, config)
    else:
        train_loader = get_3d_dataloader_train(train_images, train_labels, config)

    # load validation data
    validation_images = sorted(glob(os.path.join(config["images_dir"], "validation_slice", f"images/{config['modality_type']}*.nii.gz")))
    validation_labels = sorted(glob(os.path.join(config["images_dir"], "validation_slice", f"masks/{config['modality_type']}*.nii.gz")))
    if check_for_2d(validation_images[0]):
        val_loader = get_2d_dataloader_train(validation_images, validation_labels, config)
    else:
        val_loader = get_3d_dataloader_train(validation_images, validation_labels, config)
        
    return train_loader, val_loader







if __name__ == "__main__":
    config = read_yaml_file("config.yaml")
    train_loader, val_loader = get_dataloaders(config)
    # print(f"len train loader: {len(train_loader)}, val loader: {len(val_loader)}")
    for batch in val_loader:
        imgs, labels = batch['image'], batch['label']
        print(f"images => shape: {imgs.shape}, min: {imgs.min()}, max: {imgs.max()}")
        print(f"labels => shape: {labels.shape}, min: {labels.min()}, max: {labels.max()}")
    # x = labels>0  labels<1
    x = labels[labels<1]
    print(x[x>0])
    # print(labels[x])
    # if (labels>0).any() or (labels<1).any():
    #     print('shit')
    #     print(labels)