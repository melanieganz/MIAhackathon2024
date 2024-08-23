#%%
# modified from "https://github.com/Project-MONAI/tutorials/blob/main/2d_segmentation/torch/unet_training_array.py"
import os
import random
import sys
from glob import glob

import torch

import monai
from monai.data import decollate_batch, DataLoader, Dataset
from monai.inferers import SimpleInferer
from monai.metrics import DiceMetric
import monai.transforms as tr

import warnings
from carbontracker.tracker import CarbonTracker
from UniverSeg.universeg import universeg

warnings.filterwarnings('ignore')
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import nibabel as nib
import PIL
import torch
from torch.utils.data import Dataset
from PIL import Image 


def process_img(path: pathlib.Path, size: Tuple[int, int]):
    img = (nib.load(path).get_fdata() * 255).astype(np.uint8).squeeze()
    img = PIL.Image.fromarray(img)
    img = img.resize(size, resample=PIL.Image.BILINEAR)
    # img = img.convert("L")
    img = np.array(img)
    img = img.astype(np.float32)/255
    # img = np.rot90(img, -1)
    return img.copy()


def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    seg = nib.load(path).get_fdata().squeeze().round().clip(0,1)
    seg = PIL.Image.fromarray(seg)
    seg = seg.resize(size, resample=PIL.Image.NEAREST)
    seg = np.array(seg)
    seg = seg.astype(np.float32)
    # seg = np.rot90(seg, -1)
    return seg.copy()


def dice_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred = y_pred.long()
    y_true = y_true.long()
    score = 2*(y_pred*y_true).sum() / (y_pred.sum() + y_true.sum())
    return score.item()

def load_folder(path: pathlib.Path, mri_modality: str = 'FMRI', size: Tuple[int, int] = (128, 128), label: bool=False):
    data = []
    for file in sorted(path.glob(f"{mri_modality}*.nii.gz")):
        if label:
            img = process_seg(file, size=size)
        else:
            img = process_img(file, size=size)
        data.append(torch.from_numpy(img)[None])
    return data

def inference(model, mri_modality, device):
    train_images = torch.stack(load_folder(pathlib.Path(os.path.join(args.training_data_path, "images")), mri_modality=mri_modality)).to(device)
    train_labels = torch.stack(load_folder(pathlib.Path(os.path.join(args.training_data_path, "masks")), mri_modality=mri_modality, label=True)).to(device)
    
    validation_images = torch.stack(load_folder(pathlib.Path(os.path.join(args.validation_data_path, "images")), mri_modality=mri_modality)).to(device)
    validation_labels = torch.stack(load_folder(pathlib.Path(os.path.join(args.validation_data_path, "masks")), mri_modality=mri_modality, label=True)).to(device)
    dice_list = list()
    for val_im, val_lab in zip(validation_images, validation_labels):
        logits = model(
                val_im[None],
                train_images[:128][None],
                train_labels[:128][None]
            )[0]
        pred = torch.sigmoid(logits)
        # print('logit shape: ', logits.shape)
        dice_list.append(dice_score(pred.round().clip(0,1), val_lab))
    print(f"############# {mri_modality} #############")
    print(f"dice score => mean: {np.array(dice_list).mean()}, std: {np.array(dice_list).std()}")

def train(args):
    # tracker = CarbonTracker(epochs=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = universeg(pretrained=True).to(device)
    # tracker.epoch_start()
    for mri_modality in ["DWI", "FMRI", "T2W"]:
        inference(model, mri_modality, device)
    
    # im1 = PIL.Image.fromarray(pred[0].detach().numpy())
    # im1 = im1.convert('L')

    # im1 = im1.save("pred.png")

    # tracker.epoch_end()
    # tracker.stop()



import argparse




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0", help="Set the device to run the program")
    parser.add_argument('--training_data_path', default="data/train_slice", help="Set the path to training dataset")
    parser.add_argument('--validation_data_path', default="data/validation_slice", help="Set the path to validation dataset")
    parser.add_argument('--testing_data_path', type=str, help="Set the path to testing data (for internal testing")
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr', default=0.1, type=float, help="Learning rate")
    parser.add_argument('--num_classes', type=int, default=2, help="Number of classes / number of output layers")
    parser.add_argument('--img_size', type=int, default=128, help='input patch size of network input')
    parser.add_argument('--train', type=bool, default=True, help="Use True for training")
    parser.add_argument('--test', type=bool, default=False, help="Use True for testing")
    parser.add_argument('--model_path', type=str, help="Set the path of the model to be tested")
    parser.add_argument('--test_save_path', type=str, help="Set the path to save data")
    args = parser.parse_args()
    print(args)
    if args.train:
        if args.training_data_path is None:
            raise TypeError(
                "Please specify the path to the training data by setting the parameter "
                "--training_data_path=\"path_to_training_data\"")
        else:
            train(args)

    elif args.test:
        if args.model_path is None:
            raise TypeError("Please specify the path to model by setting the parameter --model_path=\"path_to_model\"")
        else:
            if args.testing_data_path is None:
                raise TypeError(
                    "Please specify the path to the testing data by setting the parameter "
                    "--testing_data_path=\"path_to_testing_data\"")
            else:
                pass
                # test(args)

    else:
        raise TypeError(
                    "Please specify the process by setting the parameter "
                    "--train or --test to \"True\"")
# %%
