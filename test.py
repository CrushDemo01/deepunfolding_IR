import os
from datetime import datetime

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import sys
sys.path.append("/home/data/sds/projects/High-Engneer-Math")
from utils.metrics import calculate_ssim_psnr
from model.model import DeepUnfoldingMethod
from dataset.blurimage import BlurryImageDataset

from tqdm import tqdm
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
import argparse

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)



parser = argparse.ArgumentParser(description='Image Deblurring')

parser.add_argument('--train_dir', default='/home/data/sds/datasets/reflection-removal/train/VOCdevkit/VOC2012PNG1k', type=str, help='Directory of train images')
parser.add_argument('--val_dir', default='/home/data/sds/datasets/reflection-removal/test/SIR2/WildSceneDataset/transmission_layer', type=str, help='Directory of validation images')
parser.add_argument('--model_save_dir', default='/home/data/sds/projects/High-Engneer-Math/checkpoints', type=str, help='Path to save weights')
parser.add_argument('--pretrain_weights', default='/home/data/sds/projects/High-Engneer-Math/checkpoints/model_best.pth', type=str, help='Path to pretrain-weights')
parser.add_argument('--num_epochs', default=3000, type=int, help='num_epochs')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')

args = parser.parse_args()


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = os.path.join(args.model_save_dir, timestamp)
# utils.mkdir(model_dir)
os.makedirs(model_dir, exist_ok=True)
val_result = os.path.join(model_dir, 'picture')
os.makedirs(val_result, exist_ok=True)


train_dir = args.train_dir
val_dir = args.val_dir

######### Model ###########
model_restoration = DeepUnfoldingMethod()
ckp_path = "/home/data/sds/projects/High-Engneer-Math/checkpoints/20231230_162155/model_best.pth"
checkpoint = torch.load(ckp_path)
model_restoration.load_state_dict(checkpoint['state_dict'])

# print number of model
params = sum(param.nelement() for param in model_restoration.parameters())
print(params / 1e6, "M")
model_restoration.cuda()

######### DataLoaders ###########
val_dataset = BlurryImageDataset(datadir=args.val_dir, image_size=(224, 224), val=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False, pin_memory=True)

best_psnr = 0
best_epoch = 0
iter = 0

#### Evaluation ####
model_restoration.eval()
psnr_val_rgb = []
ssim_val_rgb = []
with torch.no_grad():
    for ii, data_val in enumerate((val_loader), 0):
        target_ = data_val["target"].cuda()
        input_  = data_val["input"].cuda()
        path_ = data_val["path"]

        restored = model_restoration(input_)
        
        ssim, psnr = calculate_ssim_psnr(restored[0], target_[0])
        psnr_val_rgb.append(psnr)
        ssim_val_rgb.append(ssim)
        
        def save_image(image, path):
            image = image[0].cpu().detach().numpy().transpose((1, 2, 0))
            image = image * 255
            image = image.astype(np.uint8)
            Image.fromarray(image).save(path)

        filename_with_extension = os.path.basename(path_[0])
        filename_without_extension = os.path.splitext(filename_with_extension)[0]
        itr_val_result = os.path.join(val_result, str(iter))
        os.makedirs(itr_val_result, exist_ok=True)
        save_image(input_, os.path.join(itr_val_result, '{}_input.png'.format(filename_without_extension)))
        save_image(target_, os.path.join(itr_val_result, '{}_target.png'.format(filename_without_extension)))
        save_image(restored, os.path.join(itr_val_result, '{}_restored.png'.format(filename_without_extension)))

    psnr_val_rgb = sum(psnr_val_rgb) / len(psnr_val_rgb)
    ssim_val_rgb = sum(ssim_val_rgb) / len(ssim_val_rgb)

    print("PSNR:", psnr_val_rgb, "\t SSIM:", ssim_val_rgb)
