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

start_epoch = 1

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

num_epochs = args.num_epochs
batch_size = args.batch_size

start_lr = 2e-4
end_lr = 1e-6

######### Model ###########
model_restoration = DeepUnfoldingMethod()

# print number of model
params = sum(param.nelement() for param in model_restoration.parameters())
print(params / 1e6, "M")
model_restoration.cuda()
optimizer = optim.Adam(model_restoration.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-8)

######### Scheduler ###########
warmup_epochs = 3
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=end_lr)

######### Loss ###########
criterion = nn.L1Loss()
######### DataLoaders ###########
train_dataset = BlurryImageDataset(datadir=args.train_dir, image_size=(224, 224))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)

# train_dataset = BlurryImageDataset(datadir=args.val_dir, image_size=(224, 224))
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)

val_dataset = BlurryImageDataset(datadir=args.val_dir, image_size=(224, 224))
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False, pin_memory=True)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, num_epochs + 1))
print('===> Loading datasets')

best_psnr = 0
best_epoch = 0
writer = SummaryWriter(model_dir)
iter = 0

for epoch in range(start_epoch, num_epochs + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()
    for i, data in enumerate(tqdm(train_loader), 0):

        target_ = data["target"].cuda()
        input_  = data["input"].cuda()
        path_ = data["path"]
        # print(data["input"].shape)
        # print(data["target"].shape)
        # print(data["path"])
        
        restored = model_restoration(input_)
        
        loss = criterion(restored, target_)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        iter += 1
        writer.add_scalar('loss/iter_loss', loss, iter)
    writer.add_scalar('loss/epoch_loss', epoch_loss, epoch)
    #### Evaluation ####
    model_restoration.eval()
    psnr_val_rgb = []
    ssim_val_rgb = []
    with torch.no_grad():
        for ii, data_val in enumerate((val_loader), 0):
            target_ = data["target"].cuda()
            input_  = data["input"].cuda()
            path_ = data["path"]

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
        writer.add_scalar('val/psnr', psnr_val_rgb, epoch)
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,"model_best.pth"))

        print("[epoch %d PSNR: %.4f SSIM: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, ssim_val_rgb, best_epoch, best_psnr))

        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,f"model_epoch_{epoch}.pth")) 

    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth")) 

writer.close()
