from copy import deepcopy
from PIL import Image, ImageFilter
import os
import random
import numpy as np
import torchvision.transforms.functional as TF
import math
from scipy.signal import convolve2d
import cv2
import torchvision.transforms as transforms
import sys
sys.path.append("/home/data/sds/projects/High-Engneer-Math")
from utils.metrics import calculate_ssim_psnr


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, fns=None):
    images = []
    if fns is None:
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):                
                    path = os.path.join(root, fname)
                    images.append(path)
    else:
        for fname in fns:
            if is_image_file(fname):
                path = os.path.join(dir, fname)
                images.append(path)
    return images


# 定义要应用的变换
transform = transforms.Compose([    
    transforms.RandomRotation(degrees=20),  # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色调整
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
])

to_Tensor = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
])

def get_params(img, output_size=(224, 224)):
    w, h = img.size
    th, tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w

    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw



class BlurSyn(object):
    def __init__(self):
        # Kernel Size of the Gaussian Blurry
        self.kernel_sizes = [5, 7, 9, 11]
        self.kernel_probs = [0.1, 0.2, 0.3, 0.4]

        # Sigma of the Gaussian Blurry
        self.sigma_range = [2, 5]
        self.alpha_range = [0.6, 0.9]

    def __call__(self, R_):
        kernel_size = np.random.choice(self.kernel_sizes, p=self.kernel_probs)
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel2d = np.dot(kernel, kernel.T)
        for i in range(3):
            R_[..., i] = convolve2d(R_[..., i], kernel2d, mode='same')

        return R_
    
    
class BlurryImageDataset:
    def __init__(self, datadir, fns=None, image_size=(224, 224)):
        self.datadir = datadir
        self.image_size = image_size
        
        sortkey = lambda key: os.path.split(key)[-1]
        self.paths = sorted(make_dataset(datadir, fns), key=sortkey)
        # self.num_images = num_images
        self.apply_blur = BlurSyn()
        
    def __getitem__(self, index):
        # index = index % len(self.paths)
        path = self.paths[index]
        image = Image.open(path).convert('RGB')
        i, j, h, w = get_params(image, self.image_size)
        image = TF.crop(image, i, j, h, w)
        
        image = transform(image)
        
        # Randomly select blur effect
        
        image = np.asarray(image, np.float32) / 255.
        
        blurred_image = self.apply_blur(deepcopy(image))
        
        image = TF.to_tensor(image)
        blurred_image = TF.to_tensor(blurred_image)
        
        return {"input": blurred_image, "target": image, "path": path}

    def __len__(self):
        return len(self.paths)

if __name__ == "__main__":
    dataset = BlurryImageDataset(datadir="/home/data/sds/projects/RR_zero/checkpoints/restormer_l_all/results/20231219-165418/001/001", image_size=(128,128))
    
    from torch.utils.data import DataLoader
    
    val_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
    
    for i, data in enumerate(val_loader):
        print(i)
        print(data["input"].shape)
        print(data["target"].shape)
        print(data["path"])
        ssim, psnr = calculate_ssim_psnr(data["input"][0], data["target"][0])
        print(ssim, psnr)
        
        break



