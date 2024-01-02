import torch
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_ssim_psnr(image1, image2):
    # Assuming image1 and image2 are torch.Tensor representing images

    # Convert torch.Tensor to NumPy array
    image1_np = image1.cpu().detach().numpy().transpose((1, 2, 0))
    image2_np = image2.cpu().detach().numpy().transpose((1, 2, 0))

    # Calculate SSIM
    ssim_value, _ = ssim(image1_np, image2_np, full=True, multichannel=True)

    # Calculate PSNR
    psnr_value = psnr(image1_np, image2_np, data_range=1.0)

    return ssim_value, psnr_value

# 示例用法：
# image1 = ...  # Your first image as torch.Tensor
# image2 = ...  # Your second image as torch.Tensor
# ssim_value, psnr_value = calculate_ssim_psnr(image1, image2)
# print(f"SSIM: {ssim_value}, PSNR: {psnr_value}")
