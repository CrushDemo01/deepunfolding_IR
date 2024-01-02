from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F


class Unet(nn.Module):
    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            chans: int = 32,
            num_pool_layers: int = 4,
            normtype: str = 'in',
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.normtype = normtype

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, normtype)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, normtype))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, normtype)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, normtype))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, normtype),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image  # torch.Size([1, 30, 320, 320])
        # print(image.shape)
        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)  # output torch.Size([1, 30, 320, 320])
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, normtype='bn'):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.normtype = normtype

        if self.normtype == 'bn':
            self.layers = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)  # torch.Size([1, 2, 320, 320])


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class DeepUnfoldingMethod(nn.Module):
    def __init__(self, iters=5, in_channels=3, out_channels=3, first_channels=32, down_nums=2, stride=0.0001):
        super(DeepUnfoldingMethod, self).__init__()
        self.iters = iters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_channels = first_channels
        self.down_nums = down_nums
        self.stride = nn.Parameter(0.1 * torch.ones(self.iters, 1), requires_grad=True)
        
        self.H = nn.Sequential(*[nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                                nn.Conv2d(in_channels, in_channels, 3, 1, 1)])
        
        self.HT = nn.Sequential(*[nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1),
                                 nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1),
                                 nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1)])
        
        self.denoiser = Unet(in_channels, out_channels, chans=first_channels, num_pool_layers=down_nums)
        
        torch.nn.init.normal_(self.stride, mean=0.1, std=0.01)
        
    def forward(self, y):
        x0 = deepcopy(y)
        for _ in range(self.iters):
            t = self.stride[_]
            z_0 = x0 + t * self.H(y) - t * self.HT(self.H(x0))
            x_1 = self.denoiser(z_0)
            x_0 = x_1
            
        return x_1
                
                
if __name__ == '__main__':
    x = torch.randn((1, 3, 224, 224))
    model = DeepUnfoldingMethod()
    out = model(x)
    params = sum(param.nelement() for param in model.parameters())
    print(params / 1e6, "M")
    print(out.shape)