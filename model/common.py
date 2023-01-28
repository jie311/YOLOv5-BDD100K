import torch
from torch import nn


class Conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel=1, stride=1, pad=None):
        super().__init__()

        pad = kernel // 2 if pad is None else pad

        self.conv = nn.Conv2d(ch_in, ch_out, kernel, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class BottleNeck(nn.Module):
    def __init__(self, ch_in, ch_out, ch_, shortcut=True):
        super().__init__()

        self.conv1 = Conv(ch_in, ch_, 1, 1)
        self.conv2 = Conv(ch_, ch_out, 3, 1)

        self.shortcut = shortcut and ch_in == ch_out

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.shortcut else self.conv2(self.conv1(x))


class C3(nn.Module):
    def __init__(self, ch_in, ch_out, ch_, num=1, shortcut=True):
        super().__init__()

        self.conv1 = Conv(ch_in, ch_, 1, 1)
        self.conv2 = Conv(ch_in, ch_, 1, 1)
        self.conv3 = Conv(2 * ch_, ch_out, 1, 1)

        self.m = nn.Sequential(*(BottleNeck(ch_, ch_, ch_, shortcut) for _ in range(num)))

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), 1))


class SPP(nn.Module):
    def __init__(self):
        super().__init__()

        self.max_pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

    def forward(self, x):
        y1 = self.max_pool1(x)
        y2 = self.max_pool2(x)
        y3 = self.max_pool3(x)

        return torch.cat([x, y1, y2, y3], 1)


class SPPF(nn.Module):
    def __init__(self, ch_in, ch_out, k=5):
        super().__init__()

        ch_ = ch_in // 2
        self.conv1 = Conv(ch_in, ch_, 1, 1)
        self.conv2 = Conv(ch_ * 4, ch_out, 1, 1)
        self.max_pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.max_pool(y1)
        y3 = self.max_pool(y2)
        y4 = self.max_pool(y3)

        return self.conv2(torch.cat((y1, y2, y3, y4), 1))


class Anchor(nn.Module):
    def __init__(self, anchors, strides, num_cls, train):
        super().__init__()

        self.anchors = anchors
        self.strides = strides
        self.num_cls = num_cls
        self.train = train
        self.num_layers = len(anchors)
        self.num_ac = len(self.anchors[0])

        self.grid = [torch.zeros(1)] * self.num_layers
        self.ac_grid = [torch.zeros(1)] * self.num_layers

    def make_grid(self, A, H, W, i):
        device = self.anchors[i].device
        dtype = self.anchors[i].dtype

        shape = 1, A, H, W, 2

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                                        torch.arange(W, device=device, dtype=dtype))

        grid = torch.stack((grid_x, grid_y), 2).expand(shape)
        ac_grid = self.anchors[i].view((1, self.num_ac, 1, 1, 2)).expand(shape)

        return grid, ac_grid

    def forward(self, x):
        for i in range(self.num_layers):
            A = len(self.anchors[i])
            B, _, H, W = x[i].shape

            C = 5 + self.num_cls

            x[i] = x[i].view(B, A, C, H, W).permute(0, 1, 3, 4, 2).contiguous()  # batch、anchor、high、width、channel

            if not self.train:
                x[i] = x[i].sigmoid()
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.ac_grid[i] = self.make_grid(A, H, W, i)

                x[i][..., 0:2] = (x[i][..., 0:2] * 2 - 0.5 + self.grid[i]) * self.strides[i]
                x[i][..., 2:4] = (x[i][..., 2:4] * 2) ** 2 * self.ac_grid[i]

                x[i] = x[i].view(B, -1, C)

        return x if self.train else torch.cat(x, 1)
