from common import *


class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()
        self.p1 = Conv(3, 64, 6, 2, 2)
        self.p2 = Conv(64, 128, 3, 2)
        self.c3_1 = C3(128, 128, 64, 3, True)
        self.p3 = Conv(128, 256, 3, 2)
        self.c3_2 = C3(256, 256, 128, 6, True)
        self.p4 = Conv(256, 512, 3, 2)
        self.c3_3 = C3(512, 512, 256, 9, True)
        self.p5 = Conv(512, 1024, 3, 2)
        self.c3_4 = C3(1024, 1024, 512, 3, True)

        self.sppf = SPPF(1024, 1024)

    def forward(self, x):
        y1 = self.p1(x)
        y2 = self.p2(y1)
        y3 = self.c3_1(y2)
        y4 = self.p3(y3)
        y5 = self.c3_2(y4)
        y6 = self.p4(y5)
        y7 = self.c3_3(y6)
        y8 = self.p5(y7)
        y9 = self.c3_4(y8)
        y10 = self.sppf(y9)

        return y5, y7, y10


class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()

        self.p1 = Conv(1024, 512, 1, 1)
        self.c3_1 = C3(1024, 512, 256, 3, False)
        self.p2 = Conv(512, 256, 1, 1)
        self.c3_2 = C3(512, 256, 128, 3, False)
        self.p3 = Conv(256, 256, 3, 2)
        self.c3_3 = C3(512, 512, 256, 3, False)
        self.p4 = Conv(512, 512, 3, 2)
        self.c3_4 = C3(1024, 1024, 512, 3, False)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x1, x2, x3):
        # FPN UP
        y1 = self.p1(x3)
        y2 = torch.cat([self.upsample1(y1), x2], 1)
        y3 = self.c3_1(y2)
        y4 = self.p2(y3)
        y5 = torch.cat([self.upsample2(y4), x1], 1)
        y6 = self.c3_2(y5)

        # FPN DOWN
        y7 = torch.cat([self.p3(y6), y4], 1)
        y8 = self.c3_3(y7)
        y9 = torch.cat([self.p4(y8), y1], 1)
        y10 = self.c3_4(y9)

        return y6, y8, y10


class Head(nn.Module):
    def __init__(self, anchors, strides, num_cls, training=False):
        super(Head, self).__init__()

        channels = 3 * (5 + num_cls)

        self.layer1 = nn.Conv2d(256, channels, 1, 1, 0)
        self.layer2 = nn.Conv2d(512, channels, 1, 1, 0)
        self.layer3 = nn.Conv2d(1024, channels, 1, 1, 0)

        self.anchor = Anchor(anchors, strides, num_cls, training)

    def forward(self, x1, x2, x3):
        y1 = [self.layer1(x1), self.layer2(x2), self.layer3(x3)]
        y2 = self.anchor(y1)

        return y2


class YOLOP(nn.Module):
    def __init__(self, anchors, strides, num_cls, device, training=False):
        super().__init__()

        anchors = torch.tensor(anchors, device=device)
        strides = torch.tensor(strides, device=device)

        self.backbone = BackBone().to(device)
        self.neck = Neck().to(device)
        self.head = Head(anchors, strides, num_cls, training).to(device)

    def forward(self, x):
        y1, y2, y3 = self.backbone(x)
        y4, y5, y6 = self.neck(y1, y2, y3)
        y7 = self.head(y4, y5, y6)

        return y7