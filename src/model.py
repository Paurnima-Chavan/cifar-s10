import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=padding,
                      stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, stride=stride,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def __call__(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(BasicBlock, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=padding,
                      stride=stride, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def __call__(self, x):
        x = self.conv_block1(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Prep Layer
        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Layer 1
        self.basic_block1 = BasicBlock(in_channels=64, out_channels=128)
        self.R1 = ResBlock(in_channels=128, out_channels=128)

        # Layer 2
        self.basic_block2 = BasicBlock(in_channels=128, out_channels=256, stride=1, padding=2)

        # # Layer 3
        self.basic_block3 = BasicBlock(in_channels=256, out_channels=512)
        self.R2 = ResBlock(in_channels=512, out_channels=512)

        # MaxPooling
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=4)
        )

        # FC layer
        self.fc_layer = nn.Sequential(nn.Linear(512, 10))

    def forward(self, x):
        x = self.prep_layer(x)  # 1. Prep Layer

        x1 = self.basic_block1(x)  # 2. Layer 1
        r1 = self.R1(x1)
        x = x1 + r1

        x = self.basic_block2(x)  # 3. Layer 2

        x2 = self.basic_block3(x)  # 4. Layer 3
        r2 = self.R2(x2)
        x = x2 + r2

        x = self.pool(x)  # 5. MaxPooling

        x = x.view(-1, 512)

        x = self.fc_layer(x)  # 6. FC Layer

        return F.log_softmax(x, dim=-1)


def model_summary(model, input_size):
    """
    This function displays a summary of the model, providing information about its architecture,
    layer configuration, and the number of parameters it contains.
    :param model: model
    :param input_size: input_size for model
    :return:
    """
    summary(model, input_size=input_size)