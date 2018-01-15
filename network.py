import torch
import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.conv1 = nn.ConvTranspose2d(100, 512, kernel_size=4)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(512)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)


    def forward(self, x, input_size):
        self.input_size = input_size

        #x = x.view(-1, self.input_size)
        x = x.view(self.input_size, -1, 1,1)
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)), 0.2)
        x = self.upsampling(x)

        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        #x = F.dropout(x, training=self.training)
        x = self.upsampling(x)

        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = self.upsampling(x)

        x = F.leaky_relu(self.conv4_bn(self.conv4(x)),0.2)
        #x = F.dropout(x, training=self.training)
        x = self.upsampling(x)

        x = F.leaky_relu(self.conv5_bn(self.conv5(x)),0.2)
        #x = F.dropout(x, training=self.training)

        return F.tanh(self.conv6(x))



class Discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
	self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=2)#, padding=1)
	self.conv2_bn = nn.BatchNorm2d(128)
	self.conv3_bn = nn.BatchNorm2d(256)
	self.conv4_bn = nn.BatchNorm2d(512)

    def forward(self, x):
        #x = x.view(-1, self.input_size)
        x = F.leaky_relu(self.conv1(x),0.2)

        x = F.leaky_relu(self.conv2_bn(self.conv2(x)),0.2)

        x = F.leaky_relu(self.conv3_bn(self.conv3(x)),0.2)

        x = F.leaky_relu(self.conv4_bn(self.conv4(x)),0.2)

        return F.sigmoid(self.conv5(x))











