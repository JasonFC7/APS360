import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import floor

class prim_model(nn.Module):
    def __init__(self, num_classes = 3, input_size = [176,208], kernel_sizes = [5, 5, 5, 5], strides = [1, 1, 1, 1], paddings = [1, 1, 1, 1]):
        super(prim_model, self).__init__()
        self.name = "prim_model"
        self.conv1 = nn.Conv2d(3, 5, kernel_sizes[0], strides[0], paddings[0])
        self.conv2 = nn.Conv2d(5, 10, kernel_sizes[1], strides[1], paddings[1])
        self.conv3 = nn.Conv2d(10, 20, kernel_sizes[2], strides[2], paddings[2])
        #self.conv4 = nn.Conv2d(20, 40, kernel_sizes[3], strides[3], paddings[3])
        self.pool = nn.MaxPool2d(2, 2)

        conv_pool_output_size = lambda size, kernel_size, stride, padding: floor((size - kernel_size + 2 * padding) / stride + 1) // 2

        x = conv_pool_output_size(input_size[0], kernel_sizes[0], strides[0], paddings[0])
        x = conv_pool_output_size(x, kernel_sizes[1], strides[1], paddings[1])
        x = conv_pool_output_size(x, kernel_sizes[2], strides[2], paddings[2])
        #x = conv_pool_output_size(x, kernel_sizes[3], strides[3], paddings[3])
        
        y = conv_pool_output_size(input_size[1], kernel_sizes[0], strides[0], paddings[0])
        y = conv_pool_output_size(y, kernel_sizes[1], strides[1], paddings[1])
        y = conv_pool_output_size(y, kernel_sizes[2], strides[2], paddings[2])
        #y = conv_pool_output_size(y, kernel_sizes[3], strides[3], paddings[3])
        
        self.fc_input = 20 * x * y

        self.fc1 = nn.Linear(self.fc_input, 20)
        self.fc2 = nn.Linear(20, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, self.fc_input)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1)
        return x