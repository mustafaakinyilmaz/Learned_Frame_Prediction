
# coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import math


def conv_same(in_ch,out_ch,kernel_size):
    conv = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=kernel_size,padding=kernel_size//2,bias=True)
    return conv


    
    
class RDB(nn.Module):
    def __init__(self, channel, kernel_size):
        super(RDB, self).__init__()
        
        self.cnn1 = conv_same(channel,channel,kernel_size)
        self.cnn2 = conv_same(channel*2,channel,kernel_size)
        self.cnn3 = conv_same(channel*3,channel,kernel_size)
        self.down_conv = conv_same(channel*4,channel,1)
        
    def forward(self,x):
        c1 = F.relu(self.cnn1(x))
        c2 = F.relu(self.cnn2(torch.cat((x,c1),dim=1)))
        c3 = F.relu(self.cnn3(torch.cat((x,c1,c2),dim=1)))
        concat = torch.cat((x,c1,c2,c3),dim=1)
        out = self.down_conv(concat) + x

        return out



class Model(nn.Module):
    def __init__(self,num_frame, channel, conv_kernel, num_of_rdb):
        super(Model, self).__init__()
        
        self.conv_init = conv_same(num_frame,channel,conv_kernel)
        
        rbs = [RDB(channel, conv_kernel) for _ in range(num_of_rdb)]
        self.res_blocks = nn.Sequential(*rbs)   
            
        self.conv_final = conv_same(channel,1,conv_kernel)
        
    def forward(self,x):
        c_i = self.conv_init(x)
        r = self.res_blocks(c_i)
        out = self.conv_final(r)
        return out