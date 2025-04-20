---
title: My Attempt to Implement the BrainMRNet Paper
date: 2025-04-19  
tags: ['Deep learning', 'Machine learning', 'Paper', 'Implement', 'Brain tumor']  
---

Today I tried to implement the paper titled "BrainMRNet: Brain tumor detection using magnetic resonance images with a novel convolutional neural network model" (TOĞAÇAR et al., 2019).

The full code can be found in my [Github repo](https://github.com/Marshal-Ashif-Shawkat/bme-310-project).

In this blog post, I will try to share my experience and code chunks for the BrainMRNet model described in the above paper.

The full architecture looks like this and contains different blocks like Conv_Block, Dense_Block, Residual_Block, CBAM_Block. CBAM_Block contains two blocks namely Channel Attention Module and Spatial Attention Module. Below we described each block along with PyTorch code.

![Full model architecture](full-model.png)

## Conv_Block & Dense_Block
The easiest blocks of the architecture.

![Conv block and Dense block](conv-block.png)

```python
from torch import nn
import torch.nn.functional as F

class Conv_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3)
        self.bn = nn.BatchNorm2d(num_features=out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv2d(x)))

class Dense_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.dense = nn.Linear(in_features=in_c, out_features=out_c, bias=True)
        self.bn = nn.BatchNorm1d(num_features=out_c)

    def forward(self, x):
        return F.relu(self.bn(self.dense(x)))
```

There is no 'padding=same' option in PyTorch. So, I have to make sure the (H, W) dimensions remain same after the input feature passes through this blocks.

## Channel Attention Module

![Channel Attention Module](cam.png)

Implementing Channel Attention Module was easier than I thought. I had confusion about the shared MLP layer. I was confused whether the MLP layer share the same weight (hence they are called shared MLP) or not. Then I look into the source paper of CBAM titled "CBAM: Convolutional Block Attention Module" (Woo et al., 2018). The figure of the channel attention module from this paper cleared my confusion. They are called shared MLP because both the max_pooled feature and avg_pooled feature goes through the same MLP layer.

![CAM figure from the source paper](cam2.png)

```python
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_c, ratio):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_c, in_c//ratio, bias=True),
            nn.ReLU(),
            nn.Linear(in_c//ratio, in_c, bias=True)
        )

    def forward(self, x):
        y1 = F.max_pool2d(x, kernel_size=x.shape[2:])
        y2 = F.avg_pool2d(x, kernel_size=x.shape[2:])

        y1 = y1.view(y1.shape[0], -1)
        y1 = self.shared_mlp(y1)
        y2 = y2.view(y2.shape[0], -1)
        y2 = self.shared_mlp(y2)

        out = y1 + y2
        out = F.sigmoid(out)
        return out
```

There is no global max pool or global avg pool layer in PyTorch. To compensate this, I had to use proper kernel_size.

## Spatial Attention Module

![SAM module from the source paper](sam2.png)

```python
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3)

    def forward(self, x):
        y_max = x.max(dim=1, keepdim=True).values
        y_avg = x.mean(dim=1, keepdim=True)
        y = torch.cat((y_max, y_avg), dim=1)
        y = F.sigmoid(self.conv2d(y))
        return y
```
During implementing the Spatial Attention Module, I found a weird bug/characteristic of the PyTorch library. If I only use *y_max = x.max(dim=1, keepdim=True)* (without *.values*), then y_max is a PyTorch function, not a torch tensor. But for x.mean(), I didn't have to use *.values* to make *y_avg* a torch tensor. Strange!! Also, I carefully chose the kernel_size and the padding to make it comaptible in the next CBAM block.

## CBAM_Block

![CBAM block](cbam.png)

```python
class CBAM_Block(nn.Module):
    def __init__(self, in_c, ratio=8):
        super().__init__()
        self.cam = ChannelAttentionModule(in_c, ratio=8)
        self.sam = SpatialAttentionModule()

    def forward(self, x):
        y = self.cam(x)
        y_cam = x * y.unsqueeze(2).unsqueeze(3)
        y = self.sam(y_cam)
        out = y_cam * y 
        return out
```

## Residual_Block

Finally, there is the residual block.

![Residual block](residual.png)

```python
class Residual_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv2d1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1)
        self.conv2d2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        y = self.bn2(self.conv2d2(F.leaky_relu(self.bn1(self.conv2d1(x)))))
        out = y + x
        return F.leaky_relu(out)
```

## BrainMRNet Model
Combining all the blocks according to the first picture of this post, I get the BrainMRNet model.

```python
class BrainMRNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Stage 1
        self.convblock11 = Conv_Block(in_c=3, out_c=32)
        self.convblock12 = Conv_Block(in_c=32, out_c=32)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 2
        self.convblock21 = Conv_Block(in_c=32, out_c=64)
        self.cbamblock21 = CBAM_Block(in_c=64, ratio=8)
        self.residualblock21 = Residual_Block(in_c=64, out_c=64)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 3
        self.convblock31 = Conv_Block(in_c=64, out_c=128)
        self.cbamblock31 = CBAM_Block(in_c=128, ratio=8)
        self.residualblock31 = Residual_Block(in_c=128, out_c=128)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.denseblock1 = Dense_Block(in_c=16, out_c=256)
        self.dropout = nn.Dropout(p=0.3)
        self.denseblock2 = Dense_Block(in_c=256, out_c=256)
        self.classifier = nn.Linear(in_features=256, out_features=num_classes)
        

    def forward(self, x):
        y1 = self.maxpool1(self.convblock12(self.convblock11(x)))
        y2 = self.maxpool2(self.residualblock21(self.cbamblock21(self.convblock21(y1))))
        y3 = self.maxpool3(self.residualblock31(self.cbamblock31(self.convblock31(y2))))

        y1 = F.interpolate(y1, size=(4,4), mode='bilinear')
        y2 = F.interpolate(y2, size=(4,4), mode='bilinear')
        y3 = F.interpolate(y3, size=(4,4), mode='bilinear')
        
        y = torch.cat((y1, y2, y3), dim=1)
        y = y.mean(dim=1, keepdim=True)
        y = y.view(y.shape[0], -1)
        y = self.classifier(self.denseblock2(self.dropout(self.denseblock1(y))))
        return y
```

For the upsampling layers, I have used *torch.nn.functional.interpolate* function. I have used size (4,4) in all three stages to make it easier for concatenate which is a deviation from the architecure described in the paper. 

## My Final Thoughts

By implementing this paper, I gained confidence that I can implement moderately complex architecture. It also thought me about the usefulness of writing modular code.
