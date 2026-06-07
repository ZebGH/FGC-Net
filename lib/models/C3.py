import torch
import torch.nn as nn

class Conv(nn.Module):
    # Standard convolution block with Batch normalization and SiLU activation
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # Adjust padding automatically for kernel size k if p is None
        padding = (k // 2) * d # Auto padding for dilation > 1 might need adjustment
        if p is not None:
             padding = p
        self.conv = nn.Conv2d(c1, c2, k, s, padding=padding, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # Use SiLU activation if act=True, otherwise identity
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        # Used for model fusing (inference optimization)
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck block with residual connection
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels (using expansion factor e)
        self.cv1 = Conv(c1, c_, 1, 1) # 1x1 conv reducing channels
        self.cv2 = Conv(c_, c2, 3, 1, g=g) # 3x3 conv
        # Residual connection: add input to output if shortcut=True and c1==c2
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # Apply convolutions and add shortcut if applicable
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions (Cross Stage Partial)
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # Main convolution path (input -> 1x1 -> n bottlenecks -> output)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1) # Direct/shortcut path convolution
        self.cv3 = Conv(2 * c_, c2, 1)  # Final 1x1 conv after concatenation
        # Stack 'n' bottleneck blocks sequentially
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))) # Note: Bottleneck expansion e=1.0 here

    def forward(self, x):
        # Split input into two paths (implicitly via cv1 and cv2)
        # Path 1: Through bottlenecks
        path1 = self.m(self.cv1(x))
        # Path 2: Direct path
        path2 = self.cv2(x)
        # Concatenate the two paths along the channel dimension
        # Apply final convolution
        return self.cv3(torch.cat((path1, path2), dim=1))
