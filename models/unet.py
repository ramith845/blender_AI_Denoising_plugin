"""Small U-Net denoiser (Phase 2, Task 1).

Input : (N, 9, H, W)  log1p(noisy) RGB + albedo + normal   (see dataset.to_input)
Output: (N, 3, H, W)  clean image in log space              (see dataset.from_output)
H and W must be multiples of 16, because the image is halved 4 times.

Shapes for a 128x128 crop (channels, size):
    encoder   9,128 -> 32,128 | 48,64 | 64,32 | 96,16     (each level saved as a skip)
    bottleneck                96,8 -> 128,8
    decoder   128,8 -> 96,16 -> 64,32 -> 48,64 -> 32,128  (upsample, concat skip, convs)
    head      32,128 -> 3,128
About 1.0M parameters. Only ONNX-friendly ops (conv, relu, maxpool, upsample, concat).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(cin, cout):
    """The building block used at every level: two 3x3 convs with a ReLU after each.
    The first conv changes the channel count cin -> cout, the second keeps it.
    padding=1 keeps height and width unchanged; only pooling/upsampling change the size."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, (3, 3), padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, (3, 3), padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    # These argument names are saved in the checkpoint's config["model"] by train.py,
    # and evaluate.py rebuilds the model with UNet(**config["model"]). Keep them in sync.
    def __init__(
        self, in_channels=9, out_channels=3, widths=(32, 48, 64, 96), bottleneck=128
    ):
        super().__init__()  # must come first, nn.Module needs it before any layers are assigned
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.widths = widths  # channels per encoder level; the decoder mirrors them
        self.bottleneck_width = bottleneck

        # Encoder: one double_conv per level, 9 -> 32 -> 48 -> 64 -> 96.
        # Must be an nn.ModuleList (not a plain list), or PyTorch can't see the weights
        # and the optimizer would have nothing to train.
        chans = [in_channels, *widths]
        self.encoder = nn.ModuleList()
        for cin, cout in zip(chans[:-1], chans[1:]):
            self.encoder.append(double_conv(cin, cout))

        # No weights, so the same layer is reused after every encoder level (halves H and W).
        self.max_pool = nn.MaxPool2d(2)

        # Decoder: levels run deepest first, 128 -> 96 -> 64 -> 48 -> 32.
        # Each level's input is the upsampled channels (cin) PLUS the skip from the
        # matching encoder level (cout), so cin += cout, e.g. 128 + 96 = 224 -> 96.
        chans = [bottleneck, *widths[::-1]]
        self.decoder = nn.ModuleList()
        for cin, cout in zip(chans[:-1], chans[1:]):
            cin += cout
            self.decoder.append(
                double_conv(cin, cout),
            )

        # Doubles H and W (nearest neighbour). No weights, reused by every decoder level.
        self.upsample = nn.Upsample(scale_factor=2)

        # Smallest level, after the last pool: 96 -> 128 channels.
        self.bottleneck = double_conv(self.widths[-1], self.bottleneck_width)

        # 1x1 conv: mixes the 32 channels down to RGB for each pixel. No activation after
        # it, the network predicts log-space values directly (dataset.from_output clamps).
        self.head = nn.Conv2d(self.widths[0], self.out_channels, 1)

    def forward(self, x):
        # Encoder: convs, remember the result for the decoder, then halve the size.
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)  # saved BEFORE pooling, at this level's full size
            x = self.max_pool(x)

        x = self.bottleneck(x)

        # Decoder: double the size, glue on the encoder features of that size, convolve.
        # skips.pop() takes the last saved (deepest) first, which pairs with the first level.
        for block in self.decoder:
            x = self.upsample(x)
            x = torch.cat([x, skips.pop()], 1)  # dim 1 = channels; N, H, W must already match
            x = block(x)

        x = self.head(x)

        return x
