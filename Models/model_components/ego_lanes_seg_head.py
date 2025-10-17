import torch
import torch.nn as nn


class EgoLanesSegHead(nn.Module):
    def __init__(self):
        super(EgoLanesSegHead, self).__init__()
        # Standard
        self.GeLU = nn.GELU()

        # Segmentation Head - Output Layers
        self.upsample_layer_3 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.skip_link_layer_3 = nn.Conv2d(32, 256, 1)
        self.decode_layer_6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.decode_layer_7 = nn.Conv2d(256, 128, 3, 1, 1)

        self.upsample_layer_4 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.decode_layer_8 = nn.Conv2d(128, 128, 3, 1, 1)
        self.decode_layer_9 = nn.Conv2d(128, 64, 3, 1, 1)
        self.decode_layer_10 = nn.Conv2d(64, 64, 3, 1, 1)

        # Final output layer - 3 channels for ego left, ego right, and other lanes
        self.final_layer = nn.Conv2d(16, 3, 1, 1)