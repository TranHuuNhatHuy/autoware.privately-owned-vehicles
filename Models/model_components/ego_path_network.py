from .pre_trained_backbone import PreTrainedBackbone
from .path_context import PathContext
from .ego_path_head import EgoPathHead

import torch.nn as nn

class EgoPathNetwork(nn.Module):
    def __init__(self, pretrained):
        super(EgoPathNetwork, self).__init__()

        # Upstream blocks
        self.PreTrainedBackbone = PreTrainedBackbone(pretrained)

        # Path Context
        self.PathContext = PathContext()

        # EgoPath Head
        self.EgoPathHead = EgoPathHead()
    

    def forward(self, image):
        features = self.PreTrainedBackbone(image)
        deep_features = features[4]
        context = self.PathContext(deep_features)
        ego_path, left_lane, right_lane = self.EgoPathHead(context)
        return ego_path, left_lane, right_lane