import torch
from torch import nn


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS

        self.conv_layers_0 = nn.Sequential(
            nn.Conv2d(
                image_channels, 
                32,
                3,
                stride=1,
                padding=1
                ),
            nn.MaxPool2d(2, stride = 2),
            nn.ReLU(),
            nn.Conv2d(
                64,
                64,
                3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                64,
                64,
                3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                64,
                output_channels[0],
                3,
                stride=2,
                padding=1
            )
        )
        self.conv_layers_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                output_channels[0],
                128,
                3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                128,
                output_channels[1],
                3,
                stride=2,
                padding=1
            )
        )
        self.conv_layers_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                output_channels[1],
                256,
                3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                256,
                output_channels[2],
                3,
                stride=2,
                padding=1
            )
        )
        self.conv_layers_3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                output_channels[2],
                128,
                3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                128,
                output_channels[3],
                3,
                stride=2,
                padding=1
            )
        )
        self.conv_layers_4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                output_channels[3],
                128,
                3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                128,
                output_channels[4],
                3,
                stride=2,
                padding=1
            )
        )
        self.conv_layers_5 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                output_channels[4],
                128,
                3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                128,
                output_channels[5],
                3,
                stride=1,
                padding=0
            )
        )
    
    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        layers_vec = [self.conv_layers_0, self.conv_layers_1, self.conv_layers_2,
        self.conv_layers_3, self.conv_layers_4, self.conv_layers_5]
        for i, layers in enumerate(layers_vec):
            if(i == 0):
                out_features.append(layers(x))
            else:
                out_features.append(layers(out_features[-1]))
        for idx, feature in enumerate(out_features):
            expected_shape = (
                self.output_channels[idx],
                self.output_feature_size[idx],
                self.output_feature_size[idx])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

