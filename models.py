import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self, backbone, global_pool=None, poolkernel=7):
        super(BaseNet, self).__init__()
        if global_pool is None:
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        else:
            backbone = nn.Sequential(*list(backbone.children()))
        self.net = backbone
        layers = list(backbone.children())
        if global_pool is not None:

            if global_pool == "max":
                layers.append(nn.MaxPool2d(poolkernel))
            elif global_pool == "avg":
                layers.append(nn.AvgPool2d(poolkernel))
        self.net = nn.Sequential(*layers)

    def forward(self, x0):
        return self.net.forward(x0)