# -*- coding:utf-8 -*-
"""
 @FileName   : samsiam.py
 @Time       : 9/25/24 11:05 AM
 @Author     : Woldier Wong
 @Description: refer https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py
"""
import torch, torch.nn as nn
from torch.nn import Module


class SimSiam(Module):
    """
    Build a SimSiam model.


    >>> criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)
    >>> backbone = xxx
    >>> model = SimSiam(backbone, dim=2048, pred_dim=512)
    >>>
    >>> image = torch.randn((2, 3, 512, 512))
    >>> normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    >>>                                  std=[0.229, 0.224, 0.225])
    >>>
    >>> # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    >>> augmentation = [
    >>>     transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    >>>     transforms.RandomApply([
    >>>         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    >>>     ], p=0.8),
    >>>     transforms.RandomGrayscale(p=0.2),
    >>>     transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
    >>>     transforms.RandomHorizontalFlip(),
    >>>     transforms.ToTensor(),
    >>>     normalize
    >>> ]
    >>> images_1, images_2 = do_aug(img, augmentation)  # to q k
    >>>
    >>> p1, p2, z1, z2 = model(x1=images_1, x2=images_2)
    >>> loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
    """

    def __init__(self, backbone, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()
        self.backbone = backbone
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(pred_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(pred_dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),  # second layer
        )
        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=False),  # hidden layer
            nn.Linear(dim, dim)  # output layer
        )

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.fc(self.backbone(x1)[-1]) # NxC
        z2 = self.fc(self.backbone(x2)[-1])  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach()
