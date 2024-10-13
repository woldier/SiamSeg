# -*- coding:utf-8 -*-
"""
 @FileName   : sim_aug_transform.py
 @Time       : 9/25/24 2:04 PM
 @Author     : Woldier Wong
 @Description:
 refer https://github.com/facebookresearch/simsiam/blob/main/simsiam/loader.py
 refer https://github.com/facebookresearch/simsiam/blob/a7bc1772896d0dad0806c51f0bb6f3b16d290468/main_simsiam.py#L229
"""
import random, torch, torch.nn as nn
from mmseg.ops.wrappers import resize
from .dacs_transform import gaussian_blur, denorm_, renorm_
import kornia, torch


def tow_transform(data=None, target=None, mean=None, std=None):
    param = _random_param()
    q, q_target = augment_transform(data=data, target=target, mean=mean, std=std, **param)

    # param = {k: (1 - v) for k, v in param.items()}
    param = _random_param()
    k, k_target = augment_transform(data=data, target=target, mean=mean, std=std, **param)

    return q, q_target, k, k_target


def _random_param():
    param = dict(
        color_jitter_p=random.uniform(0, 1),
        gray=random.uniform(0, 1),
        blur_p=random.uniform(0, 1),
        flip_lr=random.uniform(0, 1),
        flip_ud=random.uniform(0, 1),
    )
    return param


def augment_transform(
        color_jitter_p: float, gray: float, blur_p: float,
        # crop_p: float,
        flip_lr: float, flip_ud: float,
        mean, std,
        data=None, target=None
):
    """
    Parameters:
        color_jitter_p (float): Perform a color_jitter transformation of p, if p is greater than a given threshold,
            color_jitter will be used.
            进行color_jitter变换的p, 如果p大于给定阈值, color_jitter将会被使用.
        gray (float): Perform a grey color transformation of p, if p is greater than a given threshold,
            color_jitter will be used.
        blur_p (float): Perform a blur transformation of p, if p is greater than a given threshold,
            color_jitter will be used.
            进行blur变换的p, 如果p大于给定阈值, color_jitter将会被使用.
        flip_lr (float):
        flip_ud (float):
        mean (Tensor): To perform color_jitter, you need to pass through de_norm and therefore need mean and std.
            进行color_jitter 时需要将经过de_norm因此需要mean 和std
        std (Tensor):
        data (Tensor):
        target (Tensor):


    """

    data = data.clone()  # 发现在进行color_jitter  和 gaussian_blur 会让原始的data 发生变化, 因此这里做了拷贝
    target = target.clone()
    size = (512, 512)
    resize_crop = kornia.augmentation.RandomResizedCrop(size=size, scale=(0.6, 1.0))
    data = resize_crop(data)
    # TODO 对target 做resize_crop
    resize_crop._params
    # for i in range(data.shape[0]):
    #     x_mix, y_mix = resize_crop._params["src"][i][0]
    #     x_max, y_max = resize_crop._params["src"][i][2]
    #     target[i] = resize(target[i::, :, int(y_mix):int(y_max)+1, int(x_mix):int(x_max)+1].float(), size=size).long()


    data, _ = color_jitter(
        color_jitter=color_jitter_p,  # random
        # brightness: The brightness factor to apply.
        # contrast: The contrast factor to apply.
        # saturation: The saturation factor to apply.
        # hue: The hue factor to apply.
        # s=dict(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        p=0.6,  # probability of applying the color_jitter transformation
        mean=mean,
        std=std,
        data=data,
        target=None)

    data, _ = gray_scale(gray=gray, data=data, target=None)

    data, _ = gaussian_blur(
        blur=blur_p,
        data=data,
        target=None
    )

    data, target = flip_lr_and_ud(flip_lr, flip_ud, data, target)
    return data, target


def flip_lr_and_ud(flip_lr, flip_ud, data=None, target=None):
    # 水平翻转
    if flip_lr > 0.5:
        if data is not None:
            data = torch.flip(data, dims=[-1])
        target = torch.flip(target, dims=[-1])
    # 垂直翻转
    if flip_ud > 0.5:
        if data is not None:
            data = torch.flip(data, dims=[-2])
        target = torch.flip(target, dims=[-2])
    return data, target


def gray_scale(gray, data=None, target=None, p=0.9):
    if not (data is None):
        if data.shape[1] == 3:
            if gray > p:
                aug = kornia.augmentation.RandomGrayscale(p=1.0)
                data = aug(data)
    return data, target


def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[-3] == 3:  # 支持 C H W 或者 B C H W
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target
