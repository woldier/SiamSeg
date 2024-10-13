# -*- coding:utf-8 -*-
"""
 @FileName   : visualization.py.py
 @Time       : 9/24/24 5:17 PM
 @Author     : Woldier Wong
 @Description: https://github.com/lhoyer/DAFormer/blob/master/mmseg/models/utils/visualization.py
"""
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from mmseg.datasets import ISPRSDataset, LoveDADataset

_ISPRS_PALETTE = [color for sublist in ISPRSDataset.PALETTE for color in sublist]  # 转为1维数组 L* 3
_LoveDA_PALETTE = [color for sublist in LoveDADataset.PALETTE for color in sublist]
_PALETTE_DICT = {
    "isprs": _ISPRS_PALETTE,
    "loveda": _LoveDA_PALETTE
}


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def _colorize(img, cmap, mask_zero=False):
    vmin = np.min(img)
    vmax = np.max(img)
    mask = (img <= 0).squeeze()
    cm = plt.get_cmap(cmap)
    colored_image = cm(np.clip(img.squeeze(), vmin, vmax) / vmax)[:, :, :3]
    # Use white if no depth is available (<= 0)
    if mask_zero:
        colored_image[mask, :] = [1, 1, 1]
    return colored_image


def subplotimg(ax,
               img,
               title,
               range_in_title=False,
               **kwargs):
    if img is None:
        return
    with torch.no_grad():
        if torch.is_tensor(img):
            img = img.cpu()
        if len(img.shape) == 2:
            if torch.is_tensor(img):
                img = img.numpy()
        elif img.shape[0] == 1:
            if torch.is_tensor(img):
                img = img.numpy()
            img = img.squeeze(0)
        elif img.shape[0] == 3:
            img = img.permute(1, 2, 0)
            if not torch.is_tensor(img):
                img = img.numpy()
        # 排除 cmap="grey" 的情况 这种情况要将 其传给ax.imshow(img, **kwargs)
        if kwargs.get("cmap", "") in _PALETTE_DICT.keys():
            cmap = kwargs.pop('cmap')
            # assert str.lower(cmap) in _PALETTE_DICT.keys(), "support " + "|".join(_PALETTE_DICT.keys()) + f"got {cmap}"
            if torch.is_tensor(img):
                img = img.numpy()
            img = colorize_mask(img, _PALETTE_DICT[cmap])

    if range_in_title:
        vmin = np.min(img)
        vmax = np.max(img)
        title += f' {vmin:.3f}-{vmax:.3f}'

    ax.imshow(img, **kwargs)
    ax.set_title(title)
