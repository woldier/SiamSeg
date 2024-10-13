# -*- coding:utf-8 -*-
"""
 @FileName   : masking_consistency_module.py.py
 @Time       : 10/3/24 9:32 PM
 @Author     : Woldier Wong
 @Description:
 https://github.com/lhoyer/MIC/blob/2f932a98b5dd9f598aaeb32411863ceea0809314/seg/mmseg/models/uda/masking_consistency_module.py
"""

import random

import torch
from torch.nn import Module

from mmseg.models.uda.teacher_model import EMATeacher
from mmseg.models.utils.dacs_transform import get_mean_std, strong_transform
from mmseg.models.utils.masking_transforms import build_mask_generator

from typing import Optional, Union


class MaskingConsistencyModule(Module):

    def __init__(
            self,
            require_teacher,  # cfg
            color_jitter_strength=None, color_jitter_probability=None,
            mask_mode: str = 'separatetrgaug', mask_alpha: Union[str, float] = "same",
            mask_pseudo_threshold: Union[str, float] = "same", mask_lambda: float = 1.,
            mask_generator: Optional[dict] = None, **kwargs
    ):
        """
        args:
            mask_mode: Apply masking to color-augmented target images
            mask_alpha: EMA teacher alpha .default 'same', use the same teacher alpha for MIC as for DAFormer self-training (0.999).
            mask_pseudo_threshold:
            mask_lambda:
            mask_generator:
        """
        super(MaskingConsistencyModule, self).__init__()

        # self.source_only = cfg.get('source_only', False)
        # self.max_iters = cfg['max_iters']
        self.color_jitter_s = color_jitter_strength
        self.color_jitter_p = color_jitter_probability
        self.source_only = kwargs.get('source_only', False)
        self.mask_mode = mask_mode
        self.mask_alpha = mask_alpha
        self.mask_pseudo_threshold = mask_pseudo_threshold
        self.mask_lambda = mask_lambda
        if mask_generator is None:
            mask_generator = dict(
                type='block', mask_ratio=0.7, mask_block_size=64)
        self.mask_gen = build_mask_generator(mask_generator)

        assert self.mask_mode in [
            'separate', 'separatesrc', 'separatetrg', 'separateaug',
            'separatesrcaug', 'separatetrgaug'
        ]

        self.teacher = None
        if require_teacher or \
                self.mask_alpha != 'same' or \
                self.mask_pseudo_threshold != 'same':
            # self.teacher = EMATeacher(use_mask_params=True, cfg=cfg)
            self.teacher = EMATeacher(**kwargs)

    def update_weights(self, model, iter):
        if self.teacher is not None:
            self.teacher.update_weights(model, iter)

    def __call__(self,
                 model,
                 img,
                 img_metas,
                 gt_semantic_seg,
                 target_img,
                 target_img_metas,
                 valid_pseudo_mask,
                 pseudo_label=None,
                 pseudo_weight=None):
        # self.update_debug_state()
        # self.debug_output = {}
        # model.debug_output = {}
        dev = img.device
        means, stds = get_mean_std(img_metas, dev)

        if not self.source_only:
            # Share the pseudo labels with the host UDA method
            if self.teacher is None:
                assert self.mask_alpha == 'same'
                assert self.mask_pseudo_threshold == 'same'
                assert pseudo_label is not None
                assert pseudo_weight is not None
                masked_plabel = pseudo_label
                masked_pweight = pseudo_weight
            # Use a separate EMA teacher for MIC
            else:
                masked_plabel, masked_pweight = \
                    self.teacher(
                        target_img, target_img_metas, valid_pseudo_mask)
        # Don't use target images at all
        if self.source_only:
            masked_img = img
            masked_lbl = gt_semantic_seg
            b, _, h, w = gt_semantic_seg.shape
            masked_seg_weight = None
        # Use 1x source image and 1x target image for MIC
        elif self.mask_mode in ['separate', 'separateaug']:
            assert img.shape[0] == 2
            masked_img = torch.stack([img[0], target_img[0]])
            masked_lbl = torch.stack(
                [gt_semantic_seg[0], masked_plabel[0].unsqueeze(0)])
            gt_pixel_weight = torch.ones(masked_pweight[0].shape, device=dev)
            masked_seg_weight = torch.stack(
                [gt_pixel_weight, masked_pweight[0]])
        # Use only source images for MIC
        elif self.mask_mode in ['separatesrc', 'separatesrcaug']:
            masked_img = img
            masked_lbl = gt_semantic_seg
            masked_seg_weight = None
        # Use only target images for MIC
        elif self.mask_mode in ['separatetrg', 'separatetrgaug']:
            masked_img = target_img
            masked_lbl = masked_plabel.unsqueeze(1)
            masked_seg_weight = masked_pweight
        else:
            raise NotImplementedError(self.mask_mode)

        # Apply color augmentation
        if 'aug' in self.mask_mode:
            strong_parameters = {
                'mix': None,
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': self.color_jitter_s,
                'color_jitter_p': self.color_jitter_p,
                'blur': random.uniform(0, 1),
                'mean': means[0].unsqueeze(0),
                'std': stds[0].unsqueeze(0)
            }
            masked_img, _ = strong_transform(
                strong_parameters, data=masked_img.clone())

        # Apply masking to image
        masked_img = self.mask_gen.mask_image(masked_img)

        # Train on masked images
        # masked_loss = model.forward_train(
        masked_loss = model(
            masked_img,
            img_metas,
            gt_semantic_seg=masked_lbl,
            seg_weight=masked_seg_weight,
        )
        if self.mask_lambda != 1:
            masked_loss['decode.loss_seg'] *= self.mask_lambda

        return masked_loss
