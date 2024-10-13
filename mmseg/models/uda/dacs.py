# -*- coding:utf-8 -*-
"""
 @FileName   : dacs.py
 @Time       : 9/24/24 3:02 PM
 @Author     : Woldier Wong
 @Description:

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs
"""
import torch, random, math, mmcv, os, torch.nn.functional as F

from copy import deepcopy
from matplotlib import pyplot as plt
from mmcv.runner import master_only
from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.uda.teacher_model import EMATeacher
from mmseg.models.utils.dacs_transform import get_mean_std, get_class_masks, strong_transform, denorm
from mmseg.models.utils.visualization import subplotimg, _PALETTE_DICT
# link use
from mmseg.models import EncoderDecoder
# contrastive lmodel
# from mmseg.models.uda.simsiam import SimSiam
# from mmseg.models.utils.sim_aug_transform import tow_transform

from mmseg.models.uda.masking_consistency_module import MaskingConsistencyModule


@UDA.register_module()
class DACS(UDADecorator):
    def __init__(self, **cfg):
        """
        DACS UDA
        Parameters:
            model (dict): some with UDADecorator.
            max_iters: Maximum number of iterations. 最大迭代次数.

            imnet_feature_dist_lambda (float):  dacs  feature dist. not used here, so we set this to .0.
            imnet_feature_dist_classes:
            imnet_feature_dist_scale_min_ratio:

            mix (str): mix 的 type 现在仅仅支持class
            blur (bool): 在class mix 时是否使用 blur
            color_jitter_strength: color_jitter 的 参数, 支持float或者dict
                当s为float时
                brightness=s, contrast=s, saturation=s, hue=s
                当为dict时
                brightness=s["brightness"], contrast=s["contrast"], saturation=s["saturation"], hue=s["saturation"]
            color_jitter_probability: Thresholding with color_jitter. 采用 color_jitter 的阈值

            debug_img_interval (int): Number of interval iterations for visualization outputs. 可视化输出的间隔迭代次数. 默认1000
            print_grad_magnitude (bool): 打印梯度范数信息, 默认为False.

            EMATeacher:
                alpha (float): EMA alpha. 移动指数平滑中的alpha.
                pseudo_threshold (float): pseudo label confidence threshold
                psweight_ignore_top (int): Ignore the upper part of the pixel interval [0, psweight_ignore_top], default 0.
                    忽略上部分的像素区间[0, psweight_ignore_top], 默认为0.
                psweight_ignore_bottom (int): Ignore the lower portion of the pixel interval [0, psweight_ignore_bottom], default 0.
                    忽略下部分的像素区间[0, psweight_ignore_bottom], 默认为0.




        """
        super(DACS, self).__init__(**cfg)
        self._init_param(**cfg)
        # build ema teacher
        # self.ema_model = build_segmentor(ema_cfg)
        self.ema_model = EMATeacher(**cfg)

        self.mic = None
        if self.enable_masking:
            self.mic = MaskingConsistencyModule(require_teacher=False, **cfg)

        assert self.train_cfg.get("work_dir", None), "work_dir is None! please set uda.train_cfg.work_dir"
        assert self.train_cfg.get("cmap",
                                  None), "cmap is None! please set uda.train_cfg.cmap. available keys" + "|".join(
            _PALETTE_DICT.keys())

    def _init_param(
            self,
            max_iters=None,
            mix=None, blur=None, color_jitter_strength=None, color_jitter_probability=None,
            debug_img_interval=None, # print_grad_magnitude=None,
            enable_masking: bool = True,
            **kwargs
    ):
        self.local_iter = 0
        self.max_iters = max_iters
        self.mix = mix
        self.blur = blur
        self.color_jitter_s = color_jitter_strength
        self.color_jitter_p = color_jitter_probability
        self.debug_img_interval = debug_img_interval
        # self.print_grad_magnitude = print_grad_magnitude
        assert self.mix == 'class'

        # self.debug_fdist_mask = None
        # self.debug_gt_rescale = None

        self.enable_masking = enable_masking

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def update_weights(self, iter: int):
        self.get_ema_model().update_weights(self.get_model(), iter)

    def _prepare_strong_transform_param(self, means, stds):
        """
        img_metas: 图片的元数据
        dev: 设备id
        """

        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }
        return strong_parameters

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def forward_train(
            self,
            img, img_metas, gt_semantic_seg,
            target_img, target_img_metas, **kwargs
    ):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            target_img (Tensor): target input images.
            target_img_metas (list[dict]): same to img_metas

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_size, dev, log_vars, means, stds, strong_parameters = self._forward_setup(img, img_metas)

        # ==========================Train on source images==============================
        # clean_losses = self.get_model().forward_train(
        self._forward_train_source(img, img_metas, gt_semantic_seg, log_vars)

        # =============================Generate pseudo-label===========================
        gt_pixel_weight, pseudo_label, pseudo_weight = self._forward_pseudo_label_gen(target_img, target_img_metas, dev)
        # ================================Apply mixing=======================
        mix_masks, mixed_img, mixed_lbl, mixed_seg_weight = self._forward_mix(img, gt_semantic_seg, gt_pixel_weight,
                                                                              target_img, pseudo_label, pseudo_weight,
                                                                              batch_size, strong_parameters)

        # ============================Train on mixed images===================================
        self._forward_train_mix_img(mixed_img, img_metas, mixed_lbl, mixed_seg_weight, log_vars)
        # ==============================Masked Training======================================
        self._forward_train_mic(img, img_metas, gt_semantic_seg, target_img, target_img_metas, pseudo_label,
                                pseudo_weight, log_vars)
        # ====================================== vis =======================================

        self._show_img(
            img, gt_semantic_seg,
            target_img, pseudo_label, mixed_seg_weight,
            mixed_img, mixed_lbl, mix_masks,
            batch_size, means, stds,
            # q=q, k=k
        )
        self.local_iter += 1

        return log_vars

    def _forward_train_mic(self, img, img_metas, gt_semantic_seg, target_img, target_img_metas, pseudo_label,
                           pseudo_weight, log_vars):
        if self.enable_masking:
            masked_loss = self.mic(self.model, img, img_metas,
                                   gt_semantic_seg, target_img,
                                   target_img_metas, None,
                                   pseudo_label, pseudo_weight)
            masked_loss = add_prefix(masked_loss, 'masked')
            masked_loss, masked_log_vars = self._parse_losses(masked_loss)
            log_vars.update(masked_log_vars)
            masked_loss.backward()

    def _forward_train_mix_img(self, mixed_img, img_metas, mixed_lbl, mixed_seg_weight, log_vars):
        # mix_losses = self.get_model().forward_train(
        mix_losses = self.model.forward(
            mixed_img, img_metas, gt_semantic_seg=mixed_lbl, seg_weight=mixed_seg_weight, return_feat=True)
        EncoderDecoder.forward_train
        mix_losses.pop('features')
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        mix_loss.backward()

    @staticmethod
    def _forward_mix(img, gt_semantic_seg, gt_pixel_weight, target_img, pseudo_label, pseudo_weight, batch_size,
                     strong_parameters):
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mixed_seg_weight = pseudo_weight.clone()
        mix_masks = get_class_masks(gt_semantic_seg)  # TODO 这里选择的label可能将255包含进去了, 可能需要额外的操作滤除255
        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, mixed_seg_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], mixed_seg_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)
        return mix_masks, mixed_img, mixed_lbl, mixed_seg_weight

    def _forward_pseudo_label_gen(self, target_img, target_img_metas, dev):
        pseudo_label, pseudo_weight = self.get_ema_model()(target_img, target_img_metas, valid_pseudo_mask=None)
        EMATeacher.__call__
        gt_pixel_weight = torch.ones(pseudo_weight.shape, device=dev)  # 对于source, 其损失权重总是为1
        return gt_pixel_weight, pseudo_label, pseudo_weight

    def _forward_train_source(self, img, img_metas, gt_semantic_seg, log_vars):
        clean_losses = self.model.forward(
            img, img_metas, gt_semantic_seg=gt_semantic_seg, return_feat=True)
        EncoderDecoder.forward_train
        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward()

    def _forward_setup(self, img, img_metas):
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device
        # Init/update ema model
        self.update_weights(self.local_iter)
        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)
        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = self._prepare_strong_transform_param(means, stds)
        return batch_size, dev, log_vars, means, stds, strong_parameters

    @master_only
    def _show_img(
            self,
            img, gt_semantic_seg,
            target_img, pseudo_label, pseudo_weight,
            mixed_img, mixed_lbl, mix_masks,
            batch_size, means, stds,
            # q=None, k=None
    ):
        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'class_mix_debug')
            cmap = self.train_cfg['cmap']
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            rows, cols = 2, 4
            for j in range(batch_size):
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0.05,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap=cmap)
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap=cmap)
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                subplotimg(
                    axs[1][3], mixed_lbl[j], 'Seg Targ', cmap=cmap)
                subplotimg(
                    axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
