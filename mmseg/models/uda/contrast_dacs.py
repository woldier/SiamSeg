# -*- coding:utf-8 -*-
"""
 @FileName   : contrast_dacs.py
 @Time       : 10/4/24 9:47 AM
 @Author     : Woldier Wong
 @Description: TODO
"""

import torch, os
from .dacs import DACS, get_module, add_prefix, get_mean_std, get_class_masks, strong_transform, plt, subplotimg, \
    denorm, master_only
from mmseg.models import UDA
# contrastive model
from mmseg.models.uda.simsiam import SimSiam
from mmseg.models.utils.sim_aug_transform import tow_transform


@UDA.register_module()
class ContrastDACS(DACS):

    def __init__(self, **cfg):
        """
        ContrastDACS
            Parameters:
                contras_model_cfg (dict):
                    mod (str): 在那些域上做对比学习. 默认为all 即source个target 都会参与对比学习. 支持 ["all", "source", "target"]
                    dim: feature dimension (default: 2048)
                    pred_dim: hidden dimension of the predictor (default: 512)
        """
        cfg["enable_masking"] = False
        super().__init__(**cfg)
        # build contrast model
        contras_model_cfg = cfg.get("contras_model_cfg", {})
        self.contras_mod = contras_model_cfg.pop("mod", "all")
        assert self.contras_mod in ["all", "source", "target"]
        self.contras_model = SimSiam(backbone=self.model.backbone, **contras_model_cfg)
        self.contras_criterion = torch.nn.CosineSimilarity(dim=1)

    def get_contrast_model(self):
        return get_module(self.contras_model)

    @staticmethod
    def contrast_aug(img, pseudo_label, means, stds):
        q, q_label, k, k_label = [], [], [], []
        for i in range(img.shape[0]):
            _q, _q_label, _k, _k_label = tow_transform(data=img[i::img.shape[0]], target=pseudo_label[i::img.shape[0]],
                                                       mean=means[i::img.shape[0]], std=stds[i::img.shape[0]])
            q.append(_q)
            q_label.append(_q_label)
            k.append(_k)
            k_label.append(_k_label)
        q, q_label, k, k_label = torch.cat(q, dim=0), torch.cat(q_label, dim=0), torch.cat(k, dim=0), torch.cat(
            k_label, dim=0)
        return q, q_label, k, k_label

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img, target_img_metas, **kwargs):
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

        # ============================ contrast learning ==============================
        q_tgt, q_tgt_pseudo_label, k_tgt, k_tgt_pseudo_label = self.contrast_aug(target_img, pseudo_label.unsqueeze(1),
                                                                                 means, stds)
        contras_mod = self.contras_mod
        if contras_mod == "target":
            q, k = q_tgt, k_tgt
        elif contras_mod in ["source", "all"]:
            # q, q_label, k, k_label = tow_transform(img, gt_semantic_seg, mean=means, std=stds)
            q, q_label, k, k_label = self.contrast_aug(img, gt_semantic_seg, means, stds)
            if contras_mod == "all":
                q, k = torch.cat((q, q_tgt), dim=0), torch.cat((k, k_tgt), dim=0)
        else:
            raise AttributeError()
        if contras_mod == "all":
            p1_t, p2_t, z1_t, z2_t = self.get_contrast_model()(x1=q[:batch_size:], x2=k[:batch_size:])  # for target
            p1_s, p2_s, z1_s, z2_s = self.get_contrast_model()(x1=q[-batch_size::], x2=k[-batch_size::])  # for source
            p1, p2, z1, z2 = torch.cat((p1_t, p1_s), dim=0), torch.cat((p2_t, p2_s), dim=0), \
                torch.cat((z1_t, z1_s), dim=0), torch.cat((z2_t, z2_s), dim=0),
        else:
            p1, p2, z1, z2 = self.get_contrast_model()(x1=q, x2=k)
        contras_loss = -(get_module(self.contras_criterion)(p1, z2).mean() +
                         get_module(self.contras_criterion)(p2, z1).mean()) * 0.5

        contras_loss.backward()
        log_vars.update({"contras_loss": contras_loss.item()})

        # ==============================Masked Training======================================
        self._forward_train_mic(img, img_metas, gt_semantic_seg, target_img, target_img_metas, pseudo_label,
                                pseudo_weight, log_vars)
        # ====================================== vis =======================================

        self._show_img(
            img, gt_semantic_seg,
            target_img, pseudo_label, mixed_seg_weight,
            mixed_img, mixed_lbl, mix_masks,
            batch_size, means, stds,
            q=q, k=k
        )
        self.local_iter += 1

        return log_vars

    @master_only
    def _show_img(
            self,
            img, gt_semantic_seg,
            target_img, pseudo_label, pseudo_weight,
            mixed_img, mixed_lbl, mix_masks,
            batch_size, means, stds,
            q=None, k=None
    ):
        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'class_mix_debug')
            cmap = self.train_cfg['cmap']
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            if self.contras_mod in ["source", "all"]:
                source_img_aux1, source_img_aux2 = q[-batch_size::], k[
                                                                     -batch_size::]  # 如果有source 那么因为是cat操作, 只会出现在后面的batch
                source_img_aux1, source_img_aux2 = torch.clamp(denorm(source_img_aux1, means, stds), 0, 1), torch.clamp(
                    denorm(source_img_aux2, means, stds), 0, 1)
            if self.contras_mod in ["target", "all"]:
                target_img_aux1, target_img_aux2 = q[:batch_size:], k[:batch_size:]
                target_img_aux1, target_img_aux2 = torch.clamp(denorm(target_img_aux1, means, stds), 0, 1), torch.clamp(
                    denorm(target_img_aux2, means, stds), 0, 1)
            rows, cols = 3, 4
            # rows, cols = 2, 4
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
                if self.contras_mod in ["source", "all"]:
                    subplotimg(axs[2][0], source_img_aux1[j], 'Source Q Image(AUG 1)', vmin=0, vmax=1)
                    subplotimg(axs[2][1], source_img_aux2[j], 'Source K Image(AUG 2)', vmin=0, vmax=1)
                if self.contras_mod in ["target", "all"]:
                    subplotimg(axs[2][2], target_img_aux1[j], 'Target Q Image(AUG 1)', vmin=0, vmax=1)
                    subplotimg(axs[2][3], target_img_aux2[j], 'Target K Image(AUG 2)', vmin=0, vmax=1)
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
