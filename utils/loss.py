import torch
from torch import nn
from config import cfg
from box import bbox_iou


class FocalLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.loss_fn = nn.BCELoss(reduction='none')

    def forward(self, pred, label):
        loss = self.loss_fn(pred, label)

        alpha_t = label * cfg.alpha + (1 - label) * (1 - cfg.alpha)
        p_t = label * pred + (1 - label) * (1 - pred)

        loss *= alpha_t * (1.0 - p_t) ** cfg.gamma

        if self.reduction == 'mean':
            return loss.mean()

        elif self.reduction == 'sum':
            return loss.sum()

        else:
            return loss


class ComputeLoss:
    def __init__(self, model, anchors, strides, num_cls):

        self.device = next(model.parameters()).device
        self.anchors = anchors
        self.strides = strides
        self.num_cls = num_cls

        self.num_layers = len(anchors)

        self.pos, self.neg = 1.0 - 0.5 * cfg.eps, 0.5 * cfg.eps

        if cfg.focal_loss:
            self.bce_obj, self.bce_cls = FocalLoss(), FocalLoss()

        else:
            self.bce_obj, self.bce_cls = nn.BCELoss(), nn.BCELoss()

    def __call__(self, preds, targets):
        box_loss = torch.zeros(1, device=self.device)
        obj_loss = torch.zeros(1, device=self.device)
        cls_loss = torch.zeros(1, device=self.device)

        tcls, tbox, indices, scale_anchors = self.build_targets(preds, targets)

        for i, p in enumerate(preds):
            img, ac_index, grid_j, grid_i = indices[i]

            tobj = torch.zeros(p.shape[:4], dtype=p.dtype, device=self.device)

            num = img.shape[0]
            if num > 0:
                p_xy, p_wh, _, p_cls = p[img, ac_index, grid_j, grid_i].split((2, 2, 1, self.num_cls), 1)

                # box loss
                p_xy = p_xy * 2 - 0.5
                p_wh = (p_wh * 2) ** 2 * scale_anchors[i]
                p_box = torch.cat((p_xy, p_wh), 1)
                iou = bbox_iou(p_box, tbox[i], type='CIoU').squeeze()
                box_loss += (1.0 - iou).mean()

                # yolo_labels loss
                iou = iou.detach().clamp(0).type(tobj.dtype)
                tobj[img, ac_index, grid_j, grid_i] = iou
                obj_loss += self.bce_obj(p[..., 4], tobj) * cfg.balance[i]

                # cls loss
                t_cls = torch.full_like(p_cls, self.neg, device=self.device)
                t_cls[range(num), tcls[i]] = self.pos
                cls_loss += self.bce_cls(p_cls, t_cls)

        box_loss *= cfg.box_w
        obj_loss *= cfg.obj_w
        cls_loss *= cfg.cls_w

        return (box_loss + obj_loss + cls_loss) * tobj.shape[0], torch.cat((box_loss, obj_loss, cls_loss)).detach()

    def build_targets(self, preds, targets):
        num_ac, num_t = len(self.anchors[0]), len(targets)
        tcls, tbox, indices, scale_anchors = [], [], [], []

        gain = torch.ones(7, device=self.device)
        ac_index = torch.arange(num_ac, device=self.device).float().view(num_ac, 1).repeat(1, num_t)
        targets = torch.cat((targets.repeat(num_ac, 1, 1), ac_index[..., None]), 2)

        offset = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=self.device).float() * 0.5

        for i in range(self.num_layers):
            scale_anchor, shape = self.anchors[i] / self.strides[i], preds[i].shape

            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]

            # Match targets to anchors
            g_targets = targets * gain

            # Matches
            ratio = g_targets[..., 4:6] / scale_anchor[:, None]
            match_index = torch.max(ratio, 1 / ratio).max(2)[0] < cfg.anchor_t
            g_targets = g_targets[match_index]

            # Offsets
            grid_xy = g_targets[:, 2:4]
            grid_xy_inv = gain[[2, 3]] - grid_xy
            left, down = ((grid_xy % 1 < 0.5) & (grid_xy > 1)).T
            right, up = ((grid_xy_inv % 1 < 0.5) & (grid_xy_inv > 1)).T
            adjoin_index = torch.stack((torch.ones_like(left), left, down, right, up))
            g_targets = g_targets.repeat((5, 1, 1))[adjoin_index]
            offsets = (torch.zeros_like(grid_xy)[None] + offset[:, None])[adjoin_index]

            # Define
            img_cls, grid_xy, grid_wh, ac_index = g_targets.chunk(4, 1)
            ac_index, (img, cls) = ac_index.long().view(-1), img_cls.long().T
            grid_ij = (grid_xy - offsets).long()
            grid_i, grid_j = grid_ij.T

            # Append
            tcls.append(cls)
            tbox.append(torch.cat((grid_xy - grid_ij, grid_wh), 1))
            indices.append((img, ac_index, grid_j.clamp_(0, shape[2] - 1), grid_i.clamp_(0, shape[3] - 1)))
            scale_anchors.append(scale_anchor[ac_index])

        return tcls, tbox, indices, scale_anchors
