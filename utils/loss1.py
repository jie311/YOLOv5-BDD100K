import torch
from torch import nn
from box import bbox_iou


class FocalLoss(nn.Module):
    def __init__(self, loss_fn, alpha, gamma):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_fn = loss_fn
        self.loss_fn.reduction = 'none'

    def forward(self, pred, label):
        loss = self.loss_fn(pred, label)

        pred_prob = torch.sigmoid(pred)
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        alpha_t = label * self.alpha + (1 - label) * (1 - self.alpha)
        loss *= alpha_t * (1.0 - p_t) ** self.gamma

        return loss.mean()


class Loss:
    def __init__(self, anchors, strides, anchor_t, iou_rate, obj_pw, cls_pw,
                 alpha, gamma, smooth, balance, box_w, obj_w, cls_w, device):
        self.anchors = anchors
        self.strides = strides
        self.anchor_t = anchor_t
        self.iou_rate = iou_rate
        self.box_w = box_w
        self.obj_w = obj_w
        self.cls_w = cls_w
        self.device = device

        bce_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pw], device=device))
        bce_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cls_pw], device=device))

        if alpha or gamma:
            bce_obj = FocalLoss(bce_obj, alpha, gamma)
            bce_cls = FocalLoss(bce_cls, alpha, gamma)

        self.bce_obj = bce_obj
        self.bce_cls = bce_cls

        self.pos, self.neg = 1.0 - 0.5 * smooth, 0.5 * smooth
        self.balance = balance  # 3 layers: [4.0, 1.0, 0.4];  5 layers: [4.0, 1.0, 0.25, 0.06, 0.02]

    def __call__(self, preds, targets):
        box_loss = torch.zeros(1, device=self.device)
        obj_loss = torch.zeros(1, device=self.device)
        cls_loss = torch.zeros(1, device=self.device)

        t_cls, t_boxes, t_indices, t_anchors = self.build_targets(preds, targets)

        for i, p in enumerate(preds):
            img, anchor_idx, grid_j, grid_i = t_indices[i]

            num = len(img)
            if num:
                pt = p[img, anchor_idx, grid_j, grid_i]

                # box loss
                p_xy = pt[:, :2] * 2 - 0.5
                p_wh = (pt[:, 2:4] * 2) ** 2 * t_anchors[i]
                p_box = torch.cat((p_xy, p_wh), 1)
                
                # TODO xywh2xyxy
                iou = bbox_iou(p_box, t_boxes[i], type='CIoU')
                box_loss += (1.0 - iou).mean()

                # obj loss
                iou = iou.detach().clamp(0)
                obj = torch.zeros_like(p[:, :, :, :, 0], device=self.device)
                obj[img, anchor_idx, grid_j, grid_i] = (1.0 - self.iou_rate) + self.iou_rate * iou
                obj_loss += self.bce_obj(p[:, :, :, :, 4], obj) * self.balance[i]

                # cls loss
                cls = torch.full_like(pt[:, 5:], self.neg, device=self.device)
                cls[range(num), t_cls[i]] = self.pos
                cls_loss += self.bce_cls(pt[:, 5:], cls)

            else:
                # obj loss
                obj = torch.zeros_like(p[:, :, :, :, 0], device=self.device)
                obj_loss += self.bce_obj(p[:, :, :, :, 4], obj) * self.balance[i]

        box_loss *= self.box_w
        obj_loss *= self.obj_w
        cls_loss *= self.cls_w

        return (box_loss + obj_loss + cls_loss) * len(obj), torch.cat((box_loss, obj_loss, cls_loss)).detach()

    def build_targets(self, preds, targets):
        t_cls, t_boxes, t_indices, t_anchors = [], [], [], []

        num_ac, num_t = len(self.anchors[0]), len(targets)
        targets = torch.cat(
            (
                targets.repeat(num_ac, 1, 1),
                torch.arange(num_ac, device=self.device).view(num_ac, 1).repeat(1, num_t)[:, :, None]
            ), 2)

        gain = torch.ones(7, device=self.device)
        offset = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=self.device).float() * 0.5
        for i in range(len(self.anchors)):
            # Anchor in FeatureMap
            t_anchor, shape = self.anchors[i] / self.strides[i], preds[i].shape

            # Targets in FeatureMap
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]
            g_targets = targets * gain

            # Match
            ratio = g_targets[:, :, 4:6] / t_anchor[:, None]
            match_idx = torch.max(ratio, 1 / ratio).max(2)[0] < self.anchor_t
            g_targets = g_targets[match_idx]

            # Offset
            grid_xy = g_targets[:, 2:4]
            grid_xy_inv = gain[[2, 3]] - grid_xy
            left, up = ((grid_xy % 1 < 0.5) & (grid_xy > 1)).T
            right, down = ((grid_xy_inv % 1 < 0.5) & (grid_xy_inv > 1)).T
            adjoin_idx = torch.stack((torch.ones_like(left), left, up, right, down))
            g_targets = g_targets.repeat((5, 1, 1))[adjoin_idx]
            offsets = (torch.zeros_like(grid_xy)[None] + offset[:, None])[adjoin_idx]

            # Define
            img_cls, grid_xy, grid_wh, anchor_idx = g_targets.chunk(4, 1)
            anchor_idx, (img, cls) = anchor_idx.long().view(-1), img_cls.long().T
            grid_ij = (grid_xy - offsets).long()
            grid_i, grid_j = grid_ij.T

            t_cls.append(cls)
            t_boxes.append(torch.cat((grid_xy - grid_ij, grid_wh), 1))
            t_indices.append((img, anchor_idx, grid_j.clamp_(0, shape[2] - 1), grid_i.clamp_(0, shape[3] - 1)))
            t_anchors.append(t_anchor[anchor_idx])

        return t_cls, t_boxes, t_indices, t_anchors


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

preds = torch.ones([1, 16, 3, 80, 80, 80])
targets = torch.tensor([[0, 0, 0.11, 0.21, 0.25, 0.25],
                        [0, 1, 0.22, 0.22, 0.35, 0.35],
                        [1, 1, 0.33, 0.33, 0.45, 0.45]], device=device)

cfg = {'anchors': [[32, 64, 64, 128, 128, 256]], 'strides': [8], 'anchor_t': 4,
       'iou_rate': 0, 'obj_pw': 1, 'cls_pw': 1, 'alpha': 1, 'gamma': 1, 'smooth': 1,
       'balance': [1, 1, 1]}
loss = Loss(cfg, device)

loss.build_targets(preds, targets)
