import torch
from utils.boxs_utils import box_iou
from losses.commons import smooth_l1_loss, mean_max
from utils.model_utils import reduce_sum


def negative_focal_loss(logits, gamma):
    return -logits ** gamma * ((1 - logits).clamp(min=1e-12).log())


class BoxCoder(object):
    def __init__(self, weights=None):
        super(BoxCoder, self).__init__()
        if weights is None:
            weights = [0.1, 0.1, 0.2, 0.2]
        self.weights = torch.tensor(data=weights, requires_grad=False)

    def encoder(self, anchors, gt_boxes):
        """
        :param gt_boxes:[box_num, 4]
        :param anchors: [box_num, 4]
        :return:
        """
        if self.weights.device != anchors.device:
            self.weights = self.weights.to(anchors.device)
        anchors_wh = anchors[..., [2, 3]] - anchors[..., [0, 1]]
        anchors_xy = anchors[..., [0, 1]] + 0.5 * anchors_wh
        gt_wh = (gt_boxes[..., [2, 3]] - gt_boxes[..., [0, 1]]).clamp(min=1.0)
        gt_xy = gt_boxes[..., [0, 1]] + 0.5 * gt_wh
        delta_xy = (gt_xy - anchors_xy) / anchors_wh
        delta_wh = (gt_wh / anchors_wh).log()

        delta_targets = torch.cat([delta_xy, delta_wh], dim=-1) / self.weights

        return delta_targets

    def decoder(self, predicts, anchors):
        """
        :param predicts: [anchor_num, 4] or [bs, anchor_num, 4]
        :param anchors: [anchor_num, 4]
        :return: [anchor_num, 4] (x1,y1,x2,y2)
        """
        if self.weights.device != anchors.device:
            self.weights = self.weights.to(anchors.device)
        anchors_wh = anchors[:, [2, 3]] - anchors[:, [0, 1]]
        anchors_xy = anchors[:, [0, 1]] + 0.5 * anchors_wh
        scale_reg = predicts * self.weights
        scale_reg[..., :2] = anchors_xy + scale_reg[..., :2] * anchors_wh
        scale_reg[..., 2:] = scale_reg[..., 2:].exp() * anchors_wh
        scale_reg[..., :2] -= (0.5 * scale_reg[..., 2:])
        scale_reg[..., 2:] = scale_reg[..., :2] + scale_reg[..., 2:]

        return scale_reg


class FreeAnchorLoss(object):
    def __init__(self,
                 gamma=2.0,
                 alpha=0.25,
                 top_k=64,
                 iou_thresh=0.6,
                 reg_weight=0.75,
                 beta=1. / 9,
                 dist_train=True):
        super(FreeAnchorLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.top_k = top_k
        self.iou_thresh = iou_thresh
        self.reg_weight = reg_weight
        self.beta = beta
        self.dist_train = dist_train
        self.box_coder = BoxCoder()

    def __call__(self, cls_predicts, reg_predicts, anchors, targets):
        cls_predicts = torch.cat([item for item in cls_predicts], dim=1)
        reg_predicts = torch.cat([item for item in reg_predicts], dim=1)
        all_anchors = torch.cat([item for item in anchors])
        gt_boxes = targets['target'].split(targets['batch_len'])
        bs, _, cls_num = cls_predicts.shape
        device = cls_predicts.device
        batch_idx = list()
        match_anchor_idx = list()
        match_cls_idx = list()
        batch_cls_target = list()
        gt_num = 0
        for bid, cls_pred, reg_pred, gt in zip(range(bs), cls_predicts, reg_predicts, gt_boxes):
            gt_num += len(gt)
            if len(gt) == 0:
                continue
            batch_idx.append(bid)
            labels = gt[:, 0].long()
            # ------------------ negative_loss start ------------------
            box_localization = self.box_coder.decoder(reg_pred.detach(), all_anchors)
            object_box_iou = box_iou(gt[:, 1:], box_localization)
            t1 = self.iou_thresh
            t2 = object_box_iou.max(dim=1, keepdim=True).values.clamp(min=t1 + 1e-12)
            object_box_prob = (
                    (object_box_iou - t1) / (t2 - t1)
            ).clamp(min=0, max=1)
            image_box_prob = torch.zeros_like(cls_pred).type_as(object_box_prob)
            for obj_box_prob, lid in zip(object_box_prob, labels):
                pre = image_box_prob[:, lid]
                cur = torch.where(obj_box_prob > pre, obj_box_prob, pre)
                image_box_prob[:, lid] = cur
            batch_cls_target.append(image_box_prob)
            # ------------------ negative_loss end ------------------

            # ------------------ positive_loss start ------------------
            match_quality_matrix = box_iou(gt[:, 1:], all_anchors)
            _, matched = torch.topk(match_quality_matrix, self.top_k, dim=1, sorted=False)
            del match_quality_matrix
            cls_idx = labels[:, None].repeat(1, self.top_k)
            match_anchor_idx.append(matched.view(-1))
            match_cls_idx.append(cls_idx.view(-1))
            # ------------------ positive_loss end ------------------

        if cls_predicts.dtype == torch.float16:
            cls_predicts = cls_predicts.float()
        cls_predicts = cls_predicts.sigmoid()

        match_bidx = sum([[i] * len(item) for i, item in zip(batch_idx, match_anchor_idx)], [])
        match_anchor_idx = torch.cat(match_anchor_idx)
        match_cls_idx = torch.cat(match_cls_idx)
        cls_prob = cls_predicts[match_bidx, match_anchor_idx, match_cls_idx].view(-1, self.top_k)

        match_reg_predicts = reg_predicts[match_bidx, match_anchor_idx].view(-1, self.top_k, 4)
        match_anchors = all_anchors[match_anchor_idx].view(-1, self.top_k, 4)
        match_gt_box = torch.cat([gt_boxes[bid][:, 1:][:, None, :].repeat(1, self.top_k, 1)
                                  for bid in batch_idx],
                                 dim=0)
        match_reg_target = self.box_coder.encoder(match_anchors, match_gt_box)
        box_loss = smooth_l1_loss(match_reg_predicts, match_reg_target, beta=self.beta).sum(-1)
        box_prob = (-self.reg_weight * box_loss).exp()
        positive_losses = mean_max(cls_prob * box_prob).sum()

        all_cls_prob = cls_predicts[batch_idx].view(-1, cls_num)
        all_cls_target = torch.cat(batch_cls_target)
        negative_loss = negative_focal_loss(all_cls_prob * (1 - all_cls_target), self.gamma).sum()
        if self.dist_train:
            positive_losses = reduce_sum(positive_losses, clone=False)
            negative_loss = reduce_sum(negative_loss, clone=False)
            gt_num = reduce_sum(torch.tensor(gt_num, device=device)).item()
        positive_losses = positive_losses / gt_num
        negative_loss = negative_loss / (gt_num * self.top_k)
        # print(positive_losses * self.alpha, negative_loss * (1 - self.alpha))
        return positive_losses * self.alpha, negative_loss * (1 - self.alpha)
