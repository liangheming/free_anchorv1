import torch
from utils.retinanet import BoxCoder
from commons.boxs_utils import box_iou
from losses.commons import smooth_l1_loss


def mean_max(x):
    """
    :param x: [gt, tok_anchor]
    :return:
    """
    weights = 1 / ((1 - x).clamp(min=1e-10))
    weights /= weights.sum(-1)[:, None]
    bag_prob = (weights * x).sum(-1)

    return -bag_prob.clamp(min=1e-10, max=1 - 1e-10).log()


class RetinaAnchorFreeLoss(object):
    def __init__(self, gamma=2.0, alpha=0.25, top_k=64, box_iou_thresh=0.6, box_reg_weight=0.75, beta=1. / 9):
        super(RetinaAnchorFreeLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.top_k = top_k
        self.box_iou_thresh = box_iou_thresh
        self.box_reg_weight = box_reg_weight
        self.beta = beta
        self.box_coder = BoxCoder()

    def __call__(self, cls_predicts, box_predicts, anchors, targets):
        """
        :param cls_predicts:
        :param box_predicts:
        :param anchors:
        :param targets:
        :return:
        """
        device = cls_predicts[0].device
        bs = cls_predicts[0].shape[0]
        cls_num = cls_predicts[0].shape[-1]
        expand_anchor = torch.cat(anchors, dim=0)

        negative_loss_list = list()
        positive_loss_list = list()

        for bi in range(bs):
            batch_cls_predicts = torch.cat([cls_item[bi] for cls_item in cls_predicts], dim=0) \
                .sigmoid() \
                .clamp(min=1e-6, max=1 - 1e-6)
            batch_targets = targets[targets[:, 0] == bi, 1:]

            if len(batch_targets) == 0:
                negative_loss = -(1 - self.alpha) * (batch_cls_predicts ** self.gamma) * (1 - batch_cls_predicts).log()
                negative_loss_list.append(negative_loss.sum())
                continue

            batch_box_predicts = torch.cat([box_item[bi] for box_item in box_predicts], dim=0)
            # calc_positive_loss
            targets_anchor_iou = box_iou(batch_targets[:, 2:], expand_anchor)
            _, top_k_anchor_idx = targets_anchor_iou.topk(k=self.top_k, dim=1, sorted=False)

            matched_cls_prob = batch_cls_predicts[top_k_anchor_idx].gather(dim=-1, index=(
                batch_targets[:, [1]][:, None, :]).long().repeat(1, self.top_k, 1)).squeeze(-1)
            match_box_target = self.box_coder.encoder(expand_anchor[top_k_anchor_idx], batch_targets[:, None, 2:])
            matched_box_prob = (
                    -self.box_reg_weight * smooth_l1_loss(batch_box_predicts[top_k_anchor_idx], match_box_target,
                                                          self.beta).sum(-1)).exp()
            positive_loss = self.alpha * mean_max(matched_cls_prob * matched_box_prob).sum()
            positive_loss_list.append(positive_loss)

            with torch.no_grad():
                box_localization = self.box_coder.decoder(batch_box_predicts, expand_anchor)
            target_box_iou = box_iou(batch_targets[:, 2:], box_localization)
            t1 = self.box_iou_thresh
            t2 = target_box_iou.max(dim=1, keepdim=True)[0].clamp(min=t1 + 1e-6)
            target_box_prob = ((target_box_iou - t1) / (t2 - t1)).clamp(min=0., max=1.)
            indices = torch.stack([torch.arange(len(batch_targets), device=device), batch_targets[:, 1]], dim=0).long()
            object_cls_box_prob = torch.sparse_coo_tensor(indices, target_box_prob,device=device)

            cls_idx, anchor_idx = torch.sparse.sum(object_cls_box_prob, dim=0).to_dense().nonzero(as_tuple=False).t()
            if len(cls_idx) == 0:
                negative_loss = -(1 - self.alpha) * (batch_cls_predicts ** self.gamma) * (1 - batch_cls_predicts).log()
                negative_loss_list.append(negative_loss.sum())
                continue
            anchor_positive_max_prob = torch.where(
                batch_targets[:, [1]].long() == cls_idx,
                target_box_prob[:, anchor_idx],
                torch.tensor(data=0., device=device)
            ).max(dim=0)[0]

            anchor_cls_assign_prob = torch.zeros(size=(len(expand_anchor), cls_num), device=device)
            anchor_cls_assign_prob[anchor_idx, cls_idx] = anchor_positive_max_prob
            negative_prob = batch_cls_predicts * (1 - anchor_cls_assign_prob)
            negative_loss = -(1 - self.alpha) * (negative_prob ** self.gamma) * (1 - negative_prob).log()
            negative_loss_list.append(negative_loss.sum())

        negative_losses = torch.stack(negative_loss_list).sum() / max(1, len(targets))
        if len(positive_loss_list) == 0:
            total_loss = negative_losses
            return total_loss, torch.stack([negative_losses, torch.tensor(data=0., device=device)]), len(targets)

        positive_losses = torch.stack(positive_loss_list).sum() / max(1, len(targets))
        total_loss = negative_losses + positive_losses
        return total_loss, torch.stack([negative_losses, positive_losses]), len(targets)
