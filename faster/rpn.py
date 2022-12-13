import torch.nn as nn
import torch
from torch import Tensor
import torch.functional as F
from torchvision.ops import nms
import numpy as np

from typing import Union

# 4 * 9 outs for each feature in resulting feature map
# 9 anchors: regular bboxes
anchors = torch.tensor([[-83.,  -39.,  100.,   56.],
                        [-175.,  -87.,  192.,  104.],
                        [-359., -183.,  376.,  200.],
                        [-55.,  -55.,   72.,   72.],
                        [-119., -119.,  136.,  136.],
                        [-247., -247.,  264.,  264.],
                        [-35.,  -79.,   52.,   96.],
                        [-79., -167.,   96.,  184.],
                        [-167., -343.,  184.,  360.]])
num_Anchors = anchors.size(0)  # 9

train_pre_nms_topN, test_pre_nms_topN = 12000, 6000
train_post_nms_topN, test_post_nms_topN = 2000, 300
nms_thresh = 0.7
train_min_size, test_min_size = 8, 16
feat_stride = 16


def bbox_transform_inv(anchors, delta_bboxes):
    '''
    anchors:[Batch_size, feat_h*feat_w*num_Anchors, 4]
    delta_bboxes:[Batch_size, Hc*Wc*9, 4]
    batch_size:[Batch_size]
    '''
    widths = anchors[:, :, 2] - anchors[:, :, 0] + 1.0
    heights = anchors[:, :, 3] - anchors[:, :, 1] + 1.0

    ctr_x = anchors[:, :, 0] + 0.5 * widths
    ctr_y = anchors[:, :, 1] + 0.5 * heights
    """    
    # each 4th element 
    # dx = delta_bboxes[:, :, 0::4] ?
    # dy = delta_bboxes[:, :, 1::4] ?
    # dw = delta_bboxes[:, :, 2::4] ?
    # dh = delta_bboxes[:, :, 3::4] ? 
    """
    dx = delta_bboxes[:, :, 0]
    dy = delta_bboxes[:, :, 1]
    dw = delta_bboxes[:, :, 2]
    dh = delta_bboxes[:, :, 3]
    """
    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2) ?
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2) ?
    pred_w = torch.exp(dw) * widths.unsqueeze(2) ?
    pred_h = torch.exp(dh) * heights.unsqueeze(2) ?
    """
    pred_ctr_x = dx * ctr_x
    pred_ctr_y = dy * ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights
    pred_boxes = delta_bboxes.clone()
    # x1
    pred_boxes[:, :, 0] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes


def clip_boxes(boxes, im_shape, batch_size):
    for i in range(batch_size):
        boxes[i, :, 0].clamp_(0, im_shape[i, 1]-1)
        boxes[i, :, 1].clamp_(0, im_shape[i, 0]-1)
        boxes[i, :, 2].clamp_(0, im_shape[i, 1]-1)
        boxes[i, :, 3].clamp_(0, im_shape[i, 0]-1)
    return boxes

# DOC STRINGS!!!!!!!!!!!!!!!!!!!!!!!!!!!! Может быть не ту половину выбрал в scores Contrastive learning


class RPN(nn.Module):
    """
        return  
        :out: Tensor[batch_size, post_nms_topN, 5]  
        :scores: Tensor[batch_size, 9*Hc*Wc]  
        :proposals: Tensor[Batch_size, Hc*Wc*9, 4]  
    """
    def __init__(self, in_ch,) -> None:
        super().__init__()
        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(in_ch, 512, 3, 1, 1, bias=True)
        # conv 1x1 -> like oun conv_net for each feature (in_ch * 4 * 9 * 1 * 1)
        # 4 coord X 9 anchors: 9*x + 9*y + 9*w + 9*h
        self.RPN_bbox_reg = nn.Conv2d(512, 4 * 9, kernel_size=1, stride=1)
        # bg(background)|fg(foreground) probs:
        # 2[prob(feature(ij)) | 1-prob(feature(ij))] * 9 anchors
        self.RPN_bbox_cls = nn.Conv2d(512, 2 * 9, kernel_size=1, stride=1)
        self.anchors = anchors

    def bbox_delta(self, x):  # out: [Batch_size, Hc*Wc*9, 4], Hc, Wc
        batch_size = x.size(0)
        delta_bboxes_reg = self.RPN_bbox_reg(
            x)  # size: [Batch_size, 36, Hc, Wc]
        Hc, Wc = delta_bboxes_reg.size(2), delta_bboxes_reg.size(
            3)  # feature size after convs
        delta_bboxes_reg = delta_bboxes_reg.permute(
            0, 2, 3, 1)  # size: [Batch_size, Hc, Wc, 36]
        # from [36] to [4] by 1 multicell
        # all the same include order, but in new dimentions
        delta_bboxes_reg = delta_bboxes_reg.view(batch_size, -1, 4)
        return delta_bboxes_reg, Hc, Wc

    def bbox_scores(self, x):
        batch_size = x.size(0)
        bbox_scores_ = F.relu(x, inplace=True)
        #size:[Batch_size, 18, Hc, Wc]
        bbox_scores_ = self.RPN_bbox_cls(bbox_scores_)
        # 2 foreground or background (Probs)
        #size:[Batch_size, 2, Hc*9, Wc]
        bbox_scores_ = self.reshape(bbox_scores_, 2)
        bbox_scores_ = F.softmax(bbox_scores_, 1)[:, 0]
        # only foreground
        bbox_scores_ = bbox_scores_.view(batch_size, -1)
        return bbox_scores_  # size:[Batch_size, Hc*Wc*9]

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            -1,  # (input_shape[1] * input_shape[2]) / d.
            input_shape[3]
        )
        return x

    def feature_shift(self, Wc, Hc):  # out:[Hc*Wc, 4]
        shift_x, shift_y = torch.arange(
            0, Wc) * feat_stride, torch.arange(0, Hc) * feat_stride
        # meshrgid (expand-like): from [1, N] and [1, K] -> [K, N] and [K, N] if "xy" or [N, K] if "ij" like:
        """        
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        Observe the element-wise pairings across the grid, (1, 4),
        (1, 5), ..., (3, 6). This is the same thing as the
        cartesian product.
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        grid_x:
        tensor([[1, 1, 1],
                [2, 2, 2],
                [3, 3, 3]])
        grid_y:
        tensor([[4, 5, 6],
                [4, 5, 6],
                [4, 5, 6]])
        """
        shift_x, shift_y = torch.meshgrid(shift_x, shift_y, indexing='xy')
        # flatten: from [N, K] to [1, N*K]
        shifts = torch.vstack([
            shift_x.flatten(),
            shift_y.flatten(),
            shift_x.flatten(),
            shift_y.flatten()]).T  # T: ij -> ji
        return shifts

    #[Batch_size, feat_HW * num_Anchors, 4]
    def anchor_trans(self, feat_HW, batch_size, shifts):
        '''        
        K = torch.arange(0, 6).view(1, 3, 2)
        A = torch.arange(0, 8).view(4, 1, 2)
        K :=  [[[0, 1],
                [2, 3],
                [4, 5]]],
        A :=   [[[0, 1]],

                [[2, 3]],

                [[4, 5]],

                [[6, 7]]]
        K + A :=  [[[ 0,  2], # | <= A[0] + K : [0, 1] + K
                    [ 2,  4],   |
                    [ 4,  6]],  |

                    [[ 2,  4],# | <= A[1] + K : [2, 3] + K
                    [ 4,  6],
                    [ 6,  8]],

                    [[ 4,  6],# | <= A[2] + K : [4, 5] + K
                    [ 6,  8],
                    [ 8, 10]],

                    [[ 6,  8],# | <= A[3] + K : [6, 7] + K
                    [ 8, 10],
                    [10, 12]]],
        (K + A).size() := torch.Size([4, 3, 2])
        '''
        # for i (feature) in shifts[i, 1, :4] apply(add) anchors[1, :9, :4]
        A = self.anchors.view(1, num_Anchors, 4) + shifts.view(feat_HW, 1, 4)
        A = A.view(1, feat_HW * num_Anchors, 4)
        # [1, Hc * Wc, num_Anchors]
        A = A.expand(batch_size, feat_HW * num_Anchors, 4)
        return A

    def forward(self, x, im_size) -> Union[Tensor, Tensor, Tensor]:
        """
        return
        :out: Tensor[batch_size, post_nms_topN, 4] 
        :scores: Tensor[batch_size, 9*Hc*Wc]
        :proposals: Tensor[Batch_size, Hc*Wc*9, 4]
        """
        rpn_conv1 = self.RPN_Conv(x)
        batch_size = rpn_conv1.size(0)

        delta_bboxes_reg, Hc, Wc = self.bbox_delta(
            rpn_conv1)  # size:[Batch_size, Hc*Wc*9, 4]
        #size:[Batch_size, 9*Hc*Wc]
        bbox_scores_ = self.bbox_scores(rpn_conv1)

        # final feature stride !in origin image!
        shifts = self.feature_shift(Wc, Hc)  # size:[Hc*Wc, 4]
        shifts = shifts.type_as(bbox_scores_)
        feat_HW = Hc * Wc
        #A:[Batch_size, feat_HW * num_Anchors, 4]
        Anchor = self.anchor_trans(feat_HW, batch_size, shifts)

        # Convert anchors into proposals via bbox transformations
        '''        
        A:[Batch_size, Hc*Wc*num_Anchors, 4]
        delta_bboxes_reg:[Batch_size, Hc*Wc*9, 4]
        batch_size:[Batch_size]
        '''
        proposals = bbox_transform_inv(
            Anchor, delta_bboxes_reg)  # [Batch_size, Hc*Wc*9, 4]
        # crop bbox out of image borders
        proposals = clip_boxes(proposals, im_size, batch_size)

        _, order = torch.sort(bbox_scores_, dim=1, descending=True)
        # to CFG
        if self.training:
            post_nms_topN = train_post_nms_topN
            pre_nms_topN = train_pre_nms_topN
        else:
            post_nms_topN = test_post_nms_topN
            pre_nms_topN = test_pre_nms_topN

        output = bbox_scores_.new_zeros(
            (batch_size, post_nms_topN, 4))  # same type/device

        # need fix to batched_nms!
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals[i]  # [Hc*Wc*9, 4]
            scores_single = bbox_scores_[i]  # [9*Hc*Wc]
            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]
            # numel: total number of elements
            if 0 < pre_nms_topN < bbox_scores_.numel():
                order_single = order_single[:pre_nms_topN]
            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            keep_index_i = nms(
                proposals_single, scores_single, nms_thresh)  # NMS
            keep_index_i = keep_index_i.long().view(-1)
            keep_index_i = keep_index_i[:post_nms_topN]
            proposals_single = proposals_single[keep_index_i, :]
            scores_single = scores_single[keep_index_i, :]
            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            output[i, :num_proposal] = proposals_single

        return output, bbox_scores_, proposals
