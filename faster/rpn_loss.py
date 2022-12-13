import torch.functional as F 
import torch.nn as nn
import torch

from backbone import VGG_conv
from rpn import RPN


class SimpleDet(nn.Module):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride) -> None:
        super().__init__()
        self._feat_stride = feat_stride
        self.VGG = VGG_conv(3)
        self.RPN = RPN(256)

    def forward(self, image, gt_boxes):
        base_feat = self.VGG(image)
        im_size = image.shape[2], image.shape[3]
        #--------NMS использовать только на инференсе ?
        output, bbox_scores_, proposals = self.RPN(base_feat, im_size)
        cls_overlaps = bbox_overlaps_batch(proposals, gt_boxes)
        

        

        

def find_tp_fn():
    pass

def bbox_overlaps_batch(proposals, gt_boxes):
    """
    proposals: (b, N, 4) ndarray of float 
    gt_boxes: (b, K, 5) ndarray of float
    overlaps: (b, N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)
    N = proposals.size(1)
    K = proposals.size(1)

    gt_boxes = gt_boxes[:, :, :4]

    gt_boxes_x = gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1
    gt_boxes_y = gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1
    gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

    proposals_boxes_x = (proposals[:,:,2] - proposals[:,:,0] + 1)
    proposals_boxes_y = (proposals[:,:,3] - proposals[:,:,1] + 1)
    proposals_area = (proposals_boxes_x * proposals_boxes_y).view(batch_size, N, 1)

    gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
    proposals_area_zero = (proposals_boxes_x == 1) & (proposals_boxes_y == 1)

    #[N, K, 4]
    boxes = proposals.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
    query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

    #max(x1) and min(x2)
    iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
    iw[iw < 0] = 0

    #max(y1) and min(y2)
    ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = proposals_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    # mask the overlap here.
    overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
    overlaps.masked_fill_(proposals_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
    return overlaps
    