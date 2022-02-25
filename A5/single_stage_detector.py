from calendar import c
import io
from itertools import count
import time
import math
from matplotlib.transforms import Bbox
import torch 
import torch.nn as nn
from torch import Tensor, grid_sampler, optim
import torchvision
from a5_helper import *
import matplotlib.pyplot as plt


def hello_single_stage_detector():
    print("Hello from single_stage_detector.py!")


def GenerateAnchor(anc:torch.Tensor, grid:torch.Tensor):
  """
  Anchor generator.

  Inputs:
  - anc: Tensor of shape (A, 2) giving the shapes of anchor boxes to consider at
    each point in the grid. anc[a] = (w, h) gives the width and height of the
    a'th anchor shape.
  - grid: Tensor of shape (B, H', W', 2) giving the (x, y) coordinates of the
    center of each feature from the backbone feature map. This is the tensor
    returned from GenerateGrid.
  
  Outputs:
  - anchors: Tensor of shape (B, A, H', W', 4) giving the positions of all
    anchor boxes for the entire image. anchors[b, a, h, w] is an anchor box
    centered at grid[b, h, w], whose shape is given by anc[a]; we parameterize
    boxes as anchors[b, a, h, w] = (x_tl, y_tl, x_br, y_br), where (x_tl, y_tl)
    and (x_br, y_br) give the xy coordinates of the top-left and bottom-right
    corners of the box.
  """
  anchors = None
  ##############################################################################
  # TODO: Given a set of anchor shapes and a grid cell on the activation map,  #
  # generate all the anchor coordinates for each image. Support batch input.   #
  ##############################################################################
  # Replace "pass" statement with your code
  A, _ = anc.shape
  tl_offset = anc.reshape(1, A, 1, 1, 2) * (-0.5)
  br_offset = anc.reshape(1, A, 1, 1, 2) * (0.5)  
  offset = torch.cat((tl_offset, br_offset), dim=-1)  # (1, A, 1, 1, 4)
  anchors = torch.cat((grid.unsqueeze(1),grid.unsqueeze(1)), dim=-1) # (B, 1, H', W', 4)
  anchors = anchors + offset  # (B, A, H', W', 4)

  # print(anchors[0,0])
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return anchors


def GenerateProposal(anchors:torch.Tensor, offsets:torch.Tensor, method='YOLO'):
  """
  Proposal generator.

  Inputs:
  - anchors: Anchor boxes, of shape (B, A, H', W', 4). Anchors are represented
    by the coordinates of their top-left and bottom-right corners.
  - offsets: Transformations of shape (B, A, H', W', 4) that will be used to
    convert anchor boxes into region proposals. The transformation
    offsets[b, a, h, w] = (tx, ty, tw, th) will be applied to the anchor
    anchors[b, a, h, w]. For YOLO, assume that tx and ty are in the range
    (-0.5, 0.5).
  - method: Which transformation formula to use, either 'YOLO' or 'FasterRCNN'
  
  Outputs:
  - proposals: Region proposals of shape (B, A, H', W', 4), represented by the
    coordinates of their top-left and bottom-right corners. Applying the
    transform offsets[b, a, h, w] to the anchor [b, a, h, w] should give the
    proposal proposals[b, a, h, w].
  
  """
  assert(method in ['YOLO', 'FasterRCNN'])
  proposals = None
  ##############################################################################
  # TODO: Given anchor coordinates and the proposed offset for each anchor,    #
  # compute the proposal coordinates using the transformation formulas above.  #
  ##############################################################################
  # Replace "pass" statement with your code
  # (x1, y1, x2 ,y2) => (center_x, center_y, w, h)
  proposals = torch.zeros_like(anchors)
  tran_proposals = proposals.clone()
  tran_anchors = anchors.clone()
  tran_anchors[...,0:2] = (anchors[...,0:2] + anchors[...,2:4]) * 0.5 # center
  tran_anchors[...,2:4] = (anchors[...,2:4] - anchors[...,0:2]) # w and h

  if method == 'YOLO':
    tran_proposals[...,0:2] = tran_anchors[...,0:2] + offsets[...,0:2]
    tran_proposals[...,2:4] = tran_anchors[...,2:4] * torch.exp(offsets[...,2:4])
  elif method == 'FasterRCNN':
    tran_proposals[...,0:2] = tran_anchors[...,0:2] + offsets[...,0:2] * tran_anchors[...,2:4]
    tran_proposals[...,2:4] = tran_anchors[...,2:4] * torch.exp(offsets[...,2:4])

  # (center_x, center_y, w, h) => (x1, y1, x2 ,y2)
  proposals[...,0:2] = tran_proposals[...,0:2] - tran_proposals[...,2:4] * 0.5
  proposals[...,2:4] = tran_proposals[...,0:2] + tran_proposals[...,2:4] * 0.5

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return proposals


def IoU(proposals:torch.Tensor, bboxes:torch.Tensor):
  """
  Compute intersection over union between sets of bounding boxes.

  Inputs:
  - proposals: Proposals of shape (B, A, H', W', 4)
  - bboxes: Ground-truth boxes from the DataLoader of shape (B, N, 5).
    Each ground-truth box is represented as tuple (x_lr, y_lr, x_rb, y_rb, class).
    If image i has fewer than N boxes, then bboxes[i] will be padded with extra
    rows of -1.
  
  Outputs:
  - iou_mat: IoU matrix of shape (B, A*H'*W', N) where iou_mat[b, i, n] gives
    the IoU between one element of proposals[b] and bboxes[b, n].

  For this implementation you DO NOT need to filter invalid proposals or boxes;
  in particular you don't need any special handling for bboxxes that are padded
  with -1.
  """
  iou_mat = None
  ##############################################################################
  # TODO: Compute the Intersection over Union (IoU) on proposals and GT boxes. #
  # No need to filter invalid proposals/bboxes (i.e., allow region area <= 0). #
  # However, you need to make sure to compute the IoU correctly (it should be  #
  # 0 in those cases.                                                          # 
  # You need to ensure your implementation is efficient (no for loops).        #
  # HINT:                                                                      #
  # IoU = Area of Intersection / Area of Union, where                          #
  # Area of Union = Area of Proposal + Area of BBox - Area of Intersection     #
  # and the Area of Intersection can be computed using the top-left corner and #
  # bottom-right corner of proposal and bbox. Think about their relationships. #
  ##############################################################################
  # Replace "pass" statement with your code
  B, A, H, W, _ = proposals.shape
  proposals = proposals.reshape(B, A*H*W, -1).unsqueeze(dim=2) # (B, A*H*W, 1, 4)
  bboxes = bboxes.unsqueeze(dim=1) # (B, 1, N, 5)
  tl = torch.maximum(proposals[...,:2], bboxes[...,:2]) # (B, A*H*W, N, 2)
  br = torch.minimum(proposals[...,2:4], bboxes[...,2:4]) # (B, A*H*W, N, 2)

  intersect = torch.prod(br - tl, dim=3)  # (B, A*H*W, N)
  invalid_place = (tl > br).any(dim=3)
  intersect.masked_fill_(invalid_place, 0)
  p_area = torch.prod(proposals[...,2:4] - proposals[...,:2],dim=3) # (B, A*H*W, 1)
  b_area = torch.prod(bboxes[...,2:4]-bboxes[...,:2], dim=3)  # (B, 1, N)

  iou_mat = torch.div(intersect, p_area + b_area - intersect)
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return iou_mat


class PredictionNetwork(nn.Module):
  def __init__(self, in_dim, hidden_dim=128, num_anchors=9, num_classes=20, drop_ratio=0.3):
    super().__init__()

    assert(num_classes != 0 and num_anchors != 0)
    self.num_classes = num_classes
    self.num_anchors = num_anchors

    ##############################################################################
    # TODO: Set up a network that will predict outputs for all anchors. This     #
    # network should have a 1x1 convolution with hidden_dim filters, followed    #
    # by a Dropout layer with p=drop_ratio, a Leaky ReLU nonlinearity, and       #
    # finally another 1x1 convolution layer to predict all outputs. You can      #
    # use an nn.Sequential for this network, and store it in a member variable.  #
    # HINT: The output should be of shape (B, 5*A+C, 7, 7), where                #
    # A=self.num_anchors and C=self.num_classes.                                 #
    ##############################################################################
    # Make sure to name your prediction network pred_layer.
    self.pred_layer = None
    # Replace "pass" statement with your code
    self.pred_layer = nn.Sequential(
      nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=(1,1)),
      nn.Dropout(p=drop_ratio),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=hidden_dim, out_channels=5*self.num_anchors+self.num_classes, kernel_size=1)
      
    )
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

  def _extract_anchor_data(self, anchor_data, anchor_idx):
    """
    Inputs:
    - anchor_data: Tensor of shape (B, A, D, H, W) giving a vector of length
      D for each of A anchors at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving anchor indices to extract

    Returns:
    - extracted_anchors: Tensor of shape (M, D) giving anchor data for each
      of the anchors specified by anchor_idx.
    """
    B, A, D, H, W = anchor_data.shape
    anchor_data = anchor_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
    extracted_anchors = anchor_data[anchor_idx]
    return extracted_anchors
  
  def _extract_class_scores(self, all_scores, anchor_idx):
    """
    Inputs:
    - all_scores: Tensor of shape (B, C, H, W) giving classification scores for
      C classes at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors at
      which to extract classification scores

    Returns:
    - extracted_scores: Tensor of shape (M, C) giving the classification scores
      for each of the anchors specified by anchor_idx.
    """
    B, C, H, W = all_scores.shape
    A = self.num_anchors
    all_scores = all_scores.contiguous().permute(0, 2, 3, 1).contiguous()
    all_scores = all_scores.view(B, 1, H, W, C).expand(B, A, H, W, C)
    all_scores = all_scores.reshape(B * A * H * W, C)
    extracted_scores = all_scores[anchor_idx]
    return extracted_scores

  def forward(self, features, pos_anchor_idx=None, neg_anchor_idx=None):
    """
    Run the forward pass of the network to predict outputs given features
    from the backbone network.

    Inputs:
    - features: Tensor of shape (B, in_dim, 7, 7) giving image features computed
      by the backbone network.
    - pos_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as positive. These are only given during training; at test-time
      this should be None.
    - neg_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as negative. These are only given at training; at test-time this
      should be None.
    
    The outputs from this method are different during training and inference.
    
    During training, pos_anchor_idx and neg_anchor_idx are given and identify
    which anchors should be positive and negative, and this forward pass needs
    to extract only the predictions for the positive and negative anchors.

    During inference, only features are provided and this method needs to return
    predictions for all anchors.

    Outputs (During training):
    - conf_scores: Tensor of shape (2*M, 1) giving the predicted classification
      scores for positive anchors and negative anchors (in that order).
    - offsets: Tensor of shape (M, 4) giving predicted transformation for
      positive anchors.
    - class_scores: Tensor of shape (M, C) giving classification scores for
      positive anchors.

    Outputs (During inference):
    - conf_scores: Tensor of shape (B, A, H, W) giving predicted classification
      scores for all anchors.
    - offsets: Tensor of shape (B, A, 4, H, W) giving predicted transformations
      all all anchors.
    - class_scores: Tensor of shape (B, C, H, W) giving classification scores for
      each spatial position.
    """
    conf_scores, offsets, class_scores = None, None, None
    ############################################################################
    # TODO: Use backbone features to predict conf_scores, offsets, and         #
    # class_scores. Make sure conf_scores is between 0 and 1 by squashing the  #
    # network output with a sigmoid. Also make sure the first two elements t^x #
    # and t^y of offsets are between -0.5 and 0.5 by squashing with a sigmoid  #
    # and subtracting 0.5.                                                     #
    #                                                                          #
    # During training you need to extract the outputs for only the positive    #
    # and negative anchors as specified above.                                 #
    #                                                                          #
    # HINT: You can use the provided helper methods self._extract_anchor_data  #
    # and self._extract_class_scores to extract information for positive and   #
    # negative anchors specified by pos_anchor_idx and neg_anchor_idx.         #
    ############################################################################
    # Replace "pass" statement with your code
    assert pos_anchor_idx is None or pos_anchor_idx.shape[0] == neg_anchor_idx.shape[0], "pos anchor and neg anchor don't have the same shape"
    pred = self.pred_layer(features) # (B, 5*A+C, 7, 7)
    A = self.num_anchors
    C = self.num_classes
    B, _, H, W = pred.shape
    pred = pred.permute(0, 2, 3, 1) # (B, H, W, 5*A+C)
    conf_scores = pred[..., 0:5*A:5] # (B, H, W, A)
    conf_scores = torch.sigmoid(conf_scores) # (B, H, W, A)
    conf_scores = conf_scores.permute(0, 3, 1, 2) # (B, A, H, W)
    offsets = pred[..., :5*A].view(B, H, W, A, 5)[:, :, :, :, 1:] # (B, H, W, A, 4)
    offsets[:,:,:,:,:2] = torch.sigmoid(offsets[:,:,:,:, :2]) - torch.ones_like(offsets[:,:,:,:,:2]) * 0.5 # (B, H, W, A, 4)
    offsets = offsets.permute(0, 3, 4, 1, 2) # (B, A, 4, H, W)
    class_scores = pred[..., 5*A:].permute(0, 3, 1, 2) # (B, C, H, W)

    # training
    if pos_anchor_idx is not None:
      class_scores = self._extract_class_scores(class_scores, pos_anchor_idx)
      conf_scores = conf_scores.contiguous().view(-1)
      conf_scores = conf_scores[torch.cat((pos_anchor_idx, neg_anchor_idx))].reshape(-1, 1)
      offsets = self._extract_anchor_data(offsets, pos_anchor_idx)
    # None means inference
    

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return conf_scores, offsets, class_scores


class SingleStageDetector(nn.Module):
  def __init__(self):
    super().__init__()

    self.anchor_list = torch.tensor([[1., 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]]) # READ ONLY
    self.feat_extractor = FeatureExtractor()
    self.num_classes = 20
    self.pred_network = PredictionNetwork(1280, num_anchors=self.anchor_list.shape[0], \
                                          num_classes=self.num_classes)
  def forward(self, images, bboxes):
    """
    Training-time forward pass for the single-stage detector.

    Inputs:
    - images: Input images, of shape (B, 3, 224, 224)
    - bboxes: GT bounding boxes of shape (B, N, 5) (padded)

    Outputs:
    - total_loss: Torch scalar giving the total loss for the batch.
    """
    # weights to multiple to each loss term
    w_conf = 1 # for conf_scores
    w_reg = 1 # for offsets
    w_cls = 1 # for class_prob

    total_loss = None
    ##############################################################################
    # TODO: Implement the forward pass of SingleStageDetector.                   #
    # A few key steps are outlined as follows:                                   #
    # i) Image feature extraction,                                               #
    # ii) Grid and anchor generation,                                            #
    # iii) Compute IoU between anchors and GT boxes and then determine activated/#
    #      negative anchors, and GT_conf_scores, GT_offsets, GT_class,           #
    # iv) Compute conf_scores, offsets, class_prob through the prediction network#
    # v) Compute the total_loss which is formulated as:                          #
    #    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss,  #
    #    where conf_loss is determined by ConfScoreRegression, w_reg by          #
    #    BboxRegression, and w_cls by ObjectClassification.                      #
    # HINT: Set `neg_thresh=0.2` in ReferenceOnActivatedAnchors in this notebook #
    #       (A5-1) for a better performance than with the default value.         #
    ##############################################################################
    # Replace "pass" statement with your code
    feature = self.feat_extractor(images)
    grid = GenerateGrid(images.shape[0])
    anchors = GenerateAnchor(self.anchor_list.to(grid.device), grid)
    IoU_mat = IoU(anchors, bboxes)
    activated_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class, \
    activated_anc_coord, negative_anc_coord = ReferenceOnActivatedAnchors(
                                              anchors=anchors,
                                              bboxes=bboxes,
                                              grid=grid,
                                              iou_mat=IoU_mat,
                                              pos_thresh=0.7,
                                              neg_thresh=0.2,
                                              method='YOLO'
                                              )
    conf_scores, offsets, class_scores = self.pred_network(feature, activated_anc_ind, negative_anc_ind)
    conf_loss = ConfScoreRegression(conf_scores, GT_conf_scores)
    reg_loss = BboxRegression(offsets, GT_offsets)
    B, A, H_prime, W_prime, _ = anchors.shape
    anc_per_img = A*H_prime*W_prime
    cls_loss = ObjectClassification(class_prob=class_scores, 
                                    GT_class=GT_class,
                                    batch_size=images.shape[0], 
                                    anc_per_img=anc_per_img, 
                                    activated_anc_ind=activated_anc_ind
                                    )
    # print("conf loss  ", conf_loss)
    # print("reg loss  ", reg_loss)
    # print("cls_loss  ", cls_loss)

    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return total_loss
  
  def inference(self, images, thresh=0.5, nms_thresh=0.7):
    """"
    Inference-time forward pass for the single stage detector.

    Inputs:
    - images: Input images
    - thresh: Threshold value on confidence scores
    - nms_thresh: Threshold value on NMS

    Outputs:
    - final_propsals: Keeped proposals after confidence score thresholding and NMS,
                      a list of B (*x4) tensors
    - final_conf_scores: Corresponding confidence scores, a list of B (*x1) tensors
    - final_class: Corresponding class predictions, a list of B  (*x1) tensors
    """
    final_proposals, final_conf_scores, final_class = [], [], []
    ##############################################################################
    # TODO: Predicting the final proposal coordinates `final_proposals`,         #
    # confidence scores `final_conf_scores`, and the class index `final_class`.  #
    # The overall steps are similar to the forward pass but now you do not need  #
    # to decide the activated nor negative anchors.                              #
    # HINT: Thresholding the conf_scores based on the threshold value `thresh`.  #
    # Then, apply NMS (torchvision.ops.nms) to the filtered proposals given the  #
    # threshold `nms_thresh`.                                                    #
    # The class index is determined by the class with the maximal probability.   #
    # Note that `final_propsals`, `final_conf_scores`, and `final_class` are all #
    # lists of B 2-D tensors (you may need to unsqueeze dim=1 for the last two). #
    ##############################################################################
    # Replace "pass" statement with your code
    with torch.no_grad():
      feature = self.feat_extractor(images)
 
      # conf_scores (B, A, H, W)
      # offsets (B, A, 4, H, W)
      # class_scores (B, C, H, W)
      conf_scores, offsets, class_scores = self.pred_network(feature)
      B = images.shape[0]
      grid = GenerateGrid(B)
      anchors = GenerateAnchor(self.anchor_list.to(grid), grid)
      proposals = GenerateProposal(anchors=anchors, offsets=offsets.permute(0,1,3,4,2).contiguous(), method='YOLO') # (B, A, H, W, 4)
      proposals = proposals.reshape(B, -1, 4) # (B, A*H*W, 4)
      conf_scores = conf_scores.contiguous().view(B, -1) # (B, A*H*W)

      class_scores = class_scores.permute(0, 2, 3, 1).unsqueeze(1) # (B, 1, H, W, C)
      class_scores = class_scores.expand(-1, conf_scores.shape[1], -1, -1, -1) # (B, A, H, W, C)
      class_scores = torch.argmax(class_scores, dim=-1) # (B, A, H, W)
      class_scores = class_scores.contiguous().view(B, -1) # (B, A*H*W)
      for i in range(B):
        cur_proposals = proposals[i] # (A*H*W, 4)
        cur_conf_scores = conf_scores[i] # (A*H*W, )
        to_choose = torch.nonzero(cur_conf_scores > thresh).squeeze(1)
        cur_proposals = cur_proposals[to_choose]
        cur_conf_scores = cur_conf_scores[to_choose]
        cur_class_scores = class_scores[i]
        cur_class_scores = cur_class_scores[to_choose]
        remain = torchvision.ops.nms( boxes=cur_proposals.contiguous().view(-1, 4),
                                      scores=cur_conf_scores.contiguous().view(-1),
                                      iou_threshold=nms_thresh
                                      )
        final_proposals.append(cur_proposals.contiguous().view(-1, 4)[remain])         
        final_conf_scores.append(cur_conf_scores.contiguous().view(-1)[remain].unsqueeze(1))
        final_class.append(cur_class_scores.contiguous().view(-1)[remain].unsqueeze(1))
      


    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return final_proposals, final_conf_scores, final_class


def nms(boxes:torch.Tensor, scores:torch.Tensor, iou_threshold=0.5, topk=None):
  """
  Non-maximum suppression removes overlapping bounding boxes.

  Inputs:
  - boxes: top-left and bottom-right coordinate values of the bounding boxes
    to perform NMS on, of shape Nx4
  - scores: scores for each one of the boxes, of shape N
  - iou_threshold: discards all overlapping boxes with IoU > iou_threshold; float
  - topk: If this is not None, then return only the topk highest-scoring boxes.
    Otherwise if this is None, then return all boxes that pass NMS.

  Outputs:
  - keep: torch.long tensor with the indices of the elements that have been
    kept by NMS, sorted in decreasing order of scores; of shape [num_kept_boxes]
  """
  # numel() will return the total number of elements
  if (not boxes.numel()) or (not scores.numel()):
    return torch.zeros(0, dtype=torch.long)

  keep = None
  #############################################################################
  # TODO: Implement non-maximum suppression which iterates the following:     #
  #       1. Select the highest-scoring box among the remaining ones,         #
  #          which has not been chosen in this step before                    #
  #       2. Eliminate boxes with IoU > threshold                             #
  #       3. If any boxes remain, GOTO 1                                      #
  #       Your implementation should not depend on a specific device type;    #
  #       you can use the device of the input if necessary.                   #
  # HINT: You can refer to the torchvision library code:                      #
  #   github.com/pytorch/vision/blob/master/torchvision/csrc/cpu/nms_cpu.cpp  #
  #############################################################################
  # Replace "pass" statement with your code

  # cacuclate the IoU matrix recording the IoU for each box
  import datetime
  # print(datetime.datetime.now().strftime('%X'))
  row_boxes = boxes.unsqueeze(dim=1) # (N, 1, 4)
  col_boxes = boxes.unsqueeze(dim=0) # (1, N, 4)
  tl = torch.maximum(row_boxes[:,:,:2], col_boxes[:,:,:2]) # (N,N,2)
  br = torch.minimum(row_boxes[:,:,2:], col_boxes[:,:,2:]) # (N,N,2)

  intersect = br - tl
  invalid_mask = torch.any(intersect<0, dim=-1) # (N, N)
  intersect = torch.prod(intersect, dim=-1) # (N, N)
  a = row_boxes[:, :, 2:] - row_boxes[:, :, :2]
  a = torch.prod(a, dim=-1) # (N, 1)
  b = col_boxes[:, :, 2:] - col_boxes[:, :, :2]
  b = torch.prod(b, dim=-1) # (1, N)
  iou_mat = torch.div(intersect, a+b-intersect) # (N, N)
  iou_mat[invalid_mask] = 0 
  # print(datetime.datetime.now().strftime('%X'))
  for i in range(iou_mat.shape[0]):
    iou_mat[i, i] = 0 # make the elements in the diagonal be zero

  values, indexes = torch.sort(scores, descending=True)
  removed = set()
  keep = []

  for idx in indexes:
    if idx in removed:
      continue
    removed.add(idx)
    keep.append(idx)
    if topk is not None and len(keep) >= topk:
      break
    to_remove = iou_mat[idx] > iou_threshold
    to_remove = torch.nonzero(to_remove.view(-1)).squeeze(-1)
    # print(to_remove)
    iou_mat[:, to_remove] = 0
    removed = removed.union(set(to_remove.tolist()))

  keep = torch.tensor(keep).to(device=boxes.device, dtype=torch.long)
  # print(datetime.datetime.now().strftime('%X'))
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  return keep

def ConfScoreRegression(conf_scores, GT_conf_scores):
  """
  Use sum-squared error as in YOLO

  Inputs:
  - conf_scores: Predicted confidence scores
  - GT_conf_scores: GT confidence scores
  
  Outputs:
  - conf_score_loss
  """
  # the target conf_scores for negative samples are zeros
  GT_conf_scores = torch.cat((torch.ones_like(GT_conf_scores), \
                              torch.zeros_like(GT_conf_scores)), dim=0).view(-1, 1)
  conf_score_loss = torch.sum((conf_scores - GT_conf_scores)**2) * 1. / GT_conf_scores.shape[0]
  return conf_score_loss


def BboxRegression(offsets, GT_offsets):
  """"
  Use sum-squared error as in YOLO
  For both xy and wh

  Inputs:
  - offsets: Predicted box offsets
  - GT_offsets: GT box offsets
  
  Outputs:
  - bbox_reg_loss
  """
  bbox_reg_loss = torch.sum((offsets - GT_offsets)**2) * 1. / GT_offsets.shape[0]
  return bbox_reg_loss

