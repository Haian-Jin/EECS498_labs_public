from pyexpat import features
import time
import math
from tkinter.tix import Tree
from turtle import right
import torch 
import torch.nn as nn
from torch import optim
import torchvision
from a5_helper import *
import matplotlib.pyplot as plt
from single_stage_detector import GenerateAnchor, GenerateProposal, IoU


def hello_two_stage_detector():
    print("Hello from two_stage_detector.py!")

class ProposalModule(nn.Module):
  def __init__(self, in_dim, hidden_dim=256, num_anchors=9, drop_ratio=0.3):
    super().__init__()

    assert(num_anchors != 0)
    self.num_anchors = num_anchors
    ##############################################################################
    # TODO: Define the region proposal layer - a sequential module with a 3x3    #
    # conv layer, followed by a Dropout (p=drop_ratio), a Leaky ReLU and         #
    # a 1x1 conv.                                                                #
    # HINT: The output should be of shape Bx(Ax6)x7x7, where A=self.num_anchors. #
    #       Determine the padding of the 3x3 conv layer given the output dim.    #
    ##############################################################################
    # Make sure that your region proposal module is called pred_layer
    self.pred_layer = None      
    # Replace "pass" statement with your code
    self.pred_layer = nn.Sequential(
                      nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
                      nn.Dropout2d(p=drop_ratio),
                      nn.LeakyReLU(),
                      nn.Conv2d(in_channels=hidden_dim, out_channels=6*num_anchors, kernel_size=1)
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

  def forward(self, features, pos_anchor_coord=None, \
              pos_anchor_idx=None, neg_anchor_idx=None):
    """
    Run the forward pass of the proposal module.

    Inputs:
    - features: Tensor of shape (B, in_dim, H', W') giving features from the
      backbone network.
    - pos_anchor_coord: Tensor of shape (M, 4) giving the coordinates of
      positive anchors. Anchors are specified as (x_tl, y_tl, x_br, y_br) with
      the coordinates of the top-left corner (x_tl, y_tl) and bottom-right
      corner (x_br, y_br). During inference this is None.
    - pos_anchor_idx: int64 Tensor of shape (M,) giving the indices of positive
      anchors. During inference this is None.
    - neg_anchor_idx: int64 Tensor of shape (M,) giving the indicdes of negative
      anchors. During inference this is None.

    The outputs from this module are different during training and inference.
    
    During training, pos_anchor_coord, pos_anchor_idx, and neg_anchor_idx are
    all provided, and we only output predictions for the positive and negative
    anchors. During inference, these are all None and we must output predictions
    for all anchors.

    Outputs (during training):
    - conf_scores: Tensor of shape (2M, 2) giving the classification scores
      (object vs background) for each of the M positive and M negative anchors.
    - offsets: Tensor of shape (M, 4) giving predicted transforms for the
      M positive anchors.
    - proposals: Tensor of shape (M, 4) giving predicted region proposals for
      the M positive anchors.

    Outputs (during inference):
    - conf_scores: Tensor of shape (B, A, 2, H', W') giving the predicted
      classification scores (object vs background) for all anchors
    - offsets: Tensor of shape (B, A, 4, H', W') giving the predicted transforms
      for all anchors
    """
    if pos_anchor_coord is None or pos_anchor_idx is None or neg_anchor_idx is None:
      mode = 'eval'
    else:
      mode = 'train'
    conf_scores, offsets, proposals = None, None, None
    ############################################################################
    # TODO: Predict classification scores (object vs background) and transforms#
    # for all anchors. During inference, simply output predictions for all     #
    # anchors. During training, extract the predictions for only the positive  #
    # and negative anchors as described above, and also apply the transforms to#
    # the positive anchors to compute the coordinates of the region proposals. #
    #                                                                          #
    # HINT: You can extract information about specific proposals using the     #
    # provided helper function self._extract_anchor_data.                      #
    # HINT: You can compute proposal coordinates using the GenerateProposal    #
    # function from the previous notebook.                                     #
    ############################################################################
    # Replace "pass" statement with your code
    B, _ ,H, W = features.shape
    A = self.num_anchors
    predicted_result = self.pred_layer(features) # Bx(Ax6)x7x7
    predicted_result = predicted_result.contiguous().view(B, A, 6, H, W)
    conf_scores = predicted_result[:,:,:2,:,:] # (B, A, 2, H, W)
    offsets = predicted_result[:,:,2:,:,:] # (B, A, 4, H, W)
    # print(pos_anchor_idx)
    if mode == 'train':
      conf_scores = self._extract_anchor_data(conf_scores, torch.cat((pos_anchor_idx, neg_anchor_idx))) # (2*M, 2)
      offsets = self._extract_anchor_data(offsets, pos_anchor_idx) # (M, 4)
      M = pos_anchor_coord.shape[0]
      proposals = GenerateProposal(
                                    anchors=pos_anchor_coord.contiguous().view(M,1,1,1,4),
                                    offsets=offsets.contiguous().view(M,1,1,1,4),
                                    method='FasterRCNN'
                                  ).squeeze() # (M, 4)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    if mode == 'train':
      return conf_scores, offsets, proposals
    elif mode == 'eval':
      return conf_scores, offsets


def ConfScoreRegression(conf_scores, batch_size):
  """
  Binary cross-entropy loss

  Inputs:
  - conf_scores: Predicted confidence scores, of shape (2M, 2). Assume that the
    first M are positive samples, and the last M are negative samples.

  Outputs:
  - conf_score_loss: Torch scalar
  """
  # the target conf_scores for positive samples are ones and negative are zeros
  M = conf_scores.shape[0] // 2
  GT_conf_scores = torch.zeros_like(conf_scores)
  GT_conf_scores[:M, 0] = 1. # M * (1, 0)
  GT_conf_scores[M:, 1] = 1. # M * (0, 1)

  conf_score_loss = nn.functional.binary_cross_entropy_with_logits(conf_scores, GT_conf_scores, \
                                     reduction='sum') * 1. / batch_size
  return conf_score_loss


def BboxRegression(offsets, GT_offsets, batch_size):
  """"
  Use SmoothL1 loss as in Faster R-CNN

  Inputs:
  - offsets: Predicted box offsets, of shape (M, 4)
  - GT_offsets: GT box offsets, of shape (M, 4)
  
  Outputs:
  - bbox_reg_loss: Torch scalar
  """
  bbox_reg_loss = nn.functional.smooth_l1_loss(offsets, GT_offsets, reduction='sum') * 1. / batch_size
  return bbox_reg_loss


class RPN(nn.Module):
  def __init__(self):
    super().__init__()

    # READ ONLY
    self.anchor_list = torch.tensor([[1., 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]])
    self.feat_extractor = FeatureExtractor()
    self.prop_module = ProposalModule(1280, num_anchors=self.anchor_list.shape[0])

  def forward(self, images, bboxes, output_mode='loss'):
    """
    Training-time forward pass for the Region Proposal Network.

    Inputs:
    - images: Tensor of shape (B, 3, 224, 224) giving input images
    - bboxes: Tensor of ground-truth bounding boxes, returned from the DataLoader
    - output_mode: One of 'loss' or 'all' that determines what is returned:
      If output_mode is 'loss' then the output is:
      - total_loss: Torch scalar giving the total RPN loss for the minibatch
      If output_mode is 'all' then the output is:
      - total_loss: Torch scalar giving the total RPN loss for the minibatch
      - pos_conf_scores: Tensor of shape (M, 1) giving the object classification
        scores (object vs background) for the positive anchors
      - proposals: Tensor of shape (M, 4) giving the coordiantes of the region
        proposals for the positive anchors
      - features: Tensor of features computed from the backbone network
      - GT_class: Tensor of shape (M,) giving the ground-truth category label
        for the positive anchors.
      - pos_anchor_idx: Tensor of shape (M,) giving indices of positive anchors
      - neg_anchor_idx: Tensor of shape (M,) giving indices of negative anchors
      - anc_per_image: Torch scalar giving the number of anchors per image.
    
    Outputs: See output_mode

    HINT: The function ReferenceOnActivatedAnchors from the previous notebook
    can compute many of these outputs -- you should study it in detail:
    - pos_anchor_idx (also called activated_anc_ind)
    - neg_anchor_idx (also called negative_anc_ind)
    - GT_class
    """
    # weights to multiply to each loss term
    w_conf = 1 # for conf_scores
    w_reg = 5 # for offsets

    assert output_mode in ('loss', 'all'), 'invalid output mode!'
    total_loss = None
    conf_scores, proposals, features, GT_class, pos_anchor_idx, anc_per_img = \
      None, None, None, None, None, None
    ##############################################################################
    # TODO: Implement the forward pass of RPN.                                   #
    # A few key steps are outlined as follows:                                   #
    # i) Image feature extraction,                                               #
    # ii) Grid and anchor generation,                                            #
    # iii) Compute IoU between anchors and GT boxes and then determine activated/#
    #      negative anchors, and GT_conf_scores, GT_offsets, GT_class,           #
    # iv) Compute conf_scores, offsets, proposals through the region proposal    #
    #     module                                                                 #
    # v) Compute the total_loss for RPN which is formulated as:                  #
    #    total_loss = w_conf * conf_loss + w_reg * reg_loss,                     #
    #    where conf_loss is determined by ConfScoreRegression, w_reg by          #
    #    BboxRegression. Note that RPN does not predict any class info.          #
    #    We have written this part for you which you've already practiced earlier#
    # HINT: Do not apply thresholding nor NMS on the proposals during training   #
    #       as positive/negative anchors have been explicitly targeted.          #
    ##############################################################################
    # Replace "pass" statement with your code
    B = images.shape[0]
    assert bboxes.shape[0] == B, "bboxes don't have correct batch size"
    features = self.feat_extractor(images)
    grids = GenerateGrid(batch_size=B) # (B, H, W, 2)
    anchors = GenerateAnchor(anc=self.anchor_list.to(grids.device), grid=grids) # (B, A, H, W, 4)
    iou_mat = IoU(proposals=anchors, bboxes=bboxes) # (B, A*H*W, N)
    activated_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class, \
         activated_anc_coord, negative_anc_coord = ReferenceOnActivatedAnchors(
                                                                                anchors=anchors,
                                                                                bboxes=bboxes,
                                                                                grid=grids,
                                                                                iou_mat=iou_mat,
                                                                                method='FasterRCNN'
                                                                              )

    conf_scores, offsets, proposals = self.prop_module(
                                                        features=features,
                                                        pos_anchor_coord=activated_anc_coord,
                                                        pos_anchor_idx=activated_anc_ind,
                                                        neg_anchor_idx=negative_anc_ind,
                                                      )
    conf_loss = ConfScoreRegression(conf_scores=conf_scores, batch_size=B)
    reg_loss = BboxRegression(offsets=offsets, GT_offsets=GT_offsets, batch_size=B)
    total_loss = w_conf * conf_loss + w_reg * reg_loss
    pos_anchor_idx = activated_anc_ind
    anc_per_img = anchors.shape[1] * anchors.shape[2] * anchors.shape[3]
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    if output_mode == 'loss':
      return total_loss
    else:
      return total_loss, conf_scores, proposals, features, GT_class, pos_anchor_idx, anc_per_img


  def inference(self, images, thresh=0.5, nms_thresh=0.7, mode='RPN'):
    """
    Inference-time forward pass for the Region Proposal Network.

    Inputs:
    - images: Tensor of shape (B, 3, H, W) giving input images
    - thresh: Threshold value on confidence scores. Proposals with a predicted
      object probability above thresh should be kept. HINT: You can convert the
      object score to an object probability using a sigmoid nonlinearity.
    - nms_thresh: IoU threshold for non-maximum suppression
    - mode: One of 'RPN' or 'FasterRCNN' to determine the outputs.

    The region proposal network can output a variable number of region proposals
    per input image. We assume that the input image images[i] gives rise to
    P_i final propsals after thresholding and NMS.

    NOTE: NMS is performed independently per-image!

    Outputs:
    - final_proposals: List of length B, where final_proposals[i] is a Tensor
      of shape (P_i, 4) giving the coordinates of the predicted region proposals
      for the input image images[i].
    - final_conf_probs: List of length B, where final_conf_probs[i] is a
      Tensor of shape (P_i,) giving the predicted object probabilities for each
      predicted region proposal for images[i]. Note that these are
      *probabilities*, not scores, so they should be between 0 and 1.
    - features: Tensor of shape (B, D, H', W') giving the image features
      predicted by the backbone network for each element of images.
      If mode is "RPN" then this is a dummy list of zeros instead.
    """
    assert mode in ('RPN', 'FasterRCNN'), 'invalid inference mode!'

    features, final_conf_probs, final_proposals = None, None, None
    ##############################################################################
    # TODO: Predicting the RPN proposal coordinates `final_proposals` and        #
    # confidence scores `final_conf_probs`.                                     #
    # The overall steps are similar to the forward pass but now you do not need  #
    # to decide the activated nor negative anchors.                              #
    # HINT: Threshold the conf_scores based on the threshold value `thresh`.     #
    # Then, apply NMS to the filtered proposals given the threshold `nms_thresh`.#
    # HINT: Use `torch.no_grad` as context to speed up the computation.          #
    ##############################################################################
    # Replace "pass" statement with your code
    with torch.no_grad():
      final_proposals = []
      final_conf_probs = []
      B = images.shape[0]
      features = self.feat_extractor(images)
      # conf_scores: (B, A, 2, H, W)
      # offsets: (B, A, 4, H, W)
      # print('features.shape:', features.shape)
      conf_scores, offsets = self.prop_module(features=features)
      # print('conf_scores.shape:', conf_scores.shape)
      # print('offsets.shape:', offsets.shape)
      grid = GenerateGrid(B)
      anchors = GenerateAnchor(self.anchor_list.to(grid.device), grid) # (B, A, W, H, 4)
      proposals = GenerateProposal(anchors=anchors, offsets=offsets.permute(0,1,3,4,2), method='FasterRCNN') # (B, A, W, H, 4)
      unselected_conf_scores = conf_scores.permute(0,1,3,4,2) # (B, A, H, W, 2)
      unselected_conf_scores = torch.sigmoid(unselected_conf_scores[...,0]) # (B, A, H, W)
      for i in range(B):
        cur_conf_scores = unselected_conf_scores[i] # (A, H, W)
        cur_proposals = proposals[i] # (A, W, H, 4)
        to_choose = cur_conf_scores > thresh # bool index: (A, H, W)
        probs_for_nms = cur_conf_scores[to_choose].contiguous().view(-1)
        proposals_for_nms = cur_proposals[to_choose].contiguous().view(-1, 4)
        selected = torchvision.ops.nms(boxes=proposals_for_nms, scores=probs_for_nms, iou_threshold=nms_thresh)
        final_proposals.append(proposals_for_nms[selected])
        final_conf_probs.append(probs_for_nms[selected].unsqueeze(1))
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    if mode == 'RPN':
      features = [torch.zeros_like(i) for i in final_conf_probs] # dummy class
    return final_proposals, final_conf_probs, features


class TwoStageDetector(nn.Module):
  def __init__(self, in_dim=1280, hidden_dim=256, num_classes=20, \
               roi_output_w=2, roi_output_h=2, drop_ratio=0.3):
    super().__init__()

    assert(num_classes != 0)
    self.num_classes = num_classes
    self.roi_output_w, self.roi_output_h = roi_output_w, roi_output_h
    ##############################################################################
    # TODO: Declare your RPN and the region classification layer (in Fast R-CNN).#
    # The region classification layer is a sequential module with a Linear layer,#
    # followed by a Dropout (p=drop_ratio), a ReLU nonlinearity and another      #
    # Linear layer that predicts classification scores for each proposal.        #
    # HINT: The dimension of the two Linear layers are in_dim -> hidden_dim and  #
    # hidden_dim -> num_classes.                                                 #
    ##############################################################################
    # Your RPN and classification layers should be named as follows
    self.rpn = None
    self.cls_layer = None

    # Replace "pass" statement with your code
    self.rpn = RPN()
    self.cls_layer = nn.Sequential(
                                    nn.Linear(in_features=in_dim, out_features=hidden_dim),
                                    nn.Dropout(p=drop_ratio),
                                    nn.Linear(in_features=hidden_dim, out_features=num_classes)
                                  )
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

  def forward(self, images, bboxes):
    """
    Training-time forward pass for our two-stage Faster R-CNN detector.

    Inputs:
    - images: Tensor of shape (B, 3, H, W) giving input images
    - bboxes: Tensor of shape (B, N, 5) giving ground-truth bounding boxes
      and category labels, from the dataloader.

    Outputs:
    - total_loss: Torch scalar giving the overall training loss.
    """
    total_loss = None
    ##############################################################################
    # TODO: Implement the forward pass of TwoStageDetector.                      #
    # A few key steps are outlined as follows:                                   #
    # i) RPN, including image feature extraction, grid/anchor/proposal           #
    #       generation, activated and negative anchors determination.            #
    # ii) Perform RoI Align on proposals and meanpool the feature in the spatial #
    #     dimension.                                                             #
    # iii) Pass the RoI feature through the region classification layer which    #
    #      gives the class probilities.                                          #
    # iv) Compute class_prob through the prediction network and compute the      #
    #     cross entropy loss (cls_loss) between the prediction class_prob and    #
    #      the reference GT_class. Hint: Use F.cross_entropy loss.               #
    # v) Compute the total_loss which is formulated as:                          #
    #    total_loss = rpn_loss + cls_loss.                                       #
    ##############################################################################
    # Replace "pass" statement with your code

    # features is (B, C, H, W), the first dim of other outputs is M
    # print("images: ", images.shape)
    # print("bboxes: ", bboxes.shape)
    rpn_loss, conf_scores, proposals, features, GT_class, pos_anchor_idx, anc_per_img = self.rpn(images, bboxes, output_mode='all')
    M = proposals.shape[0]
    boxes = torch.zeros((M, 5)).to(proposals)

    boxes[:, 0] =  torch.div(pos_anchor_idx, anc_per_img, rounding_mode='trunc') 
    boxes[:, 1:5] = proposals
    # pooled_roi is (M, C, roi_H, roi_W)
    aligned_roi_features = torchvision.ops.roi_align(
                                            input=features,
                                            boxes=boxes,
                                            output_size=(self.roi_output_h, self.roi_output_w)
                                          )
    # mean pool (M, C)
    pooled_roi_features = torch.mean(aligned_roi_features, dim=(-2, -1))
    # (M, classes_num)
    cls_probabilities = self.cls_layer(pooled_roi_features)

    cls_loss = torch.nn.functional.cross_entropy(cls_probabilities, GT_class)
    total_loss = cls_loss + rpn_loss
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return total_loss

  def inference(self, images, thresh=0.5, nms_thresh=0.7):
    """"
    Inference-time forward pass for our two-stage Faster R-CNN detector

    Inputs:
    - images: Tensor of shape (B, 3, H, W) giving input images
    - thresh: Threshold value on NMS object probabilities
    - nms_thresh: IoU threshold for NMS in the RPN

    We can output a variable number of predicted boxes per input image.
    In particular we assume that the input images[i] gives rise to P_i final
    predicted boxes.

    Outputs:
    - final_proposals: List of length (B,) where final_proposals[i] is a Tensor
      of shape (P_i, 4) giving the coordinates of the final predicted boxes for
      the input images[i]
    - final_conf_probs: List of length (B,) where final_conf_probs[i] is a
      Tensor of shape (P_i,) giving the predicted probabilites that the boxes
      in final_proposals[i] are objects (vs background)
    - final_class: List of length (B,), where final_class[i] is an int64 Tensor
      of shape (P_i,) giving the predicted category labels for each box in
      final_proposals[i].
    """
    final_proposals, final_conf_probs, final_class = None, None, None
    ##############################################################################
    # TODO: Predicting the final proposal coordinates `final_proposals`,        #
    # confidence scores `final_conf_probs`, and the class index `final_class`.  #
    # The overall steps are similar to the forward pass but now you do not need #
    # to decide the activated nor negative anchors.                             #
    # HINT: Use the RPN inference function to perform thresholding and NMS, and #
    # to compute final_proposals and final_conf_probs. Use the predicted class  #
    # probabilities from the second-stage network to compute final_class.       #
    ##############################################################################
    # Replace "pass" statement with your code
    with torch.no_grad():
      final_proposals, final_conf_probs, features = self.rpn.inference(images, thresh, nms_thresh, mode='FasterRCNN')

      # pooled_roi is (M, C, roi_H, roi_W) and M is the the total number of proposals
      aligned_roi_features = torchvision.ops.roi_align(
                                              input=features,
                                              boxes=final_proposals,
                                              output_size=(self.roi_output_h, self.roi_output_w)
                                            )
      # mean pool (M, C)
      pooled_roi_features = torch.mean(aligned_roi_features, dim=(-2, -1))
      # (M, classes_num)
      cls_probabilities = self.cls_layer(pooled_roi_features)
      classes = torch.argmax(cls_probabilities, dim=-1, keepdim=True) # (M, 1)
      proposal_num_per_img = [final_proposals[bacth].shape[0] for bacth in range(len(final_proposals))]
      final_class = torch.split_with_sizes(classes, proposal_num_per_img)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return final_proposals, final_conf_probs, final_class
