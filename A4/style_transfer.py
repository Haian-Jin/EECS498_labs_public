"""
Implements a style transfer in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

from numpy import imag
import torch
import torch.nn as nn
from a4_helper import *

def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from style_transfer.py!')

def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).
    
    Returns:
    - scalar content loss
    """
    ##############################################################################
    # TODO: Compute the content loss for style transfer.                         #
    ##############################################################################
    # Replace "pass" statement with your code
    
    differ = content_current - content_original
    loss = torch.sum(torch.pow(differ,2)) * content_weight

    return loss

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################


def gram_matrix(features:torch.Tensor, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    gram = None
    ##############################################################################
    # TODO: Compute the Gram matrix from features.                               #
    # Don't forget to implement for both normalized and non-normalized version   #
    ##############################################################################
    # Replace "pass" statement with your code
    flat_feat = features.flatten(start_dim=2).unsqueeze(2) # (N, C, 1, D)
    trans_flat_feat = flat_feat.transpose(1,2) # (N, 1, C, D)
    # gram = flat_feat**2 + trans_flat_feat**2 - 2 * flat_feat * trans_flat_feat
    gram = flat_feat * trans_flat_feat
    gram = torch.sum(gram, dim=3)
    
    if normalize:
      N, C, H, W = features.shape
      number = C * H * W
      gram = gram / number

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return gram


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ##############################################################################
    # TODO: Computes the style loss at a set of layers.                          #
    # Hint: you can do this with one for loop over the style layers, and should  #
    # not be very much code (~5 lines).                                          #
    # You will need to use your gram_matrix function.                            #
    ##############################################################################
    # Replace "pass" statement with your code
    content_grams = []
    style_loss = 0
    for idx in style_layers:
      content_grams.append(gram_matrix(feats[idx]))
    assert len(content_grams) == len(style_targets), "Content and target don't have the same number of items"
    for i in range(len(content_grams)):
      style_loss += torch.sum((content_grams[i] - style_targets[i])**2)*style_weights[i]
    
    return style_loss
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################


def tv_loss(img:torch.Tensor, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    ##############################################################################
    # TODO: Compute total variation loss.                                        #
    # Your implementation should be vectorized and not require any loops!        #
    ##############################################################################
    # Replace "pass" statement with your code
    img_squeeze = img.squeeze()
    loss = torch.sum((img_squeeze[:,1:,:]-img_squeeze[:,:-1,:])**2) + torch.sum((img_squeeze[:,:,1:]-img_squeeze[:,:,:-1])**2)
    return loss * tv_weight
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################