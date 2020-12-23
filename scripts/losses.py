from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

NO_CLASS = 29

sigmoid = nn.Sigmoid()
device = torch.device('cuda')

def get_balanced_weight(target, alpha = 0, beta = 6, eps = 1e-13 ):
  label_ambiguity = target.sum(axis=1)
  ambiguity = 1/(1+torch.exp(alpha*label_ambiguity.clamp(min=eps)-beta))
  return (ambiguity).to(device)

def get_margin(bias, ambiguity, k = 0.1, in_batch = False, eps=1e-13):
  # if in_batch:
  #   pos_num = bias.eq(1).sum(0).float().clamp(min=eps)
  #   sample_num = bias.shape[0]
  #   bias = (pos_num/sample_num).to(device)
  amb = torch.log10(1/(1+(9*torch.exp(-ambiguity))))+1
  b = -torch.log((1/bias)-1)
  v = -k*amb*b
  return v.to(device)

def get_negative_weight(target, q = 8):
  size = target.size(0)*target.size(1)
  pos_inds = target.eq(1).float().to(device)
  num_pos = pos_inds.sum()
  neg_w = num_pos/(size * NO_CLASS)
  neg_w = neg_w**(1/q)
  return neg_w.to(device)


def _bce_balanced(pred, target, ambiguity, reduction):
  loss = F.binary_cross_entropy(pred, target, reduction=reduction)
  balancing_weight = (1+ambiguity).to(device)
  balanced_loss = balancing_weight * loss
  return balanced_loss

def _BCELoss_balanced(pred, target, balanced_weight, eps=1e-13, reduction = 'mean', gamma = 5):
  loss = 0
  if reduction == 'mean': 
    loss = -(target*pred.clamp(min=eps).log()+(1-target)*(1-pred).clamp(min=eps).log()).mean()
  if reduction == 'sum':
    loss = -(target*pred.clamp(min=eps).log()+(1-target)*(1-pred).clamp(min=eps).log()).sum()
  if reduction == 'none':
    loss = -(target*pred.clamp(min=eps).log()+(1-target)*(1-pred).clamp(min=eps).log())
  loss_balanced = balanced_weight.reshape((balanced_weight.shape[0], 1))*loss
  return loss_balanced

class BCELoss_balanced(nn.Module):
  def __init__(self, reduction='mean'):
    super(BCELoss_balanced, self).__init__()
    self.bce_balanced = _BCELoss_balanced
    self.reduction = reduction
    self.eps = 1e-13
    self.gamma = 1
  def forward(self, pred_dict, target, ambiguity, class_dist, class_bias):
    # margin = get_margin(class_bias, ambiguity, in_batch=False)
    # print(torch.max(margin))
    pred = (sigmoid(pred_dict['outputs'])
            *sigmoid(pred_dict['outputs_combine'])
            *sigmoid(pred_dict['outputs_single'])
            *sigmoid(pred_dict['outputs_gem'])
            *pred_dict['hum_obj_mask'])
    balanced_weight = get_balanced_weight(target, alpha=1, beta=4)
    return self.bce_balanced(pred, target, balanced_weight, eps=self.eps, reduction=self.reduction, gamma=self.gamma)

def _neg_loss_balanced(pred, target, alpha=0.5, gamma=2, reduction='none', eps=1e-13):
  bce_loss = -(target*pred.clamp(min=eps).log()+(1-target)*(1-pred).clamp(min=eps).log())
  pt = torch.exp(-bce_loss)
  focal_loss = alpha*(1-pt)**gamma*bce_loss
  return focal_loss


class FocalLoss_balanced(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss_balanced, self).__init__()
    self.neg_loss_balanced = _neg_loss_balanced

  def forward(self, pred, target):
    return self.neg_loss_balanced(pred, target)