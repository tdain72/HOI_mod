from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda')

def _bce_balanced(out, target, ambiguity):
    loss = F.binary_cross_entropy(out, target)
    balancing_weight = torch.exp(ambiguity).to(device)
    balanced_loss = balancing_weight * loss
    return balanced_loss

class BCELoss_balanced(nn.Module):
    def __init__(self):
        super(BCELoss_balanced, self).__init__()
        self.bce_balanced = _bce_balanced
    def forward(self, out, target, ambiguity):
        return self.bce_balanced(out, target, ambiguity)

def _neg_loss_balanced(pred, gt, R, balance):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0
  # P_pos = 1
  # P_neg = 1.5
  # O = 1.5
  R_a = torch.log((R**2)+1)+1
  # exp_R_neg = torch.exp(R*P_neg)
  # Q = 1+(1/(1+(-10)*torch.exp(balance - 0.5)))
  # Q = torch.exp(balance*O)
  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * R_a * pos_inds 
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * R_a * neg_inds 
  # neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


class FocalLoss_balanced(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss_balanced, self).__init__()
    self.neg_loss_balanced = _neg_loss_balanced

  def forward(self, out, target, R, balance):
    return self.neg_loss_balanced(out, target, R, balance)