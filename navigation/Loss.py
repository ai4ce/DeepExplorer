import torch.nn as nn
import torch

class ActionAssignerLoss(object):
    def __init__(self):
        self.ce_loss = nn.CrossEntropyLoss()
    def compute_loss(self, action_pred_logits, action_gt ):
        actionclass_num = action_pred_logits.shape[-1]
        # import pdb
        # pdb.set_trace()
        action_pred_logits = torch.reshape(action_pred_logits, shape=[-1, actionclass_num])
        # action_gt = torch.reshape(action_gt, shape=[-1, actionclass_num])
        action_gt = torch.reshape(action_gt, shape=[-1])

        loss = self.ce_loss(action_pred_logits, action_gt)

        return loss
