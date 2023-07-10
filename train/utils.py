import torch
import torch.nn.functional as F

def get_accuracy_rate(seq_pred, seq_gt):
    '''
    compute the accuracy rate of imgobs to action list prediction
    :param seq_pred: logits, of shape [batchsize, seq, class_num]
    :param seq_gt: ground truth label, of shape [batchsize, seq]
    :return: scalar, the classification accuracy rate
    '''
    # seq_pred = seq_pred[:, 1:, :]
    class_num = seq_pred.shape[2]
    seq_pred = torch.reshape(seq_pred, shape=(-1, class_num))
    input_num = seq_pred.shape[0]
    seq_pred = F.softmax(seq_pred, dim=1)
    seq_pred = torch.argmax(seq_pred, dim=1, keepdim=False)

    seq_gt = torch.reshape(seq_gt, shape=(-1,))
    accu_num = torch.sum(seq_pred == seq_gt, dtype=torch.float32)

    accu_rate = accu_num / (input_num + 0.000001)

    return accu_rate

def rearrange_gtlabel(action_gt):
    '''
    rearrange the input action label w.r.t. different tasks
    :param action_gt: [B, N]
    :return: different action gt label
    '''
    motionplanner_action_gt = action_gt[:, 0:-1]
    taskplanner_action_gt = action_gt[:, 0:-1]
    reinforced_action_gt = action_gt[:, -1]

    return motionplanner_action_gt, taskplanner_action_gt, reinforced_action_gt

def my_feat_pred_loss(pred_feat, gt_feat):
    '''
    calculate the l2 loss between two features
    :param pred_feat: [B, featlen] or [B, seqlen, featlen]
    :param gt_feat: [B, featlen] or [B, seqlen, featlen]
    :return: the scalar, loss value
    '''
    if len(pred_feat.shape) == 3 and len(gt_feat.shape) == 3:
        pred_feat = pred_feat.contiguous()
        gt_feat = gt_feat.contiguous()
        batch_size = pred_feat.shape[0]
        seq_len = pred_feat.shape[1]
        feat_len = pred_feat.shape[2]
        pred_feat = pred_feat.view( batch_size*seq_len, feat_len )
        gt_feat = gt_feat.view( batch_size*seq_len, feat_len )
    squred_dist = torch.square(pred_feat - gt_feat)
    squred_dist = torch.mean(input=squred_dist, dim=1, keepdim=False)
    loss = torch.mean(squred_dist, dim=0, keepdim=False)

    return loss


def gather_all_losses(motionplanner_logit,
                      current_feat_pred,
                      feat_embed_no_future,
                      reinforced_action_logit,
                      nextstep_feat_pred,
                      nextstep_obs_embed,
                      motionplanner_action_gt,
                      reinforced_action_gt):
    ce_loss = torch.nn.CrossEntropyLoss()
    class_num = 3
    batch_size = motionplanner_logit.shape[0]
    seq_len = motionplanner_logit.shape[1]
    motionplanner_logit = motionplanner_logit.view(batch_size * seq_len,
                                                   class_num)
    motionplanner_action_gt = motionplanner_action_gt.contiguous().view(-1)
    motionplanner_classify_loss = ce_loss(motionplanner_logit,
                                               motionplanner_action_gt)
    reinforced_classify_loss = ce_loss(reinforced_action_logit,
                                            reinforced_action_gt)

    feat_pred_loss = my_feat_pred_loss(nextstep_feat_pred,
                                         nextstep_obs_embed)

    lstm_feat_pred_loss = my_feat_pred_loss(current_feat_pred,
                                              feat_embed_no_future)

    loss = motionplanner_classify_loss + lstm_feat_pred_loss + reinforced_classify_loss + feat_pred_loss

    return loss
