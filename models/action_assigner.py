import torch
import torch.nn as nn
import torchvision.models as models

class ActionAssigner(nn.Module):
    def __init__(self,
                 backbone_nn='resnet18',
                 input_size=512,
                 hidden_size=512,
                 actionlen=6,
                 actionclass_num=4):
        super(ActionAssigner, self).__init__()
        self.backbone_nn = backbone_nn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.actionlen = actionlen
        self.backbone_nn_name = backbone_nn
        self.actionclass_num = actionclass_num

        self.construct_actionseq_predFC()
        self.construct_actionseq_predict_logits()
        self.construct_backbone_nn()
        self.construct_merge_fc_layer()
        self.construct_LSTM_layer()

    def construct_LSTM_layer(self):
        self.lstm = torch.nn.LSTM(input_size=self.input_size//2,
                                  hidden_size=self.hidden_size//2,
                                  num_layers=1,
                                  bias=True,
                                  batch_first=True,
                                  bidirectional=True,
                                  proj_size=0)

    def construct_actionseq_predict_logits(self):
        self.action_classify_logits = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=self.actionclass_num,
            bias=True)

    def construct_actionseq_predFC(self):
        self.actionseq_pred_FC0 = self.linear_layer = nn.Sequential(
            torch.nn.Linear(in_features=self.input_size,
                            out_features=self.input_size//2,
                            bias=True),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=self.input_size//2))

        self.actionseq_pred_FC1 = self.linear_layer = nn.Sequential(
            torch.nn.Linear(in_features=self.input_size,
                            out_features=self.input_size//2,
                            bias=True),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=self.input_size//2))

        self.actionseq_pred_FC2 = self.linear_layer = nn.Sequential(
            torch.nn.Linear(in_features=self.input_size,
                            out_features=self.input_size//2,
                            bias=True),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=self.input_size//2))

        self.actionseq_pred_FC3 = self.linear_layer = nn.Sequential(
            torch.nn.Linear(in_features=self.input_size,
                            out_features=self.input_size//2,
                            bias=True),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=self.input_size//2))

        self.actionseq_pred_FC4 = self.linear_layer = nn.Sequential(
            torch.nn.Linear(in_features=self.input_size,
                            out_features=self.input_size//2,
                            bias=True),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=self.input_size//2))

        self.actionseq_pred_FC5 = self.linear_layer = nn.Sequential(
            torch.nn.Linear(in_features=self.input_size,
                            out_features=self.input_size//2,
                            bias=True),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=self.input_size//2))


    def construct_backbone_nn(self):
        assert self.backbone_nn_name in ['resnet18', 'resnet50']
        if self.backbone_nn_name == 'resnet18':
            self.embed_backbone = self.obtain_resnet18_pretrained_model()
        elif self.backbone_nn_name == 'resnet50':
            self.embed_backbone = self.obtain_resnet50_pretrained_model()
        else:
            raise ValueError('unknown backbone neural network!')

    def obtain_resnet18_pretrained_model(self):
        resnet18_model = models.resnet18(pretrained=True)
        resnet18_model = nn.Sequential(*list(resnet18_model.children())[:-1])

        return resnet18_model

    def obtain_resnet50_pretrained_model(self):
        resnet50_model = models.resnet50(pretrained=True)
        resnet50_model = nn.Sequential(*list(resnet50_model.children())[:-1])

        return resnet50_model

    def construct_merge_fc_layer(self):
        self.fc_merger = self.linear_layer = nn.Sequential(
            torch.nn.Linear(in_features=self.input_size * 2,
                            out_features=self.input_size,
                            bias=True),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=self.input_size))


    def forward(self, start_img_input, target_img_input,  ):
        '''
        Local motion planner, in each forward process, takes one current observation
        image and one target observation image, as well as the action-sequence leading
        the agent from the current observation to the target observation. Please note
        that the maximum action list length is pre-set as `steplen`, otherwise it is
        padded to `steplen` by concatenating `stop` action. The actions are:
        0: MoveForward, 1: Turn-Left, 2: Turn-Right, 3: Stop
        :param video_obs_input: [batch, seqlen, channel, height, width]
        :param target_obs: [batch, channel, height, width]
        :param action_input: [batch, steplen], of integers in [0,1,2,3]
        :return: logits
        '''
        start_img_embed = self.embed_backbone( start_img_input )
        target_img_embed = self.embed_backbone( target_img_input )

        start_img_embed = torch.squeeze( start_img_embed )
        target_img_embed = torch.squeeze( target_img_embed )

        concat_embed_feat = torch.cat(tensors=(start_img_embed,
                                               target_img_embed),
                                      dim=-1)

        concat_embed_feat = concat_embed_feat.contiguous()
        merged_feat = self.fc_merger(concat_embed_feat)

        actseq_fc0 = self.actionseq_pred_FC0(merged_feat)
        actseq_fc1 = self.actionseq_pred_FC1(merged_feat)
        actseq_fc2 = self.actionseq_pred_FC2(merged_feat)
        actseq_fc3 = self.actionseq_pred_FC3(merged_feat)
        actseq_fc4 = self.actionseq_pred_FC4(merged_feat)
        actseq_fc5 = self.actionseq_pred_FC5(merged_feat)

        actseq_feat = torch.stack(tensors=(actseq_fc0,
                                           actseq_fc1,
                                           actseq_fc2,
                                           actseq_fc3,
                                           actseq_fc4,
                                           actseq_fc5),
                                  dim=1)

        actseq_feat = self.lstm(actseq_feat)[0] #[batchsize, seqlen, 512]

        actionseq_pred_logits = self.action_classify_logits(actseq_feat)


        return actionseq_pred_logits


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
