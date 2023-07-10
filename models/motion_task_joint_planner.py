import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class MotionTaskJointPlanner(nn.Module):
    def __init__(self,
                 steplen=10,
                 backbone_nn='resnet18',
                 input_size=512,
                 hidden_size=512,
                 lstm_num_layers=2,
                 class_num=3,
                 add_history_memory = False ):
        super(MotionTaskJointPlanner, self).__init__()
        self.steplen = steplen
        self.backbone_nn_name = backbone_nn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.class_num = class_num
        self.add_history_memory = add_history_memory
        self.construct_backbone_nn()
        self.construct_lstm_model()
        self.construct_merge_fc_layer()
        self.construct_feat_predct_FC()
        self.construct_action_classify_logits()

        self.ce_loss = nn.CrossEntropyLoss()

    def construct_feat_predct_FC(self):
        '''
        explicitly add a FC layer to predict the next-step representation
        '''
        self.feat_predict = nn.Linear(in_features=self.input_size, out_features=self.input_size)

    def construct_action_classify_logits(self):
        self.action_classify_logits = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=self.class_num,
            bias=True)

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

        # remove the last fc-layer
        resnet18_model = nn.Sequential(*list(resnet18_model.children())[:-1])

        return resnet18_model

    def obtain_resnet50_pretrained_model(self):
        resnet50_model = models.resnet50(pretrained=True)

        # remove the last fc-layer
        resnet50_model = nn.Sequential(*list(resnet50_model.children())[:-1])

        return resnet50_model

    def construct_lstm_model(self):
        self.lstm = torch.nn.LSTM(input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.lstm_num_layers,
                                  bias=True,
                                  batch_first=True,
                                  dropout=0.02,
                                  bidirectional=False,
                                  proj_size=0)

    def construct_merge_fc_layer(self):
        self.fc_merger = self.linear_layer = nn.Sequential(
            torch.nn.Linear(in_features=self.input_size * 2,
                            out_features=self.input_size,
                            bias=True),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=self.input_size))


    def taskplanner_head(self, local_video_embed, history_memory_embed ):
        '''
        Given the LSTM output, we add two FC layers to map the LSTM output to
        get the prediction for each time step, and further map the predict to low
        dimension for stepwise action classification
        :param local_video_embed: [B, Seqlen, featsize]
        :param history_memory_embed: [B, featsize]
        :return: next-step feat prediction, action-mapped feat
        '''
        history_memory_embed = torch.unsqueeze(history_memory_embed, dim=0)
        if self.add_history_memory:
            history_memory_embed = torch.tile(history_memory_embed,
                                              dims=[self.lstm_num_layers,
                                                    1,
                                                    1])
            lstm_output, _ = self.lstm(local_video_embed, (history_memory_embed,
                                                           history_memory_embed))
        else:
            lstm_output, _ = self.lstm( local_video_embed, ) #[B, seq, featsize]

        seq_output = lstm_output #[B, seq-1, featuresize]
        seq_output = seq_output.contiguous()
        batch_size = seq_output.shape[0]
        seq_len = seq_output.shape[1]
        feat_size = seq_output.shape[2]

        lstm_feat_pred = seq_output.view(batch_size*seq_len, feat_size)
        lstm_fc_feat_pred = self.feat_predict( lstm_feat_pred )
        lstm_fc_feat_pred = lstm_fc_feat_pred.view(batch_size,
                                                   seq_len,
                                                   self.input_size)

        future_feat_pred = lstm_fc_feat_pred[:,-1,:]
        future_feat_pred = torch.squeeze( future_feat_pred ) #[B, SeqLen-1, featlen]
        current_feat_pred = lstm_fc_feat_pred[:,0:-1,:]
        current_feat_pred = torch.squeeze( current_feat_pred ) #[B, featlen]

        return current_feat_pred, future_feat_pred

    def motionplanner_head(self, first_step_embed, second_step_embed ):
        '''
        Give two neighboring image embed, the motion planner first merge them,
        then predict the action
        :param first_step_embed: [B, seqlen, featsize]
        :param second_step_embed: [B, seqlen, featsize]
        :return: [B, classnum], logits
        '''
        concat_embed = torch.cat(tensors=(first_step_embed, second_step_embed),
                                 dim=2 )

        batch_size = concat_embed.shape[0]
        seq_len = concat_embed.shape[1]
        feat_size = concat_embed.shape[2]

        concat_embed = concat_embed.view(batch_size*seq_len, feat_size )

        FC_embed = self.fc_merger( concat_embed )

        action_logit = self.action_classify_logits( FC_embed )

        action_logit = action_logit.view(batch_size, -1, self.class_num )

        return action_logit

    def action_classify_loss(self, action_logit, action_gt ):
        ce_loss = self.ce_loss( action_logit, action_gt )

        return ce_loss

    def feat_pred_loss(self, pred_feat, gt_feat ):
        squred_dist = torch.square( pred_feat - gt_feat )
        squred_dist = torch.mean(input=squred_dist, dim=1, keepdim=False)
        loss = torch.mean( squred_dist,dim=0,keepdim=False)

        return loss

    def predict_next_action(self, current_obs, local_video_input, history_memory_input ):
        '''
        during exploration, we input local video input, and history memory input
        to predict the next action to execute
        :param local_video_input: local video input: [1, seqlen, 3, 224, 224 ]
        :param history_memory_input: history_memory input: [1, seqlen1, 3, 224, 224]
        :return: next action to execute
        '''
        local_video_embed = self.embed_backbone(local_video_input)
        local_video_embed = torch.squeeze( local_video_embed )
        local_video_embed = torch.unsqueeze( local_video_embed, dim=0 )

        history_memory_input = self.embed_backbone( history_memory_input )
        history_memory_input = torch.squeeze( history_memory_input )
        history_memory_input = torch.unsqueeze( history_memory_input, dim=0 )
        history_memory_input = torch.mean(history_memory_input,
                                          dim=1,
                                          keepdim=False)

        current_obs_embed = self.embed_backbone( current_obs )
        current_obs_embed = torch.squeeze( current_obs_embed )
        current_obs_embed = torch.unsqueeze( current_obs_embed, dim=0 )

        current_feat_pred, future_feat_pred = self.taskplanner_head(
            local_video_embed=local_video_embed,
            history_memory_embed=history_memory_input)

        current_obs_embed = torch.unsqueeze( current_obs_embed, dim=1 )
        nextstep_feat_pred = torch.unsqueeze( future_feat_pred, dim=0 )
        nextstep_feat_pred = torch.unsqueeze(nextstep_feat_pred, dim=0)
        reinforced_action_logit = self.motionplanner_head(current_obs_embed,
                                                          nextstep_feat_pred)

        reinforced_action_logit = torch.squeeze( reinforced_action_logit )

        action_prob = F.softmax( reinforced_action_logit, dim=-1 )

        return action_prob

    def predict_next_action_withfeat(self, current_obs, local_video_input, history_memory_input ):
        '''
        during exploration, we input local video input, and history memory input
        to predict the next action to execute
        :param local_video_input: local video input: [1, seqlen, 3, 224, 224 ]
        :param history_memory_input: history_memory input: [1, seqlen1, 3, 224, 224]
        :return: next action to execute
        '''
        local_video_embed = self.embed_backbone(local_video_input)
        local_video_embed = torch.squeeze( local_video_embed )
        local_video_embed = torch.unsqueeze( local_video_embed, dim=0 )

        history_memory_input = self.embed_backbone( history_memory_input )
        history_memory_input = torch.squeeze( history_memory_input )
        history_memory_input = torch.unsqueeze( history_memory_input, dim=0 )
        history_memory_input = torch.mean(history_memory_input,
                                          dim=1,
                                          keepdim=False)

        current_obs_embed = self.embed_backbone( current_obs )
        current_obs_embed = torch.squeeze( current_obs_embed )
        current_obs_embed = torch.unsqueeze( current_obs_embed, dim=0 )

        current_feat_pred, future_feat_pred = self.taskplanner_head(
            local_video_embed=local_video_embed,
            history_memory_embed=history_memory_input)

        current_obs_embed = torch.unsqueeze( current_obs_embed, dim=1 )
        nextstep_feat_pred = torch.unsqueeze( future_feat_pred, dim=0 )
        nextstep_feat_pred = torch.unsqueeze(nextstep_feat_pred, dim=0)
        reinforced_action_logit = self.motionplanner_head(current_obs_embed,
                                                          nextstep_feat_pred)

        reinforced_action_logit = torch.squeeze( reinforced_action_logit )

        action_prob = F.softmax( reinforced_action_logit, dim=-1 )

        return action_prob, current_obs_embed, nextstep_feat_pred

    def eval_test(self, local_video_input ):
        local_video_embed = self.embed_backbone( local_video_input )
        local_video_embed = torch.squeeze( local_video_embed )

        first_obs_embed = local_video_embed[:,0:-1,:]
        second_obs_embed = local_video_embed[:,1:,:]

        action_logit = self.motionplanner_head( first_obs_embed,
                                                second_obs_embed )

        return action_logit

    def gather_all_losses(self,
                          motionplanner_logit,
                          current_feat_pred,
                          feat_embed_no_future,
                          reinforced_action_logit,
                          nextstep_feat_pred,
                          nextstep_obs_embed,
                          motionplanner_action_gt,
                          reinforced_action_gt):
        batch_size = motionplanner_logit.shape[0]
        seq_len = motionplanner_logit.shape[1]
        motionplanner_logit = motionplanner_logit.view(batch_size*seq_len, self.class_num)
        motionplanner_action_gt = motionplanner_action_gt.contiguous().view(-1)
        motionplanner_classify_loss = self.ce_loss( motionplanner_logit, motionplanner_action_gt )
        reinforced_classify_loss = self.ce_loss( reinforced_action_logit, reinforced_action_gt )

        feat_pred_loss = self.feat_pred_loss( nextstep_feat_pred,
                                              nextstep_obs_embed )

        lstm_feat_pred_loss = self.feat_pred_loss( current_feat_pred,
                                                   feat_embed_no_future )

        loss = motionplanner_classify_loss + lstm_feat_pred_loss + reinforced_classify_loss + feat_pred_loss

        return loss

    def rearrange_gtlabel(self, action_gt ):
        '''
        rearrange the input action label w.r.t. different tasks
        :param action_gt: [B, N]
        :return: different action gt label
        '''
        motionplanner_action_gt = action_gt[:, 0:-1 ]
        taskplanner_action_gt = action_gt[:,0:-1]
        reinforced_action_gt = action_gt[:,-1]

        return motionplanner_action_gt, taskplanner_action_gt, reinforced_action_gt

    def forward(self, video_obs_input, history_memory_input, nextstep_obs_input ):
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
        batch_size = video_obs_input.shape[0]
        seq_len_video = video_obs_input.shape[1]
        channel_num = video_obs_input.shape[2]
        height = video_obs_input.shape[3]
        width = video_obs_input.shape[4]

        video_obs_input = video_obs_input.view(batch_size*seq_len_video,
                                               channel_num,
                                               height,
                                               width)

        video_obs_embed = self.embed_backbone( video_obs_input )
        video_obs_embed = torch.squeeze( video_obs_embed )
        embed_size = video_obs_embed.shape[-1]
        video_obs_embed = video_obs_embed.view(batch_size,
                                               seq_len_video,
                                               embed_size)

        seq_len_history = history_memory_input.shape[1]
        history_memory_input = history_memory_input.view(batch_size*seq_len_history,
                                                         channel_num,
                                                         height,
                                                         width)

        history_memory_embed = self.embed_backbone( history_memory_input )
        history_memory_embed = torch.squeeze( history_memory_embed )
        history_memory_embed = history_memory_embed.view(batch_size,
                                                         seq_len_history,
                                                         embed_size)
        history_memory_embed = torch.mean( history_memory_embed,
                                           dim=1,
                                           keepdim=False )

        nextstep_obs_embed = self.embed_backbone( nextstep_obs_input )
        nextstep_obs_embed = torch.squeeze( nextstep_obs_embed )

        first_step_embed = video_obs_embed[:,0:-1,:]
        second_step_embed = video_obs_embed[:,1:,:]

        motionplanner_action_logit = self.motionplanner_head( first_step_embed,
                                                              second_step_embed )

        current_feat_pred, future_feat_pred = self.taskplanner_head( video_obs_embed,
                                                                    history_memory_embed )

        #reinforce the predicted feature through motion planner head
        last_step_embed = video_obs_embed[:,-1,:]
        last_step_embed = torch.unsqueeze( last_step_embed, dim=1 )
        nextstep_feat_pred = torch.unsqueeze( future_feat_pred, dim=1 )
        reinforced_action_logit = self.motionplanner_head( last_step_embed,
                                                           nextstep_feat_pred )

        nextstep_feat_pred = torch.squeeze( nextstep_feat_pred )
        reinforced_action_logit = torch.squeeze( reinforced_action_logit )

        feat_embed_no_future = video_obs_embed[:,1:,:]
        feat_embed_no_future = feat_embed_no_future.contiguous()

        return motionplanner_action_logit, \
               current_feat_pred, \
               feat_embed_no_future, \
               reinforced_action_logit, \
               nextstep_feat_pred, \
               nextstep_obs_embed


