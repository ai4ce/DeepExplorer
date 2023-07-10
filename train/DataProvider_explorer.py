import os
import torch
import json
import glob
import random
import numpy as np
import cv2

class DeepExplorerDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir='./expert_data/',
                 history_memory_len = 10,
                 local_video_input_len = 10,
                 predict_future_gap = 1,
                 seq_len=10):
        self.root_dir = root_dir
        self.json_file_list = self.get_all_json_file(root_dir)
        self.history_memory_len = history_memory_len
        self.local_video_input_len = local_video_input_len
        self.predict_future_gap = predict_future_gap
        self.seq_len = seq_len

    def compute_valid_length(self):
        valid_len = 0
        for input_json_file in self.json_file_list:
            input_dict = self.parse_one_jsonfile(input_json_file)
            action_list = input_dict['action_list']
            valid_len_tmp = len(action_list)
            valid_len_tmp = valid_len_tmp - valid_len_tmp % self.seq_len
            valid_len += valid_len_tmp

        assert valid_len > 0

        return valid_len

    def parse_one_jsonfile(self, input_json_filename):
        assert os.path.exists(input_json_filename)
        with open(input_json_filename, 'r') as f:
            input_dict = json.load(f)

        return input_dict

    def get_all_json_file(self, input_dir):
        subdir_list = glob.glob(os.path.join(input_dir, '*'))
        json_file_list = list()

        for subdir_name in subdir_list:
            json_file_basename = os.path.basename(subdir_name) + '.json'
            assert os.path.exists(
                os.path.join(subdir_name, json_file_basename))
            json_file_list.append(
                os.path.join(subdir_name, json_file_basename))

        assert len(json_file_list) > 0

        return json_file_list

    def normalize_an_img(self, input_img):
        '''
        The Input Image, the channel is in R-G-B order
        :param input_img: [H, W, 3], float32, torch.tensor
        :return: normalized tensor, in [-1, 1] range
        '''
        MEAN = 255. * torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        STD = 255. * torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

        input_img = input_img.permute(-1, 0, 1)

        output_img = (input_img - MEAN[:, None, None]) / STD[:, None,
                                                         None]  # [channel, height, width]

        return output_img

    def prepare_an_img(self, img_name):
        assert os.path.exists(img_name)
        img = cv2.imread(img_name, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = cv2.resize(img, (224, 224))
        img = torch.from_numpy(img)
        img = self.normalize_an_img(img)

        return img

    def pad_action(self, input_action_array):
        output_action_array = 3 * np.ones(self.seq_len).astype(np.int32)
        output_action_array[0:input_action_array.shape[0]] = input_action_array

        return output_action_array

    def __len__(self):
        return self.compute_valid_length()

    def prepare_history_memory_feature(self, history_memory_img_list ):
        evenly_sampled_imgs = list()
        sample_step = len(history_memory_img_list)/float(self.history_memory_len)

        for sample_idx in range(self.history_memory_len):
            imgid2sample = int( sample_idx*sample_step )
            imgid2sample = len(history_memory_img_list) - 1 if imgid2sample >= len(history_memory_img_list) else imgid2sample
            evenly_sampled_imgs.append( history_memory_img_list[imgid2sample ])

        preprocessed_img_batch = list()
        for imgname_tmp in evenly_sampled_imgs:
            preprocessed_img_batch.append( self.prepare_an_img( imgname_tmp ))
        history_memory_feat = torch.stack( tensors=preprocessed_img_batch,
                                           dim=0 )

        return history_memory_feat

    def prepare_local_video_input(self, local_video_img_list ):

        preprocessed_img_batch = list()
        for imgname_tmp in local_video_img_list:
            preprocessed_img_batch.append(self.prepare_an_img(imgname_tmp))
        video_feat_input = torch.stack(tensors=preprocessed_img_batch, dim=0)

        return video_feat_input

    def prepare_target_input(self, target_img_name ):
        target_feat = self.prepare_an_img( target_img_name )

        return target_feat

    def __getitem__(self, index):
        rand_idx = random.randint(0, len(self.json_file_list) - 1)
        input_dict = self.parse_one_jsonfile(self.json_file_list[rand_idx])
        img_num = len(input_dict['panoimg_list'])
        retrieval_start_idx = random.randint(1,
                                             img_num - self.local_video_input_len - self.predict_future_gap - 1)

        output_action_list = input_dict['action_list'][
                             retrieval_start_idx:retrieval_start_idx + self.local_video_input_len]
        output_action = np.array(output_action_list, np.int32)
        output_action = output_action - 1
        output_action = torch.from_numpy(output_action)
        output_action = output_action.to(torch.long)

        local_video_img_list = input_dict['panoimg_list'][
                               retrieval_start_idx:retrieval_start_idx + self.local_video_input_len]

        target_img_name = input_dict['panoimg_list'][
            retrieval_start_idx + self.local_video_input_len + self.predict_future_gap - 1 ]

        history_memory_img_list = input_dict['panoimg_list'][
            0:retrieval_start_idx]

        history_memory_feat = self.prepare_history_memory_feature( history_memory_img_list )
        local_video_feat = self.prepare_local_video_input( local_video_img_list )
        target_img_feat = self.prepare_target_input( target_img_name )

        return history_memory_feat, local_video_feat, target_img_feat, output_action

