import os
import torch
import json
import glob
import random
import numpy as np
import cv2

class NavDataProvider(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir='data_dir/',
                 action_len=6,
                 actionclass_num = 4):
        self.root_dir = root_dir
        self.action_len = action_len
        self.actionclass_num = actionclass_num
        self.get_room_reconnect_list()
        self.get_imgactionpair_list()

    def get_room_reconnect_list(self):
        room_list = glob.glob( os.path.join( self.root_dir, '*') )
        assert len(room_list) == 14
        reconnect_filename_list = list()

        for room_name in room_list:
            reconnect_filename_list_tmp = glob.glob( os.path.join(room_name,
                                                                  '*_reconnect*.json') )

            assert len(reconnect_filename_list_tmp) > 0
            reconnect_filename_list.extend(reconnect_filename_list_tmp)

        self.reconnect_filename_list = reconnect_filename_list

    def get_imgactionpair_list(self):
        imgaction_pair_list = list()
        for reconnect_filename in self.reconnect_filename_list:
            with open(reconnect_filename, 'r') as f:
                reconnect_list_tmp = json.load(f)
            for imgaction_pair_tmp in reconnect_list_tmp:
                if len(imgaction_pair_tmp['action_list']) > self.action_len:
                    continue
                imgaction_pair_list.append(imgaction_pair_tmp)
            # imgaction_pair_list.extend(reconnect_list_tmp)

        assert len(imgaction_pair_list) > 0

        random.shuffle(imgaction_pair_list)

        self.imgaction_pair_list = imgaction_pair_list

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

    def prepare_actions(self, action_list):
        actions_onehot = np.zeros(shape=[self.action_len], dtype=np.float32)
        for seq_id, action_tmp in enumerate(action_list):
            actions_onehot[seq_id] = int(float(action_tmp))

        actions_onehot = torch.from_numpy(actions_onehot).to(torch.long)

        return actions_onehot

    def prepare_one_imgaction_pair(self, imgaction_pair_dict):
        start_imgname = imgaction_pair_dict['start_img']
        target_imgname = imgaction_pair_dict['target_img']
        action_list = imgaction_pair_dict['action_list']

        start_img = self.prepare_an_img(start_imgname)
        target_img = self.prepare_an_img(target_imgname)

        actions = self.prepare_actions(action_list)

        return start_img, target_img, actions


    def __len__(self):
        return len(self.imgaction_pair_list)

    def __getitem__(self, index):
        rand_idx = random.randint(0, len(self.imgaction_pair_list)-1)
        start_img, target_img, actions = self.prepare_one_imgaction_pair(self.imgaction_pair_list[rand_idx])

        return start_img, target_img, actions


# if __name__ == '__main__':
#     main()
