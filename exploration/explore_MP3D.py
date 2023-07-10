import torch
import numpy as np
import pickle
import habitat
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.common.baseline_registry import baseline_registry
import copy
import cv2
import random
import glob
import os
import math
from habitat_sim.utils.common import quat_from_angle_axis
import motion_task_joint_planner

class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


class DeepExplorer(object):
    def __init__(self,
                 pretrained_model=None,
                 local_video_len=10,
                 guided_step_num=10,
                 step_budget=100,
                 history_memory_len=10,
                 reverse_back_steps=6,
                 random_permutation_thred=0.2,
                 apply_action_randpermut=True,
                 safe_area_dist_thred=0.1,
                 range_sensor_steps = 10,
                 step_size = 0.25,
                 turn_angle = 10,
                 delete_imgs = True):
        self.device = torch.device(type='cuda', index=0)
        self.local_video_len = local_video_len
        self.guided_step_num = guided_step_num
        self.step_budget = step_budget
        self.history_memory_len = history_memory_len
        self.reverse_back_steps = reverse_back_steps
        self.random_permutation_thred = random_permutation_thred
        self.apply_action_randpermut = apply_action_randpermut
        self.safe_area_dist_thred = safe_area_dist_thred
        self.step_size = step_size
        self.turn_angle = turn_angle
        self.delete_imgs = delete_imgs

        self.pretrained_model = pretrained_model
        self.init_motiontask_jointplanner_model()
        self.init_video_img_list()
        self.coverage_ratio_no_distconst = list()
        self.coverage_ratio_with_distconst = list()
        self.step_num = 0
        self.meters_per_pixel = 0
        self.previous_updated_navigable_map_nodist_cons = None
        self.previous_updated_navigable_map_dist_cons = None
        self.topdown_map = None
        self.point_padding = 10
        self.point_type = [8, 8, 8]
        self.traversed_color = [1, 1, 1]
        self.valid_topdown_map = None
        self.internal_update_topdownmap = False
        self.fog_of_war_mask = None

        self.height_diff_threshold = 0.15
        self.crossfloor_reverse_steps = 3
        self.min_depth, self.max_depth = 0., 0.
        self.do_move_forward = False
        self.max_depth_all = 0.
        self.larger_than_thred_ratio = 0.
        self.range_sensor_steps = range_sensor_steps

        self.fog_of_war_mask_dict = {'update_fogwarmask': None,
                                     'update_fogwarmask_valid': None}

        self.construct_refheight_room2floors()

    def construct_refheight_room2floors(self):
        '''
        In Gibson Dataset, three rooms have two floors, we explicitly explore the two floors separately
        Internally, we construct the reference floor height
        '''
        self.floor_reference_height = dict()
        self.floor_reference_height['Mosquito'] = dict()
        self.floor_reference_height['Sands'] = dict()
        self.floor_reference_height['Scioto'] = dict()

        self.floor_reference_height['Mosquito']['floor0'] = dict()
        self.floor_reference_height['Mosquito']['floor1'] = dict()
        self.floor_reference_height['Mosquito']['floor0']['refheight'] = 0.08206
        self.floor_reference_height['Mosquito']['floor0']['need_internal_update'] = False
        self.floor_reference_height['Mosquito']['floor0']['step_budget'] = 800
        self.floor_reference_height['Mosquito']['floor1']['refheight'] = -1.96509
        self.floor_reference_height['Mosquito']['floor1']['need_internal_update'] = True
        self.floor_reference_height['Mosquito']['floor1']['step_budget'] = 200

        self.floor_reference_height['Sands']['floor0'] = dict()
        self.floor_reference_height['Sands']['floor1'] = dict()
        self.floor_reference_height['Sands']['floor0']['refheight'] = 1.599
        self.floor_reference_height['Sands']['floor0']['need_internal_update'] = False
        self.floor_reference_height['Sands']['floor0']['step_budget'] = 1000
        self.floor_reference_height['Sands']['floor1']['refheight'] = -1.0
        self.floor_reference_height['Sands']['floor1']['need_internal_update'] = True
        self.floor_reference_height['Sands']['floor1']['step_budget'] = 1000

        self.floor_reference_height['Scioto']['floor0'] = dict()
        self.floor_reference_height['Scioto']['floor1'] = dict()
        self.floor_reference_height['Scioto']['floor0']['refheight'] = 3.12
        self.floor_reference_height['Scioto']['floor0']['need_internal_update'] = False
        self.floor_reference_height['Scioto']['floor0']['step_budget'] = 1000
        self.floor_reference_height['Scioto']['floor1']['refheight'] = 0.125
        self.floor_reference_height['Scioto']['floor1']['need_internal_update'] = True
        self.floor_reference_height['Scioto']['floor1']['step_budget'] = 1000


    def init_explore_save_dict(self, room_name, img_save_dir ):
        self.explore_result_dict = dict()
        self.explore_result_dict['img_save_dir'] = img_save_dir
        self.explore_result_dict['room_name'] = room_name
        self.explore_result_dict['pano_img_list'] = list()
        self.explore_result_dict['action_list'] = list()
        self.explore_result_dict['coverage_ratio_no_distconst'] = list()
        self.explore_result_dict['coverage_ratio_with_distconst'] = list()
        self.explore_result_dict['pos_list'] = list()
        self.explore_result_dict['rot_list'] = list()
        self.explore_result_dict['meters_per_pixel'] = 0
        self.explore_result_dict['all_navigable_pixels'] = 0
        self.explore_result_dict['start_pos'] = None
        self.explore_result_dict['start_rot'] = None

    def init_explore_info(self, room_name, img_save_dir):
        self.init_explore_save_dict( room_name, img_save_dir )
        self.init_video_img_list()
        self.init_step_num()
        self.reset_meters_per_pixel()
        self.reset_previous_navigable_map()


    def init_video_img_list(self):
        self.video_img_list = list()

    def init_step_num(self):
        self.step_num = 0

    def reset_meters_per_pixel(self):
        self.meters_per_pixel = 0

    def set_meters_per_pixel(self, meters_per_pixel):
        self.meters_per_pixel = meters_per_pixel

    def reset_previous_navigable_map(self):
        self.previous_updated_navigable_map_nodist_cons = None
        self.previous_updated_navigable_map_dist_cons = None

    def update_previous_navigable_map(self,
                                      previous_updated_navigable_map_nodist_cons,
                                      previous_updated_navigable_map_dist_cons):
        self.previous_updated_navigable_map_nodist_cons = previous_updated_navigable_map_nodist_cons
        self.previous_updated_navigable_map_dist_cons = previous_updated_navigable_map_dist_cons

    def compute_meters_per_pixel(self, env, map_resolution=1024):
        lower_bound, upper_bound = env.habitat_env.sim.pathfinder.get_bounds()
        return min(
            abs(upper_bound[coord] - lower_bound[coord]) / map_resolution
            for coord in [0, 2])

    def compute_all_navigable_area(self, info, meters_per_pixel, in_pixel=True):
        MAP_INVALID_POINT = 0
        raw_topdown_map = info['top_down_map']['map']
        raw_topdown_map = raw_topdown_map.astype(np.int32)

        navigable_pixel_num = np.sum(raw_topdown_map != MAP_INVALID_POINT)

        assert navigable_pixel_num > 0

        if in_pixel:
            return navigable_pixel_num
        else:
            return navigable_pixel_num * meters_per_pixel * meters_per_pixel

    def compute_covered_area(self,
                             info,
                             meters_per_pixel,
                             previous_updated_navigable_map_nodist_cons,
                             previous_updated_navigable_map_dist_cons,
                             COVERED_POINT_ID=1,
                             view_dist_constraint=3.2, ):
        '''
        compute the covered area (in squared meters unit) up to current step.
        :param info: info obtained by the agent
        :param meters_per_pixel: pre-computed meters each pixel's length
        :param previous_updated_navigable_map: previously computed navigable map, [0,1],
            with 1 indicates already-traversed and 0 non-traversed.
        :param MAP_VALID_POINT: the symbol value indicating the traversed value
        :param view_dist_constraint: distance in meters indicating the view distance in consideration,
            which is taken as the Active Topological SLAM paper
        :param apply_view_dist_constraint: if to apply view_dist_constriant
        :return: traversed areas up to date
        '''
        raw_topdown_map = info['top_down_map']['fog_of_war_mask']  # 1: covered, 0: non-covered
        raw_topdown_map = raw_topdown_map.astype(np.int32)

        #the raw topdown map should the exclude the difference betwen update_to_date/update_valid fogwarmark
        fogwarmask_exclude = self.fog_of_war_mask_dict['update_fogwarmask'] != self.fog_of_war_mask_dict['update_fogwarmask_valid']
        fogwarmask_exclude = fogwarmask_exclude.astype(np.int32)

        raw_topdown_map -= fogwarmask_exclude
        raw_topdown_map = raw_topdown_map.astype(np.int32)
        raw_topdown_map = np.clip(raw_topdown_map, a_min=0, a_max=1)

        # step1: compute no view dist constraint
        current_covered_map = raw_topdown_map == COVERED_POINT_ID
        current_covered_map = current_covered_map.astype(np.int32)

        updated_covered_map_nodist_cons = current_covered_map | previous_updated_navigable_map_nodist_cons
        covered_pixel_num_nodist_cons = np.sum(updated_covered_map_nodist_cons == 1)

        # step2: compute with view dist constraint
        pixel_num_constrain = int(view_dist_constraint / meters_per_pixel + 0.5)
        agent_loc = info["top_down_map"]['agent_map_coord']
        agent_loc_row = agent_loc[0]
        agent_loc_col = agent_loc[1]
        agent_loc_row_start = max(0, agent_loc_row - pixel_num_constrain)
        agent_loc_row_end = min(raw_topdown_map.shape[0],agent_loc_row + pixel_num_constrain)
        agent_loc_col_start = max(0, agent_loc_col - pixel_num_constrain)
        agent_loc_col_end = min(raw_topdown_map.shape[1],agent_loc_col + pixel_num_constrain)

        view_dist_map = np.zeros(shape=raw_topdown_map.shape, dtype=np.int32)
        view_dist_map[agent_loc_row_start:agent_loc_row_end,agent_loc_col_start:agent_loc_col_end] = 1

        current_covered_map = current_covered_map & view_dist_map

        updated_covered_map_dist_cons = current_covered_map | previous_updated_navigable_map_dist_cons
        covered_pixel_num_dist_cons = np.sum(updated_covered_map_dist_cons == 1)

        return covered_pixel_num_nodist_cons, \
               updated_covered_map_nodist_cons, \
               covered_pixel_num_dist_cons, \
               updated_covered_map_dist_cons

    def init_motiontask_jointplanner_model(self):
        motiontask_joint_planner = motion_task_joint_planner.MotionTaskJointPlanner()
        pretrained_model = torch.load(self.pretrained_model)

        model_state = motiontask_joint_planner.state_dict()
        model_state.update(pretrained_model['model'])
        motiontask_joint_planner.load_state_dict(model_state)

        motiontask_joint_planner.to(device=self.device)
        motiontask_joint_planner.eval()

        self.explorer_model = motiontask_joint_planner

    def convert_rotation2dict(self, rotation):
        imag = rotation.imag
        real = rotation.real

        rotation_dict = dict()
        rotation_dict['imag'] = [str(imag[0]),
                                 str(imag[1]),
                                 str(imag[2])]

        rotation_dict['real'] = str(real)

        return rotation_dict

    def get_topdownmap_wrt_height(self, pathfinder,
                                  height,
                                  map_resolution=1024,
                                  draw_border=True,
                                  meters_per_pixel=None):
        topdown_map = maps.get_topdown_map(pathfinder,
                                           height,
                                           map_resolution,
                                           draw_border,
                                           meters_per_pixel)

        return topdown_map

    def collid_run_local_range_sensor(self, env ):
        '''
        for the agent is about to take "forward" action, sometimes it is unnecessay
        to do so because it sometimes navigates the agent to local stuck and too much time colliding with
        the wall.
        :param env: habitat lab environment
        :param step_num: the step number to take
        :return: a boolean indicating it is desirable to moveforward, the agent will be reset to the start position
        and orientation
        '''
        initial_pos = env.habitat_env.sim.get_agent_state().position
        initial_rot = env.habitat_env.sim.get_agent_state().rotation

        try_step_num = 0
        collid_with_wall = False

        while try_step_num <= self.range_sensor_steps:
            try_action = 1
            observations, reward, done, info = env.step(try_action)
            if env.habitat_env.sim.previous_step_collided:
                collid_with_wall = True
                break
            try_step_num += 1

        env.habitat_env.sim.set_agent_state(position=initial_pos,
                                            rotation=initial_rot)


        return collid_with_wall

    def update_fogofwar_map(self, info):
        current_fogofwar_map = info['top_down_map']['fog_of_war_mask'].astype(np.int32)
        fogofwar_map_valid_increase = self.fog_of_war_mask_dict['update_fogwarmask'] != current_fogofwar_map

        self.fog_of_war_mask_dict['update_fogwarmask_valid'] += fogofwar_map_valid_increase

        self.fog_of_war_mask_dict['update_fogwarmask_valid'] = current_fogofwar_map


    def execute_an_action(self,
                          env,
                          best_action,
                          cube2equirec_rgb,
                          cube2equirec_depth,
                          panoimg_save_dir):
        '''
        execute an input action and update the self.explore_result_dict
        :param env: habitat-lab env
        :param action: the action to execute, int
        :return: None, update internally
        '''
        observations, reward, done, info = env.step(best_action)

        if self.internal_update_topdownmap:
            self.update_topdown_map(env.habitat_env.sim.get_agent_state().position, env)
            info['top_down_map']['map'] = copy.deepcopy(self.topdown_map)

            info['top_down_map']['fog_of_war_mask'] = self.fog_of_war_mask

        if self.step_num == 0:
            previous_updated_navigable_map_nodist_cons = np.zeros(
                shape=info['top_down_map']['map'].shape[0:2],
                dtype=np.int32)
            previous_updated_navigable_map_dist_cons = np.zeros(
                shape=info['top_down_map']['map'].shape[0:2],
                dtype=np.int32)

            self.fog_of_war_mask_dict['update_fogwarmask'] = info['top_down_map']['fog_of_war_mask']
            self.fog_of_war_mask_dict['update_fogwarmask_valid'] = info['top_down_map']['fog_of_war_mask']

            self.update_previous_navigable_map(previous_updated_navigable_map_nodist_cons,
                                               previous_updated_navigable_map_dist_cons)

        self.update_fogofwar_map(info)

        all_navigable_area = self.compute_all_navigable_area(info, self.meters_per_pixel)

        covered_pixel_num_nodist_cons, updated_covered_map_nodist_cons, covered_pixel_num_dist_cons, updated_covered_map_dist_cons = \
            self.compute_covered_area(info,
                                      self.meters_per_pixel,
                                      self.previous_updated_navigable_map_nodist_cons,
                                      self.previous_updated_navigable_map_dist_cons)

        previous_updated_navigable_map_nodist_cons = updated_covered_map_nodist_cons
        previous_updated_navigable_map_dist_cons = updated_covered_map_dist_cons
        self.update_previous_navigable_map(previous_updated_navigable_map_nodist_cons,previous_updated_navigable_map_dist_cons)

        coverage_nodist_cons = float(covered_pixel_num_nodist_cons) / all_navigable_area
        coverage_dist_cons = float(covered_pixel_num_dist_cons) / all_navigable_area

        self.coverage_ratio_no_distconst.append(coverage_nodist_cons)
        self.coverage_ratio_with_distconst.append(coverage_dist_cons)

        self.explore_result_dict['coverage_ratio_no_distconst'].append(coverage_nodist_cons)
        self.explore_result_dict['coverage_ratio_with_distconst'].append(coverage_dist_cons)

        im, forward_rgb_img = self.get_panoimg_obs(cube2equirec_rgb, observations)
        img_save_basename = 'panoimg_{}.png'.format(self.step_num)

        img_savename = os.path.join(panoimg_save_dir, img_save_basename)
        cv2.imwrite(img_savename, im)
        self.explore_result_dict['pano_img_list'].append(img_savename)

        self.step_num += 1

        top_down_map = self.draw_top_down_map(info, im.shape[0])

        output_im = np.concatenate((im, top_down_map),axis=1)

        self.video_img_list.append(output_im)

        pos_tmp = env.habitat_env.sim.get_agent_state().position
        self.explore_result_dict['pos_list'].append(pos_tmp)
        rotate_tmp = env.habitat_env.sim.get_agent_state().rotation
        self.explore_result_dict['rot_list'].append(rotate_tmp)

        self.explore_result_dict['action_list'].append(best_action)
        self.explore_result_dict['all_navigable_pixels'] = all_navigable_area

    def get_next_action2execute_random(self, previous_action):
        '''
        random choose an action that is not previous action
        :param previous_action: int, the previous taken action
        :return: int, returned action
        '''
        action_corpus = [1, 2, 3]
        action_corpus.remove(previous_action)
        output_action = action_corpus[random.randint(0, 1)]

        return output_action

    def config_pano_camera_depth(self, config):
        CAMERA_NUM = 6
        orient = [
            [0, math.pi, 0],  # Back
            [-math.pi / 2, 0, 0],  # Down
            [0, 0, 0],  # Front
            [0, math.pi / 2, 0],  # Right
            [0, 3 / 2 * math.pi, 0],  # Left
            [math.pi / 2, 0, 0],  # Up
        ]
        sensor_uuids = []
        # Setup six cameras, Back, Down, Front, Left, Right, Up.
        if "DEPTH_SENSOR" in config.SIMULATOR.AGENT_0.SENSORS:
            config.SIMULATOR.DEPTH_SENSOR.ORIENTATION = orient[2]  # set front
            for camera_id in range(CAMERA_NUM):
                camera_template = f"DEPTH_{camera_id}"
                camera_config = copy.deepcopy(config.SIMULATOR.DEPTH_SENSOR)
                camera_config.ORIENTATION = orient[camera_id]

                camera_config.UUID = camera_template.lower()
                sensor_uuids.append(camera_config.UUID)
                setattr(config.SIMULATOR, camera_template, camera_config)
                config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)

        return config, sensor_uuids

    def config_pano_camera_rgb(self, config):
        CAMERA_NUM = 6
        orient = [
            [0, math.pi, 0],  # Back
            [-math.pi / 2, 0, 0],  # Down
            [0, 0, 0],  # Front
            [0, math.pi / 2, 0],  # Right
            [0, 3 / 2 * math.pi, 0],  # Left
            [math.pi / 2, 0, 0],  # Up
        ]
        sensor_uuids = []
        # Setup six cameras, Back, Down, Front, Left, Right, Up.
        if "RGB_SENSOR" in config.SIMULATOR.AGENT_0.SENSORS:
            config.SIMULATOR.RGB_SENSOR.ORIENTATION = orient[2]  # set front
            for camera_id in range(CAMERA_NUM):
                camera_template = f"RGB_{camera_id}"
                camera_config = copy.deepcopy(config.SIMULATOR.RGB_SENSOR)
                camera_config.ORIENTATION = orient[camera_id]

                camera_config.UUID = camera_template.lower()
                sensor_uuids.append(camera_config.UUID)
                setattr(config.SIMULATOR, camera_template, camera_config)
                config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)

        return config, sensor_uuids

    def get_current_obs(self, env):
        current_pos = env.habitat_env.sim.get_agent_state().position
        current_rotation = env.habitat_env.sim.get_agent_state().rotation
        current_obs = env.habitat_env.sim.get_observations_at(
            position=current_pos,
            rotation=current_rotation,
            keep_agent_at_new_pose=False)

        return current_obs

    def get_panoimg_obs(self, cube2equirec, observation):
        '''
        obtain the panorama image from temporary observation
        :param cube2equirec: function to convert cubic-image to pano image
        :param observation: the input sensor observation
        :return: an pano image
        '''
        batch = batch_obs([observation])
        trans_observations = cube2equirec(batch)
        equirect = trans_observations['rgb_0'].numpy().squeeze()
        # pano_img = equirect
        pano_img = equirect[:, :, [2, 1, 0]]

        forward_rgb_img = trans_observations['rgb_2'].numpy().squeeze()
        forward_rgb_img = forward_rgb_img[:,:,[2,1,0]]

        return pano_img, forward_rgb_img

    def get_panoimg_obs_depth(self, cube2equirec, observation):
        '''
        obtain the panorama image from temporary observation
        :param cube2equirec: function to convert cubic-image to pano image
        :param observation: the input sensor observation
        :return: an pano image
        '''
        batch = batch_obs([observation])
        trans_observations = cube2equirec(batch)
        equirect = trans_observations['depth_0'].numpy().squeeze()
        # pano_img = equirect
        # pano_img_depth = equirect[:, :, [2, 1, 0]]
        pano_img_depth = equirect
        forward_img_depth = trans_observations['depth_2'].numpy().squeeze()


        return pano_img_depth, forward_img_depth

    def visualize_depth_map(self, input_raw_depthmap ):
        '''
        To visualize input raw depthmap, which is float format, we rescale it to
        lie between [0, 255] explicitly for better visualization
        :param input_raw_depthmap: float32, [H,W]
        :return: vis_depth_map, lie in [0, 255]
        '''
        # depthmap = input_raw_depthmap * (255. / 0.7)
        depthmap = input_raw_depthmap * (255. / 7)
        depthmap = depthmap.astype(np.uint8)
        depthmap = np.clip(depthmap, a_min=0, a_max=255)

        return depthmap

    def draw_top_down_map(self, info, output_size):
        return maps.colorize_draw_agent_and_fit_to_height(info["top_down_map"], output_size)

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

    def prepare_history_memory_feature(self, history_memory_img_list):
        evenly_sampled_imgs = list()
        sample_step = len(history_memory_img_list) / float(
            self.history_memory_len)

        for sample_idx in range(self.history_memory_len):
            imgid2sample = int(sample_idx * sample_step)
            imgid2sample = len(
                history_memory_img_list) - 1 if imgid2sample >= len(
                history_memory_img_list) else imgid2sample
            evenly_sampled_imgs.append(history_memory_img_list[imgid2sample])

        preprocessed_img_input = list()
        for row_id, img_filename in enumerate(evenly_sampled_imgs):
            input_img_tmp = self.prepare_an_img(img_filename)
            preprocessed_img_input.append(input_img_tmp)

        history_memory_imginput = torch.stack(preprocessed_img_input,
                                              dim=0)

        history_memory_feat = self.motion_planner.get_current_obs_embed(
            history_memory_imginput)
        history_memory_feat = torch.squeeze(history_memory_feat)
        history_memory_feat = torch.mean(history_memory_feat, dim=0,
                                         keepdim=False)

        return history_memory_feat


    def map_action(self, input_action_list):
        '''
        Due to the fact that local motion predicted action index is different from
        the action index used by habitat-lab. We have explicitly map it.
        MotionPlanner: 0: MoveForward, 1: Turn-Left, 2: Turn-Right, 3: Stop
        Habitat-Lab:   0: Stop,        1: MoveForward,2: Turn-Left, 3: TurnRight
        :param input_action_list: action list predicted by local motion planner
        :return: mapped action list
        '''
        action_mapper = np.zeros([4], dtype=np.int32)
        action_mapper[0] = 1
        action_mapper[1] = 2
        action_mapper[2] = 3
        action_mapper[3] = 0

        output_action = action_mapper[input_action_list]

        return output_action

    def prepare_img_batch(self, input_img_filename_list):
        '''
        given a list of input images, prepare them to a batch input
        :param input_img_filename_list: input image name list
        :return: a torch tensor, [img_num, channel_num, height, width]
        '''
        preprocessed_img_input = list()
        for row_id, img_filename in enumerate(input_img_filename_list):
            input_img_tmp = self.prepare_an_img(img_filename)
            preprocessed_img_input.append(input_img_tmp)

        batch_input_imgs = torch.stack(preprocessed_img_input, dim=0)

        return batch_input_imgs

    def prepare_history_memory_input_img_list(self, history_memory_img_list):
        evenly_sampled_imgs = list()
        sample_step = len(history_memory_img_list) / float(self.history_memory_len)

        for sample_idx in range(self.history_memory_len):
            imgid2sample = int(sample_idx * sample_step)
            imgid2sample = len(
                history_memory_img_list) - 1 if imgid2sample >= len(
                history_memory_img_list) else imgid2sample
            evenly_sampled_imgs.append(history_memory_img_list[imgid2sample])

        return evenly_sampled_imgs

    def call_active_topo_mapper(self, input_json_file):
        '''
        Call our proposed active topology mapper to get active list to execute
        :param input_json_file: already constructed json file, including observed panoimg list
        :return: action list to execute
        '''
        panoimg_list_len = len(input_json_file['pano_img_list'])
        local_video_img_list = input_json_file['pano_img_list'][
                               panoimg_list_len - self.local_video_len:
                               panoimg_list_len]

        history_memory_img_list = input_json_file['pano_img_list'][
                                  0:panoimg_list_len - self.local_video_len]

        sampled_history_img_list = self.prepare_history_memory_input_img_list(
            history_memory_img_list)

        history_prep_imgbatch = self.prepare_img_batch(
            sampled_history_img_list)
        history_prep_imgbatch = history_prep_imgbatch.to(device=self.device)

        # step2: prepare local video image input
        local_video_prop_img_batch = self.prepare_img_batch(
            local_video_img_list)
        local_video_prop_img_batch = local_video_prop_img_batch.to(device=self.device)

        # step3: prepare current image input
        current_obs_img_batch = self.prepare_img_batch([local_video_img_list[-1]])
        current_obs_img_batch = current_obs_img_batch.to(device=self.device)

        # step4: call pretrained model to predict the next action
        action_prob = self.explorer_model.predict_next_action(
            current_obs=current_obs_img_batch,
            local_video_input=local_video_prop_img_batch,
            history_memory_input=history_prep_imgbatch
        )

        action_prob = action_prob.detach().cpu().numpy()

        return action_prob

    def update_traversed_area(self, position, sim):
        t_x, t_y = maps.to_grid(
            position[2],
            position[0],
            self.topdown_map.shape[0:2],
            sim=sim, )

        padding_len = int(3.2 / self.meters_per_pixel + 0.5)

        x_min = t_x - padding_len
        x_min = max(0, x_min)

        x_max = t_x + padding_len
        x_max = min(x_max, self.topdown_map.shape[0] - 1)

        y_min = t_y - padding_len
        y_min = max(0, y_min)

        y_max = t_y + padding_len
        y_max = min(y_max, self.topdown_map.shape[1] - 1)

        self.topdown_map[x_min:x_max + 1, y_min:y_max + 1] = self.traversed_color[0]

        self.fog_of_war_mask[x_min:x_max + 1, y_min:y_max + 1] = 1

        self.topdown_map = np.multiply(self.topdown_map,
                                       self.valid_topdown_map).astype(np.uint8)
        self.fog_of_war_mask = np.multiply(self.fog_of_war_mask,
                                           self.valid_topdown_map).astype(np.uint8)


    def get_topdownmap_height(self, env,
                              pathfinder,
                              height,
                              map_resolution=1024,
                              draw_border=True, ):
        meters_per_pixel = self.compute_meters_per_pixel(env)
        topdown_map = maps.get_topdown_map(pathfinder,
                                           height,
                                           map_resolution,
                                           draw_border,
                                           meters_per_pixel)

        self.topdown_map = topdown_map
        self.valid_topdown_map = (topdown_map != 0).astype(np.uint8)
        self.fog_of_war_mask = np.zeros(self.topdown_map.shape,
                                        np.uint8)

    def _draw_point(self, position, sim):
        t_x, t_y = maps.to_grid(
            position[2],
            position[0],
            self.topdown_map.shape[0:2],
            sim=sim,
        )
        self.topdown_map[
        t_x - self.point_padding: t_x + self.point_padding + 1,
        t_y - self.point_padding: t_y + self.point_padding + 1] = \
            self.point_type[0]

    def update_topdown_map(self, position, env):
        self._draw_point(position, env.habitat_env.sim)
        self.update_traversed_area(position=position,
                                   sim=env.habitat_env.sim)

    def detect_cross_floor(self, env, ref_refheight):
        agent_height = float(env.habitat_env.sim.get_agent_state().position[1])

        return abs(agent_height - ref_refheight) >= self.height_diff_threshold

    def decide_if_forward_traversible(self, forward_depthmap ):
        '''
        Given the forward-looking depth image, decide if the agent need to
        move-forward in order to cover more area.
        :param forward_depthmap: 2D depth map.
        :return:
        '''
        #first double-check if the center-located area contains enough depth values that larger than
        max_depth_all = np.max(forward_depthmap)
        cfa_half_width = 25
        cfa_h_start = 70
        cfa_h_end = 200

        depth_val_thred = 3.0
        depth2explore_ratio = 0.5

        mid_width = forward_depthmap.shape[1]//2
        center_forward_area = forward_depthmap[cfa_h_start:cfa_h_end,
                              mid_width-cfa_half_width:mid_width+cfa_half_width]
        larger_than_thred_area = np.sum(center_forward_area>depth_val_thred)
        larger_than_thred_ratio = float(larger_than_thred_area)/center_forward_area.size

        if max_depth_all < depth_val_thred or larger_than_thred_ratio < depth2explore_ratio:
            do_move_forward = False
        else:
            do_move_forward = True

        self.max_depth_all = max_depth_all
        self.larger_than_thred_ratio = larger_than_thred_ratio

        #second crop the center-bottom area of the depth map for deciding the agent can
        #take the move-forward action
        width = forward_depthmap.shape[1]
        mid_width = width//2
        height = forward_depthmap.shape[0]
        crop_depthmap = forward_depthmap[height-60:height-10,mid_width-25:mid_width+25]
        min_depth = np.min(crop_depthmap)
        max_depth = np.max(crop_depthmap)

        do_move_forward = do_move_forward and min_depth > .5 and max_depth > 1.5

        return min_depth, max_depth, do_move_forward

    def initialize_N_steps(self, env, cube2equirec_rgb, cube2equirec_depth, panoimg_save_dir):
        '''
        Give the agent an initial N steps exploration
        :param env:
        :return:
        '''
        best_action = 1 #by default, we execute move-forward action
        while True:
            if self.collid_run_local_range_sensor(env):
                best_action = self.get_next_action2execute_random(1)
            self.execute_an_action(env=env,
                                   best_action=best_action,
                                   cube2equirec_rgb=cube2equirec_rgb,
                                   cube2equirec_depth=cube2equirec_depth,
                                   panoimg_save_dir=panoimg_save_dir)

            if env.habitat_env.sim.previous_step_collided:
                best_action = self.get_next_action2execute_random(1)
            else:
                best_action = 1

            if self.step_num >= self.guided_step_num:
                break


    def sanitize_explore_result(self):
        '''
        Post-process the exploration results.
        :return: None
        '''
        self.explore_result_dict['pano_img_list'] = self.explore_result_dict['pano_img_list'][0:self.step_num+1]
        self.explore_result_dict['action_list'] = self.explore_result_dict['action_list'][0:self.step_num]
        self.explore_result_dict['coverage_ratio_no_distconst'] = self.explore_result_dict['action_list'][0:self.step_num]
        self.explore_result_dict['coverage_ratio_with_distconst'] = self.explore_result_dict['coverage_ratio_with_distconst'][0:self.step_num]
        self.explore_result_dict['pos_list'] =  self.explore_result_dict['pos_list'][0:self.step_num+1]
        self.explore_result_dict['rot_list'] = self.explore_result_dict['rot_list'][0:self.step_num+1]

        self.video_img_list = self.video_img_list[0:self.step_num+1]

        if self.delete_imgs:
            for filename in os.listdir(self.explore_result_dict['img_save_dir']):
                if filename.endswith('.png'):
                    os.remove( os.path.join(self.explore_result_dict['img_save_dir'], filename))


    def get_all_navigable_pos(self, env, floor0_refheight = 0., floor1_refheight = 1. ):
        nav_init_pos = dict()
        nav_init_pos['floor0'] = list()
        nav_init_pos['floor1'] = list()

        for episode_tmp in env.episodes:
            start_pos = episode_tmp.start_position
            start_rot = episode_tmp.start_rotation

            y_ref = start_pos[1]

            floor_id = 0 if abs(floor0_refheight-y_ref) < abs(floor1_refheight - y_ref) else 1

            nav_init_pos['floor{}'.format(floor_id)].append({'pos': start_pos,
                                                             'rot': start_rot})


        assert len(nav_init_pos['floor0']) > 0
        assert len(nav_init_pos['floor1']) > 0

        return nav_init_pos

    def explore_oneroom_twofloors(self,
                                  room_json_file,
                                  img_save_dir,
                                  random_seed = 0,):
        '''
        Explore a room with just one floor
        :param room_json_file:
        :param pano_img_save_dir:
        :param random_seed:
        :param actpert_rate:
        :param floor_dict_name:
        :return:
        '''
        room_name = os.path.basename(room_json_file).replace('.json.gz', '')
        self.init_explore_info( room_name, img_save_dir )

        #update the configuration file
        config = habitat.get_config(config_paths="configs/tasks/pointnav_gibson.yaml")
        config.defrost()
        config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
        config.SIMULATOR.FORWARD_STEP_SIZE = self.step_size
        config.SIMULATOR.TURN_ANGLE = self.turn_angle
        config, sensor_uuids = self.config_pano_camera_rgb(config)
        config.freeze()

        obs_trans = baseline_registry.get_obs_transformer("CubeMap2Equirec")
        cube2equirec = obs_trans(sensor_uuids, (256, 512), 256, False, None)
        cube2equirec_depth = None

        with SimpleRLEnv(config=config) as env:
            env.seed(random_seed)
            env.reset()

            nav_init_pos = self.get_all_navigable_pos( env,
                                                       self.floor_reference_height[room_name]['floor0']['refheight'],
                                                       self.floor_reference_height[room_name]['floor1']['refheight'])

        nav_result_2floors = dict()
        nav_result_2floors['floor0'] = dict()
        nav_result_2floors['floor1'] = dict()

        for floor_id in range(0,2):
            need_internal_update = self.floor_reference_height[room_name]['floor{}'.format(floor_id)]['need_internal_update']
            self.internal_update_topdownmap = need_internal_update
            step_budget = self.floor_reference_height[room_name]['floor{}'.format(floor_id)]['step_budget']
            refheight = self.floor_reference_height[room_name]['floor{}'.format(floor_id)]['refheight']

            self.init_explore_info( room_name, img_save_dir )
            with SimpleRLEnv(config=config) as env:
                env.reset()
                env.seed(random_seed)
                self.set_meters_per_pixel(self.compute_meters_per_pixel(env))
                self.explore_result_dict['meters_per_pixel'] = self.meters_per_pixel

                #randomly set the agent at position and orientation
                init_agent_pos = random.choice(nav_init_pos['floor{}'.format(floor_id)])
                init_pos = init_agent_pos['pos']
                init_rot = init_agent_pos['rot']

                env.habitat_env.sim.set_agent_state(position=init_pos,rotation=init_rot)


                if self.internal_update_topdownmap:
                    self.get_topdownmap_height(env,
                                               pathfinder=env.habitat_env.sim.pathfinder,
                                               height=init_pos[1])

                self.explore_result_dict['start_pos'] = init_pos
                self.explore_result_dict['start_rot'] = init_rot

                current_obs = self.get_current_obs(env)
                im, forward_rgb_img = self.get_panoimg_obs(cube2equirec,observation=current_obs)

                img_save_basename = 'panoimg_{}.png'.format(self.step_num)
                img_savename = os.path.join(img_save_dir,img_save_basename)
                cv2.imwrite(img_savename, im)
                self.explore_result_dict['pano_img_list'].append(img_savename)
                pos_tmp = env.habitat_env.sim.get_agent_state().position
                self.explore_result_dict['pos_list'].append(pos_tmp)
                rot_tmp = env.habitat_env.sim.get_agent_state().rotation
                self.explore_result_dict['rot_list'].append(rot_tmp)

                #get the initial N steps exploration
                self.initialize_N_steps(env,cube2equirec,cube2equirec_depth,img_save_dir)

                while True:
                    predict_action_list = self.call_active_topo_mapper(self.explore_result_dict)
                    best_action = np.argsort(predict_action_list)[-1] + 1

                    if best_action == 1:
                        local_range_info = self.execute_local_range_sensor(env, just_forward_sense=False)
                        #don't further go forward because it will collides the wall
                        if len(local_range_info) != 1:
                            self.explore_according_local_sensor(env=env,
                                                                cube2equirec_rgb=cube2equirec,
                                                                cube2equirec_depth=cube2equirec_depth,
                                                                panoimg_save_dir=img_save_dir,
                                                                local_range_info=local_range_info)
                        else:
                            self.execute_an_action(env=env,
                                                   best_action=best_action,
                                                   cube2equirec_rgb=cube2equirec,
                                                   cube2equirec_depth=cube2equirec_depth,
                                                   panoimg_save_dir=img_save_dir)

                    else:
                        self.execute_an_action(env=env,
                                               best_action=best_action,
                                               cube2equirec_rgb=cube2equirec,
                                               cube2equirec_depth=cube2equirec_depth,
                                               panoimg_save_dir=img_save_dir)

                    if env.habitat_env.sim.previous_step_collided:
                        print('collision detected!')
                        self.execute_fine_grained_turning(env,
                                                          cube2equirec_rgb=cube2equirec,
                                                          cube2equirec_depth=cube2equirec_depth,
                                                          panoimg_save_dir=img_save_dir)

                    if self.local_stuck_detection():
                        print('local stuck detected')
                        self.execute_fine_grained_turning(env,
                                                          cube2equirec_rgb=cube2equirec,
                                                          cube2equirec_depth=cube2equirec_depth,
                                                          panoimg_save_dir=img_save_dir)

                    if self.step_num >= step_budget:
                        break

                self.sanitize_explore_result()

                with open(os.path.join(img_save_dir, 'explore_rst_floor{}.pickle'.format(floor_id)), 'wb') as handle:
                    pickle.dump(self.explore_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            nav_result_2floors['floor{}'.format(floor_id)]['start_pos'] = self.explore_result_dict['start_pos']
            nav_result_2floors['floor{}'.format(floor_id)]['start_rot'] = self.explore_result_dict['start_rot']
            nav_result_2floors['floor{}'.format(floor_id)]['action_list'] = self.explore_result_dict['action_list']

        return nav_result_2floors

    def explore_oneroom(self,
                        room_json_file,
                        img_save_dir,
                        random_seed = 0,):

        room_name = os.path.basename(room_json_file).replace('.json.gz', '')
        self.init_explore_save_dict( room_name, img_save_dir )

        #update the configuration file
        config = habitat.get_config(config_paths="configs/tasks/pointnav_gibson.yaml")
        config.defrost()
        config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
        config.SIMULATOR.FORWARD_STEP_SIZE = self.step_size
        config.SIMULATOR.TURN_ANGLE = self.turn_angle
        config, sensor_uuids = self.config_pano_camera_rgb(config)
        config.freeze()

        obs_trans = baseline_registry.get_obs_transformer("CubeMap2Equirec")
        cube2equirec = obs_trans(sensor_uuids, (256, 512), 256, False, None)
        cube2equirec_depth = None

        with SimpleRLEnv(config=config) as env:
            env.reset()
            env.seed(random_seed)
            self.set_meters_per_pixel(self.compute_meters_per_pixel(env))
            self.explore_result_dict['meters_per_pixel'] = self.meters_per_pixel

            #randomly set the agent at position and orientation
            init_pos = env.habitat_env.sim.sample_navigable_point()
            init_rot = quat_from_angle_axis(random.uniform(0, 2.0 * np.pi), np.array([0, 1, 0]))

            env.habitat_env.sim.set_agent_state(position=init_pos,rotation=init_rot)

            self.explore_result_dict['start_pos'] = init_pos
            self.explore_result_dict['start_rot'] = init_rot

            current_obs = self.get_current_obs(env)
            im, forward_rgb_img = self.get_panoimg_obs(cube2equirec,observation=current_obs)

            img_save_basename = 'panoimg_{}.png'.format(self.step_num)
            img_savename = os.path.join(img_save_dir,img_save_basename)
            cv2.imwrite(img_savename, im)
            self.explore_result_dict['pano_img_list'].append(img_savename)
            pos_tmp = env.habitat_env.sim.get_agent_state().position
            self.explore_result_dict['pos_list'].append(pos_tmp)
            rot_tmp = env.habitat_env.sim.get_agent_state().rotation
            self.explore_result_dict['rot_list'].append(rot_tmp)

            #get the initial N steps exploration
            self.initialize_N_steps(env,cube2equirec,cube2equirec_depth,img_save_dir)

            while True:
                predict_action_list = self.call_active_topo_mapper(self.explore_result_dict)
                best_action = np.argsort(predict_action_list)[-1] + 1

                if best_action == 1:
                    local_range_info = self.execute_local_range_sensor(env, just_forward_sense=False)
                    #don't further go forward because it will collides the wall
                    if len(local_range_info) != 1:
                        self.explore_according_local_sensor(env=env,
                                                            cube2equirec_rgb=cube2equirec,
                                                            cube2equirec_depth=cube2equirec_depth,
                                                            panoimg_save_dir=img_save_dir,
                                                            local_range_info=local_range_info)
                    else:
                        self.execute_an_action(env=env,
                                               best_action=best_action,
                                               cube2equirec_rgb=cube2equirec,
                                               cube2equirec_depth=cube2equirec_depth,
                                               panoimg_save_dir=img_save_dir)

                else:
                    self.execute_an_action(env=env,
                                           best_action=best_action,
                                           cube2equirec_rgb=cube2equirec,
                                           cube2equirec_depth=cube2equirec_depth,
                                           panoimg_save_dir=img_save_dir)


                if env.habitat_env.sim.previous_step_collided:
                    self.execute_fine_grained_turning(env,
                                                      cube2equirec_rgb=cube2equirec,
                                                      cube2equirec_depth=cube2equirec_depth,
                                                      panoimg_save_dir=img_save_dir)

                if self.local_stuck_detection():
                    self.execute_fine_grained_turning(env,
                                                      cube2equirec_rgb=cube2equirec,
                                                      cube2equirec_depth=cube2equirec_depth,
                                                      panoimg_save_dir=img_save_dir)

                if self.step_num >= self.step_budget:
                    break

            self.sanitize_explore_result()

            with open(os.path.join(img_save_dir, 'explore_rst.pickle'), 'wb') as handle:
                pickle.dump(self.explore_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


            return self.explore_result_dict['start_pos'], self.explore_result_dict['start_rot'], self.explore_result_dict['action_list']

    def gen_exploration_video(self, room_json_file, start_pos, start_rot, action_list, img_save_dir, save_basename = None ):
        room_name = os.path.basename(room_json_file).replace('.json.gz', '')

        #update the configuration file
        config = habitat.get_config(config_paths="configs/tasks/pointnav_gibson.yaml")
        config.defrost()
        config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
        config.SIMULATOR.FORWARD_STEP_SIZE = self.step_size
        config.SIMULATOR.TURN_ANGLE = self.turn_angle
        config, sensor_uuids = self.config_pano_camera_rgb(config)
        config, sensor_uuids_depth = self.config_pano_camera_depth(config)
        config.freeze()

        obs_trans = baseline_registry.get_obs_transformer("CubeMap2Equirec")
        cube2equirec_rgb = obs_trans(sensor_uuids, (256, 512), 256, False, None)

        video_img_list = list()

        with SimpleRLEnv(config=config) as env:
            env.reset()
            env.habitat_env.sim.set_agent_state(position=start_pos,
                                                rotation=start_rot)

            for action_tmp in action_list:
                observations, reward, done, info = env.step(action_tmp)
                im, forward_rgb_img = self.get_panoimg_obs(cube2equirec_rgb,
                                                           observations)

                top_down_map = self.draw_top_down_map(info, im.shape[0])

                output_im = np.concatenate((im, top_down_map), axis=1)

                video_img_list.append(output_im)

        print('writing the exploration video')

        images_to_video(video_img_list, img_save_dir, "{}_{}".format(save_basename,room_name))


    def local_stuck_detection(self):
        '''
        We explicitly add local stuck detection so as to help the agent to jump out the local stuck
        :return: boolean, if local stuck happens
        '''
        last_end_idx = 12 if len(self.explore_result_dict['action_list']) > 12 else len(self.explore_result_dict['action_list'])
        last_end_idx = -1*last_end_idx
        lastacts = self.explore_result_dict['action_list'][last_end_idx:]

        lastacts = np.array(lastacts, np.int32)

        return np.sum(lastacts==1) == 0

    def split_episode_list(self, env, floor_info):
        '''
        split the given episode list into two floors, according to the pre-defined
        floor height values
        :param env: input environment
        :return: a dict with two floor navigation list
        '''
        floor_episode_dict = dict()
        floor_episode_dict[0] = list()
        floor_episode_dict[1] = list()
        for episode_tmp in env.episodes:
            anchor_height = episode_tmp.start_position[1]

            if abs(anchor_height-floor_info[0]['refheight']) < abs(anchor_height-floor_info[1]['refheight']):
                floor_episode_dict[0].append(episode_tmp)
            else:
                floor_episode_dict[1].append(episode_tmp)

        return floor_episode_dict

    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians

    def execute_local_range_sensor(self, input_env, just_forward_sense = False):
        '''
        :param input_env: the initial habitat-sim environment
        :param initial_pos: the initial position
        :param initial_rot: the initial rotation information
        :return:
        '''
        init_pos_input = input_env.habitat_env.sim.get_agent_state().position
        init_rot_input = input_env.habitat_env.sim.get_agent_state().rotation

        range_rst = []

        MAX_STEPS = 10
        #step1: execute forward
        execute_forward_times = 0
        while execute_forward_times < MAX_STEPS:
            init_pos = input_env.habitat_env.sim.get_agent_state().position
            obs_tmp, reward, done, info = input_env.step(1)
            init_pos = np.array(init_pos, np.float32)
            then_pos = input_env.habitat_env.sim.get_agent_state().position
            then_pos = np.array(then_pos, np.float32)
            pos_displacement = np.sqrt(np.sum(np.square(init_pos-then_pos)))
            if input_env.habitat_env.sim.previous_step_collided or pos_displacement < self.step_size/2.:
                break
            execute_forward_times += 1

        range_rst.append( execute_forward_times )

        circular_sense = execute_forward_times <= 1
        if circular_sense and (not just_forward_sense):
            #since turn-angle is 30 degree, so we rotate 11 times to cover a circle
            for iter_idx in range(6):
                #step2: iterate counter-clockwisely, constantly turn left
                input_env.habitat_env.sim.set_agent_state(position=init_pos_input,rotation=init_rot_input)

                #turn to the designated direction
                for i in range(iter_idx+1):
                    for turn_time in range(30//self.turn_angle):
                        obs_tmp, reward, done, info = input_env.step(2)
                execute_forward_times = 0
                while execute_forward_times < MAX_STEPS:
                    init_pos = input_env.habitat_env.sim.get_agent_state().position
                    obs_tmp, reward, done, info = input_env.step(1)
                    init_pos = np.array(init_pos, np.float32)
                    then_pos = input_env.habitat_env.sim.get_agent_state().position
                    then_pos = np.array(then_pos, np.float32)
                    pos_displacement = np.sqrt(np.sum(np.square(init_pos - then_pos)))
                    if input_env.habitat_env.sim.previous_step_collided or pos_displacement < self.step_size/2.:
                        break
                    execute_forward_times += 1
                range_rst.append( execute_forward_times )

            for iter_idx in range(5):
                #step2: iterate counter-clockwisely, constantly turn left
                input_env.habitat_env.sim.set_agent_state(position=init_pos_input,rotation=init_rot_input)
                for i in range(iter_idx+1):
                    for turn_time in range(30//self.turn_angle):
                        obs_tmp, reward, done, info = input_env.step(3)
                execute_forward_times = 0
                while execute_forward_times < MAX_STEPS:
                    init_pos = input_env.habitat_env.sim.get_agent_state().position
                    obs_tmp, reward, done, info = input_env.step(1)
                    init_pos = np.array(init_pos, np.float32)
                    then_pos = input_env.habitat_env.sim.get_agent_state().position
                    then_pos = np.array(then_pos, np.float32)
                    pos_displacement = np.sqrt(np.sum(np.square(init_pos - then_pos)))
                    if input_env.habitat_env.sim.previous_step_collided or pos_displacement < self.step_size/2.:
                        break
                    execute_forward_times += 1
                range_rst.append( execute_forward_times )

        self.fog_of_war_mask_dict['update_fogwarmask'] = info['top_down_map']['fog_of_war_mask']

        input_env.habitat_env.sim.set_agent_state(position=init_pos_input, rotation=init_rot_input)

        return range_rst

    def execute_turn_action(self, env, action, turn_times = 1,
                            append_action = 1,
                            cube2equirec_rgb = None,
                            cube2equirec_depth = None,
                            panoimg_save_dir = None):
        '''
        When the agent will collide the wall, we usually execute turn-left/right action, usually
        we append an extra action after executing the turn-left/right action
        :param env:
        :param action:
        :param turn_times:
        :param append_action:
        :return:
        '''
        for i in range(turn_times):
            self.execute_an_action(env=env,
                                   best_action=action,
                                   cube2equirec_rgb=cube2equirec_rgb,
                                   cube2equirec_depth=cube2equirec_depth,
                                   panoimg_save_dir=panoimg_save_dir)

        self.execute_an_action(env=env, best_action=append_action,
                               cube2equirec_rgb=cube2equirec_rgb,
                               cube2equirec_depth=cube2equirec_depth,
                               panoimg_save_dir=panoimg_save_dir)

    def execute_fine_grained_turning(self, env, cube2equirec_rgb, cube2equirec_depth, panoimg_save_dir):
        '''
        in some cases, we randomly choose a turning action, then gradually execute this action that can potentially
        navigates the agent to an open area
        :param env:  input environment
        :return: None, all info is internally updated
        '''

        best_action = self.get_next_action2execute_random(1)  # turn-right/turn-left
        # keep executing the chosen action until the agent is able to head to a safe
        rotate_times = 0
        while rotate_times < 360//self.turn_angle:
            self.execute_an_action(env=env,
                                   best_action=best_action,
                                   cube2equirec_rgb=cube2equirec_rgb,
                                   cube2equirec_depth=cube2equirec_depth,
                                   panoimg_save_dir=panoimg_save_dir)

            range_result = self.execute_local_range_sensor( env, just_forward_sense=True )
            if range_result[0] >= 1:
                best_action = 1
                break
            rotate_times += 1

        self.execute_an_action(env=env,
                               best_action=best_action,
                               cube2equirec_rgb=cube2equirec_rgb,
                               cube2equirec_depth=cube2equirec_depth,
                               panoimg_save_dir=panoimg_save_dir)

    def select_nextact_from_localsense(self, local_range_info):
        '''
        Given a list of range-sensing result, we select one direction that the agent should probably navigate to
        :param local_range_info: a list of range sense results
        :return: the index of most probable direction to turn to
        '''
        FORWARD_HALF_THRESHOLD = 4

        forward_half_info = [[local_range_info[0], local_range_info[1], local_range_info[2], local_range_info[3],
                              local_range_info[7], local_range_info[8], local_range_info[9]],[0,1,2,3,7,8,9]]

        forward_half_info = np.array(forward_half_info, np.int32)

        max_forward_steps = np.max(forward_half_info[0,:])

        #just allow the agent to move in the forward half area
        if max_forward_steps >= FORWARD_HALF_THRESHOLD:
            max_step = np.max(forward_half_info[0,:])
            max_step_idx = np.where(forward_half_info[0,:] == max_step)[0]
            max_step_idx = list(max_step_idx)
            random.shuffle(max_step_idx)

            assert len(max_step_idx) >= 1

            return forward_half_info[1,:][max_step_idx[0]]

        ordered_index = list(np.arange(len(local_range_info)))
        random.shuffle(ordered_index)

        local_range_info_shuffle = list(np.array(local_range_info,np.int32)[ordered_index])

        return ordered_index[local_range_info_shuffle.index(max(local_range_info_shuffle))]


    def explore_according_local_sensor(self, env, cube2equirec_rgb, cube2equirec_depth, panoimg_save_dir, local_range_info ):
        # don't further go forward because it will collides the wall
        if len(local_range_info) != 1:
            # max_depth_idx = local_range_info.index(max(local_range_info))
            max_depth_idx = self.select_nextact_from_localsense(local_range_info)

            # disaster happens, needs to call fine-grained collision-avoidance
            if max_depth_idx == 0:
                self.execute_fine_grained_turning(env,
                                                  cube2equirec_rgb=cube2equirec_rgb,
                                                  cube2equirec_depth=cube2equirec_depth,
                                                  panoimg_save_dir=panoimg_save_dir)

            if max_depth_idx <= 6:
                self.execute_turn_action(env, action=2,
                                         turn_times=max_depth_idx * (30//self.turn_angle),
                                         append_action=1,
                                         cube2equirec_rgb=cube2equirec_rgb,
                                         cube2equirec_depth=cube2equirec_depth,
                                         panoimg_save_dir=panoimg_save_dir)

            elif max_depth_idx > 6:
                self.execute_turn_action(env, action=3,
                                         turn_times=(max_depth_idx - 6) * (30//self.turn_angle),
                                         append_action=1,
                                         cube2equirec_rgb=cube2equirec_rgb,
                                         cube2equirec_depth=cube2equirec_depth,
                                         panoimg_save_dir=panoimg_save_dir)

def main():
    # global parameter configuration
    step_size = 0.25
    turn_angle = 10
    delete_imgs = True

    motiontask_jointplanner_pretrained_model = 'model_epoch_70.pth'
    input_room_list = glob.glob('./data/datasets/pointnav/mp3d/v1/test/content/*.json.gz')

    assert len(input_room_list) > 0

    random_seed_filename = 'seed_list.txt'
    random_seeds = [int(line_tmp.rstrip('\n')) for line_tmp in open(random_seed_filename, 'r').readlines()]

    for seed_num_idx in range(len(random_seeds)):
        for idx, input_room in enumerate(input_room_list):
            roomname = os.path.basename(input_room).replace('.json.gz', '')
            print('processing room {}'.format(roomname))

            output_dir = 'output/{}_{}'.format(roomname, random_seeds[seed_num_idx])
            os.makedirs(output_dir) if not os.path.exists(output_dir) else None

            deep_explorer = DeepExplorer(
                pretrained_model=motiontask_jointplanner_pretrained_model,
                step_size=step_size,
                turn_angle=turn_angle,
                delete_imgs=delete_imgs)
            _, _, _, = deep_explorer.explore_oneroom_onefloor(input_room,
                                                              output_dir,
                                                              random_seed=random_seeds[seed_num_idx])

        print('Done!')

if __name__ == '__main__':
    main()
