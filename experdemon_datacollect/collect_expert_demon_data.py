import os
import math
import copy
import numpy as np
import glob
import json
import sklearn
import cv2

import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.common.baseline_registry import baseline_registry


def get_panoimg_obs(cube2equirec, observation):
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

    return pano_img

def config_pano_camera_rgb(config):
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


class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def get_current_obs(env):
    current_pos = env.habitat_env.sim.get_agent_state().position
    current_rotation = env.habitat_env.sim.get_agent_state().rotation
    current_obs = env.habitat_env.sim.get_observations_at(position=current_pos,
                                                          rotation=current_rotation,
                                                          keep_agent_at_new_pose=False)

    return current_obs


def check_if_close(stored_control_pts, new_control_pt, dist_thred=1.):
    '''
    double-check if the new_control_pt lies too close with the stored control pts
    :param stored_control_pts: [N, 3], float64
    :param new_control_pt: [3], float64
    :return: a boolean,
    '''
    if len(new_control_pt.shape) == 1:
        new_control_pt = np.reshape(new_control_pt, newshape=(1, -1))

    dist = sklearn.metrics.pairwise_distances(stored_control_pts,
                                              new_control_pt)
    dist = np.squeeze(dist)

    return np.any(dist < dist_thred)


def get_episode_list(env):
    #explicitly remove too close episode points
    navigable_points = list()
    navigable_points.append(env.episodes[0].start_position)
    navigable_points.append(env.episodes[0].goals[0].position)

    for episode_id in range(1, len(env.episodes)):
        start_position = env.episodes[episode_id].start_position
        target_position = env.episodes[episode_id].goals[0].position
        if not check_if_close(np.array(navigable_points, np.float64),
                              np.array(start_position, np.float64)):
            navigable_points.append(start_position)
        if not check_if_close(np.array(navigable_points, np.float64),
                              np.array(target_position, np.float64)):
            navigable_points.append(target_position)

    assert len(navigable_points) > 0

    return navigable_points


def sort_anchor_points(input_anchor_list,
                           env,
                           neighbor_dist_thred=3.,
                           search_neighor=True):
    '''
    choose the first point, as the start_point, the recursively retrieve the
    nearest point as the next point to traverse, the distance to choose is geodesic
    distance here.
    :param input_control_list: control points
    :param env: habitat-sim environment
    :return: sorted input_control_list
    '''
    # step1: compute  anchor point pair geodesic distance
    geodist = 1000 * np.ones(
        shape=(len(input_anchor_list), len(input_anchor_list)),
        dtype=np.float32)
    for row_idx in range(geodist.shape[0]):
        for col_idx in range(row_idx + 1, geodist.shape[1]):
            ref_pos = input_anchor_list[row_idx]
            goal_pos = input_anchor_list[col_idx]
            geodist_tmp = env.habitat_env.sim.geodesic_distance(
                position_a=ref_pos,
                position_b=goal_pos)
            geodist[row_idx, col_idx] = geodist_tmp
            geodist[col_idx, row_idx] = geodist_tmp

    sorted_control_list = list()
    sorted_control_list.append(input_anchor_list[0])

    traversed_idx = [0]
    ref_idx = 0

    if not search_neighor:
        while True:
            if len(sorted_control_list) == len(input_anchor_list):
                break
            ref_dist_vec = geodist[ref_idx, :]
            sorted_idx = np.argsort(ref_dist_vec)
            for next_id in sorted_idx:
                if not next_id in traversed_idx:
                    sorted_control_list.append(input_anchor_list[next_id])
                    traversed_idx.append(next_id)
                    ref_idx = next_id
                    break
    else:
        while True:
            if len(sorted_control_list) == len(input_anchor_list):
                break
            ref_dist_vec = geodist[ref_idx, :]
            sorted_idx = np.argsort(ref_dist_vec)
            find_neighbor = False
            for next_id in sorted_idx:
                if (next_id not in traversed_idx) and (
                    ref_dist_vec[next_id] <= neighbor_dist_thred):
                    sorted_control_list.append(input_anchor_list[next_id])
                    if len(sorted_control_list) == len(input_anchor_list):
                        break
                    traversed_idx.append(next_id)
                    ref_idx = next_id
                    find_neighbor = True

            if not find_neighbor:
                while True:
                    if len(sorted_control_list) == len(input_anchor_list):
                        break
                    ref_dist_vec = geodist[ref_idx, :]
                    sorted_idx = np.argsort(ref_dist_vec)
                    for next_id in sorted_idx:
                        if not next_id in traversed_idx:
                            sorted_control_list.append(input_anchor_list[next_id])
                            traversed_idx.append(next_id)
                            ref_idx = next_id
                            break

    return sorted_control_list


def collect_expert_demon_data(step_size, turn_angle, save_dir):
    config_filename_list = glob.glob('.data/datasets/pointnav/gibson/v1/train/content/*.json.gz')

    for json_filename in config_filename_list:
        config = habitat.get_config(config_paths="configs/tasks/pointnav_gibson.yaml")
        config.defrost()
        config.DATASET.DATA_PATH = json_filename
        config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        config, sensor_uuids = config_pano_camera_rgb(config)
        config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
        config.SIMULATOR.FORWARD_STEP_SIZE = step_size
        config.SIMULATOR.TURN_ANGLE = turn_angle
        config.freeze()

        room_name = os.path.basename(json_filename).replace('.json.gz', '')
        save_dir_tmp = os.path.join(save_dir, room_name)
        os.makedirs(save_dir_tmp) if not os.path.exists(save_dir_tmp) else None

        nav_dict = dict()
        nav_dict['room'] = room_name
        nav_dict['action_list'] = list()
        nav_dict['panoimg_list'] = list()

        with SimpleRLEnv(config=config) as env:
            env.reset()
            goal_radius = env.episodes[0].goals[0].radius
            if goal_radius is None:
                goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
            print('obtaining the anchor points...')
            anchor_pos_list = get_episode_list(env)
            sorted_anchor_pos_list = sort_anchor_points(anchor_pos_list, env)

            follower = ShortestPathFollower(env.habitat_env.sim,goal_radius,False)

            obs_trans = baseline_registry.get_obs_transformer("CubeMap2Equirec")

            cube2equirec = obs_trans(sensor_uuids, (256, 512), 256, False,None)

            # first put the agent at the first anchor point
            init_rotation = env.habitat_env.sim.get_agent_state().rotation
            target_position = np.array(sorted_anchor_pos_list[0],np.float32)
            env.habitat_env.sim.set_agent_state(position=target_position, rotation=init_rotation)

            current_obs = get_current_obs(env)
            im = get_panoimg_obs(cube2equirec, observation=current_obs)
            img_save_basename = 'panoimg_{}.png'.format(0)
            img_savename = os.path.join(save_dir_tmp, img_save_basename)
            assert cv2.imwrite(img_savename, im)
            nav_dict['panoimg_list'].append(img_savename)

            # begin exploration
            step_id = 1
            for anchor_id, anchor_pos in enumerate(sorted_anchor_pos_list[1:]):
                while True:
                    best_action = follower.get_next_action(anchor_pos)

                    if best_action == 0: #reached the goal
                        break

                    nav_dict['action_list'].append(best_action)
                    observations, reward, done, info = env.step(best_action)

                    im = get_panoimg_obs(cube2equirec, observations)
                    img_save_basename = 'panoimg_{}.png'.format(step_id)
                    img_savename = os.path.join(save_dir_tmp, img_save_basename)
                    assert  cv2.imwrite(img_savename, im)
                    nav_dict['panoimg_list'].append(img_savename)
                    step_id += 1

            output_json_filename = os.path.join(save_dir_tmp, '{}.json'.format(room_name))
            with open(output_json_filename, 'w', encoding='utf-8') as f:
                json.dump(nav_dict, f)

    print("Expert Demonstration Data Creation Done!")


def main():
    #global configuration
    save_dir = './expert_demon_data/'
    step_size = 0.25
    turn_angle = 10

    collect_expert_demon_data(step_size, turn_angle, save_dir)


if __name__ == "__main__":
    main()
