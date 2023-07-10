import json
import os
import numpy as np

def get_episode_nav_result_list(episode_nav_filename_list):
    episode_nav_result_list = list()

    for episode_nav_filename in episode_nav_filename_list:
        with open(episode_nav_filename, 'r') as f:
            nav_result = json.load(f)

        for key_name in nav_result.keys():
            episode_nav_result_list.append( nav_result[key_name] )

    return episode_nav_result_list


def compute_SPL_SuccRate( episode_nav_result_list ):
    actual_pathlen_list = list()
    geodist_list = list()
    success_list = list()

    for episode_nav_result in episode_nav_result_list:
        geodist_list.append(float(episode_nav_result['geodesic_dist']))
        actual_pathlen_list.append(float(episode_nav_result['navigation_result']['pathlen']))
        success_list.append(episode_nav_result['accurate_localized'])

    actual_pathlen_vec = np.array(actual_pathlen_list, np.float32)
    geodist_vec = np.array(geodist_list, np.float32)
    success_vec = np.array(success_list, np.float32)

    assert success_vec.shape[0] > 0

    max_pathlen_geodist = np.maximum(geodist_vec, actual_pathlen_vec)

    geo_div_max = np.divide(geodist_vec, max_pathlen_geodist)

    geo_div_max = np.multiply(success_vec, geo_div_max )

    SPL = np.sum(geo_div_max)/len(geodist_list)

    SuccRate = np.sum(success_vec)/success_vec.shape[0]

    return SPL, SuccRate


def main():
    root_dir = '.result_dir/'

    room_name_list = ['Cantwell', 'Denmark', 'Eastville', 'Edgemere', 'Elmira',
                      'Eudora','Greigsville', 'Pablo', 'Ribera', 'Sisters', 'Swormville',
                      'Mosquito','Sands','Scioto']

    episode_nav_filename_list = list()

    for room_name in room_name_list:
        nav_result_filename = os.path.join( root_dir,
                                            room_name,
                                            '{}_reconnect_with_geodist_navresult.json'.format(room_name))
        assert os.path.exists(nav_result_filename)

        episode_nav_filename_list.append(nav_result_filename)

    episode_nav_result_list = get_episode_nav_result_list(episode_nav_filename_list)

    SPL, SuccRate = compute_SPL_SuccRate( episode_nav_result_list )

    print('SPL = {}, SuccRate = {}'.format(SPL, SuccRate ) )
    print('Done!')

if __name__ == '__main__':
    main()


