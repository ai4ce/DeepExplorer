import pickle
import os
import numpy as np

def compute_CRA_from_two_files( dirname, roomname ):
    if roomname == 'Mosquito':
        floor_step_budget = {0: 800, 1:200}
    elif roomname == 'Sands':
        floor_step_budget = {0: 500, 1:500}
    elif roomname == 'Scioto':
        floor_step_budget = {0: 500, 1:500}

    all_navigable_areas = 0.
    covered_area = 0.
    for floor_id in range(2):
        filename = os.path.join( dirname, 'explore_rst_floor{}.pickle'.format(floor_id))

        assert os.path.exists(filename)

        with open( filename, 'rb') as f:
            nav_dict = pickle.load(f)
            meters_per_pixel = nav_dict[roomname]['meters_per_pixel']
            meters_per_pixel = 0.1
            assert meters_per_pixel != 0.
            all_navigable_pixels = int(nav_dict['all_navigable_pixels'])
            coverage_ratio_list = nav_dict['coverage_ratio_with_distconst'][0:floor_step_budget[floor_id]]
            coverage_ratio = max(coverage_ratio_list)
            covered_area_tmp = all_navigable_pixels*coverage_ratio*meters_per_pixel*meters_per_pixel
            covered_area += covered_area_tmp
            all_navigable_areas += all_navigable_pixels*meters_per_pixel*meters_per_pixel

    coverage_ratio  = covered_area / all_navigable_areas

    return coverage_ratio, covered_area


def compute_CRA_from_one_file( dirname, step_budget = 1000 ):
    input_filename = os.path.join(dirname, 'explore_rst.pickle')
    assert os.path.exists(input_filename)

    with open(input_filename,'rb') as f:
        nav_dict = pickle.load(f)

    meters_per_pixel = nav_dict['meters_per_pixel']
    meters_per_pixel = 0.1
    coverage_ratio_list = nav_dict['coverage_ratio_with_distconst'][0:step_budget]
    coverage_ratio = max(coverage_ratio_list)
    all_navigable_pixels = int(nav_dict['all_navigable_pixels'])

    covered_area = all_navigable_pixels*coverage_ratio*meters_per_pixel*meters_per_pixel

    return coverage_ratio, covered_area


if __name__ == '__main__':
    result_dir = './result_dir/'
    room_num = 14
    step_budget = 1000
    covered_areas = np.zeros([71, room_num], np.float32)
    coverage_ratios = np.zeros([71,room_num], np.float32)

    coverage_ratio_area_large_list = list()
    coverage_curve_large_list = list()

    coverage_ratio_area_small_list = list()
    coverage_curve_small_list = list()

    seed_list = [int(line_tmp.rstrip('\n')) for line_tmp in open('seeds_list.txt').readlines()]


    roomname_list = ['Cantwell', 'Denmark', 'Eastville', 'Edgemere', 'Elmira',
                     'Eudora', 'Greigsville', 'Mosquito', 'Pablo', 'Ribera',
                     'Sands', 'Scioto', 'Sisters', 'Swormville']

    twofloor_room_list = ['Sands', 'Scioto', 'Mosquito']

    for seed_id, seed_tmp in enumerate(seed_list):
        for room_id, roomname in enumerate(roomname_list):
            sub_dir_tmp = os.path.join(result_dir, '{}_{}'.format(roomname, seed_tmp))
            assert os.path.exists(sub_dir_tmp)
            if roomname in twofloor_room_list:
                coverage_ratio, covered_area = compute_CRA_from_two_files(sub_dir_tmp, roomname)
            else:
                coverage_ratio, covered_area = compute_CRA_from_one_file( sub_dir_tmp, step_budget = 1000 )

            covered_areas[seed_id, room_id] = covered_area
            coverage_ratios[seed_id, room_id] = coverage_ratio

    covered_area = np.mean(covered_areas, axis=1, keepdims=False)
    coverage_ratio = np.mean(coverage_ratios, axis=1, keepdims=False)

    coverage_area = np.mean(covered_area)
    coverage_ratio = np.mean(coverage_ratio)

    print('coverage area: {}, coverage ratio: {}'.format(coverage_area, coverage_ratio))
