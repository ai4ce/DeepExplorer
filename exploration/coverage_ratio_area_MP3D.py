import pickle
import os
import numpy as np


def compute_CRA_from_one_file( dirname, step_budget = 1000 ):
    input_filename = os.path.join(dirname, 'explore_rst.pickle')
    assert os.path.exists(input_filename)

    with open(input_filename,'rb') as f:
        nav_dict = pickle.load(f)

    meters_per_pixel = nav_dict['meters_per_pixel']
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

    roomname_list = ['2t7WUuJeko7',
                     '5ZKStnWn8Zo',
                     'ARNzJeq3xxb',
                     'fzynW3qQPVF',
                     'gxdoqLR6rwA',
                     'gYvKGZ5eRqb',
                     'jtcxE69GiFV',
                     'pa4otMbVnkk',
                     'q9vSo1VnCiC',
                     'RPmz2sHmrrY',
                     'rqfALeAoiTq',
                     'UwV83HsGsw3',
                     'Vt2qJdWjCF2',
                     'wc2JMjhGNzB',
                     'WYY7iVyf5p8',
                     'YFuZgdQ5vWj',
                     'yqstnuAEVhm',
                     'YVUC4YcDtcY']

    for seed_id, seed_tmp in enumerate(seed_list):
        for room_id, roomname in enumerate(roomname_list):
            sub_dir_tmp = os.path.join(result_dir, '{}_{}'.format(roomname, seed_tmp))
            assert os.path.exists(sub_dir_tmp)
            coverage_ratio, covered_area = compute_CRA_from_one_file( sub_dir_tmp, step_budget = 1000 )

            covered_areas[seed_id, room_id] = covered_area
            coverage_ratios[seed_id, room_id] = coverage_ratio

    covered_area = np.mean(covered_areas, axis=1, keepdims=False)
    coverage_ratio = np.mean(coverage_ratios, axis=1, keepdims=False)

    coverage_area = np.mean(covered_area)
    coverage_ratio = np.mean(coverage_ratio)

    print('coverage area: {}, coverage ratio: {}'.format(coverage_area, coverage_ratio))
