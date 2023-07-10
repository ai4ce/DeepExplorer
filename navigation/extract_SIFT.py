import cv2
import pickle
import os
import glob
import numpy as np

def extract_sift_feat( thred_id, input_img_dir ):
    img_list = glob.glob(os.path.join(input_img_dir, '*.png'))
    assert len(img_list) > 0

    for idx, img_name in enumerate(img_list):
        try:
            if idx%10 == 0:
                print('processed {} imgs in thred {}.'.format(idx, thred_id))
            img = cv2.imread( img_name, 1 )
            if img is None:
                print('img is broken: {}'.format(img_name))
                continue

            img_basename = os.path.basename(img_name)
            kp_save_basename = img_basename.replace('.png', '_kp.pkl')
            kp_save_name = os.path.join(img_name.replace(img_basename,''), kp_save_basename)
            if os.path.exists(kp_save_name):
                continue

            img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
            sift = cv2.xfeatures2d.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img, None)
            index = []
            for point in kp1:
                temp = (point.pt, point.size, point.angle, point.response, point.octave,point.class_id)
                index.append(temp)

            with open( kp_save_name, 'wb') as f:
                f.write(pickle.dumps( index ))

            des_save_name = os.path.join(img_name.replace(img_basename, ''), img_basename.replace('.png','_des.npy') )
            np.save( des_save_name, des1 )
        except:
            print('bad image!')
            continue

def main():
    root_dir = 'data_dir/'
    subdir_list = os.listdir(root_dir)

    for subdir_tmp in subdir_list:
        img_dir = os.path.join(root_dir, subdir_tmp)
        assert os.path.exists(img_dir)
        extract_sift_feat(0, img_dir)

    print('Done!')

if __name__ == '__main__':
    main()
