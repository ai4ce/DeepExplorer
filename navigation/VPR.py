
import numpy as np
import cv2
import warnings
from pathlib import Path
from tqdm import tqdm

import torch

from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree

from superpoint import SuperPoint

warnings.filterwarnings('ignore')

class SIFT_VPR:
    def __init__(self,
        np_path,
        n_clusters=16,
        n_init=1,
        verbose=1,
        leaf_size=60,
        metric='minkowski'):
        '''
        np_path: str, Path to all the .npy files.
        n_clusters: int, Number of cluster when generating the codebook. Larger the number of imgs, larger the cluster.
        n_init: int, Number of time the codebook algorithm will be run with different centroid seeds.
        verbose=1: int, How verbose the codebook generating process is.
        leaf_size=60: int, Leaf size in the Ball Tree.
        metric: str, the metric used in the Ball Tree. 'minkowski' is L2. See the scikit-learn doc for more options
        '''
        np_path = Path(np_path)
        SIFT_path_list = np_path.glob('*.npy') # get a list of SIFT feature path

        print('INIT: Reading All the SIFT feature...')
        self.SIFT_dict = dict() # name is the key, SIFT value is the value
        self.all_SIFT_feat = self.init_SIFT_dict(SIFT_path_list) # all the SIFT features in a big list

        print('INIT: Generating SIFT codebook...')
        self.codebook = KMeans(n_clusters = n_clusters, init='k-means++', n_init=n_init, verbose=verbose).fit(self.all_SIFT_feat)

        print('INIT: Calculating VLAD features...')
        self.VLAD, self.VLAD_name = self.init_VLAD() # The same index will get the corresponding name and its VLAD feature

        print('INIT: Indexing VLAD feature with a BallTree...')
        self.tree = BallTree(self.VLAD, leaf_size=leaf_size, metric = metric)

        print('INIT DONE!!')


    ############# Initialization Functions. Don't Call #############
    def init_SIFT_dict(self, SIFT_path_list):
        '''
        Initialization method. Get all the SIFT features. Do not call this.
        '''
        all_descriptors = list()
        for npy_path in tqdm(list(SIFT_path_list)):

            # get feature
            des = np.load(npy_path)

            # get the name
            npy_name = npy_path.name

            self.SIFT_dict[npy_name.replace('_des.npy', '.png')] = des
            all_descriptors.extend(des)


        # flatten all the SIFT feature into a big list
        return np.asarray(all_descriptors)

    def init_VLAD(self):
        '''
        Initialization method. Get VLAD feature for every image based on the codebook. Do not call this.
        '''

        database_VLAD = list()
        database_name = list()
        for SIFT_name in tqdm(self.SIFT_dict.keys()):
            des = self.SIFT_dict[SIFT_name]
            VLAD = self.get_VLAD(des, self.codebook)
            database_VLAD.append(VLAD)
            database_name.append(SIFT_name)

        database_VLAD = np.asarray(database_VLAD)

        return database_VLAD, database_name


    ############# Helper Functions. Don't Call #############
    def get_VLAD(self, X, codebook):
        '''
        Initialization method. Get VLAD feature for a specific image based on the codebook. Do not call this.
        '''

        predictedLabels = codebook.predict(X)
        centroids = codebook.cluster_centers_
        labels = codebook.labels_
        k = codebook.n_clusters

        m,d = X.shape
        VLAD_feature = np.zeros([k,d])
        #computing the differences

        # for all the clusters (visual words)
        for i in range(k):
            # if there is at least one descriptor in that cluster
            if np.sum(predictedLabels == i) > 0:
                # add the diferences
                VLAD_feature[i] = np.sum(X[predictedLabels==i,:] - centroids[i],axis=0)


        VLAD_feature = VLAD_feature.flatten()
        # power normalization, also called square-rooting normalization
        VLAD_feature = np.sign(VLAD_feature)*np.sqrt(np.abs(VLAD_feature))

        # L2 normalization
        VLAD_feature = VLAD_feature/np.linalg.norm(VLAD_feature)
        return VLAD_feature


    ############# Only Call this function #############
    def query(self, query_name, n_results):
        '''
        Return similar image's names given the query_name

        query_name: str, the name of the image you want to find similar images
        I will assume the name is a .npy name. But if you want to query with .png, path, or pure name, let me know

        n_results: int, the number of similar images you want to query
        '''
        name_list = list()
        q_des = self.SIFT_dict[query_name]
        query_VLAD = self.get_VLAD(q_des, self.codebook).reshape(1, -1)

        # we only want the cloest one
        dist, index = self.tree.query(query_VLAD, n_results)
        # index is an array of array of 1
        for idx in index[0]:
            value_name = self.VLAD_name[idx]
            name_list.append(value_name)

        return name_list


class Superpoint_VPR:
    def __init__(self,
        np_path,
        n_clusters=64,
        n_init=1,
        verbose=1,
        leaf_size=60,
        metric='minkowski'):
        '''
        np_path: str, Path to all the .npy files.
        n_clusters: int, Number of cluster when generating the codebook. Larger the number of imgs, larger the cluster.
        n_init: int, Number of time the codebook algorithm will be run with different centroid seeds.
        verbose=1: int, How verbose the codebook generating process is.
        leaf_size=60: int, Leaf size in the Ball Tree.
        metric: str, the metric used in the Ball Tree. 'minkowski' is L2. See the scikit-learn doc for more options
        '''

        print('INIT: Loading SuperPoint Model...')
        config = {}
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = SuperPoint(config).to(self.device) # the weights folder needs to be placed at the same folder


        print('INIT: Extracting All the Superpoint feature...')
        np_path = Path(np_path)
        img_path_list = np_path.glob('*.png') # get a list of SIFT feature path

        self.SP_dict = dict() # .png name is the key, SP value is the value
        self.all_SP_feat = self.init_SP_dict(img_path_list) # all the SIFT features in a big list

        print('INIT: Generating SuperPoint codebook...')
        self.codebook = KMeans(n_clusters = n_clusters, init='k-means++', n_init=n_init, verbose=verbose).fit(self.all_SP_feat)

        print('INIT: Calculating VLAD features...')
        self.VLAD, self.VLAD_name = self.init_VLAD() # The same index will get the corresponding name and its VLAD feature

        print('INIT: Indexing VLAD feature with a BallTree...')
        self.tree = BallTree(self.VLAD, leaf_size=leaf_size, metric = metric)

        print('INIT DONE!!')

    ############# Initialization Functions. Don't Call #############
    def init_SP_dict(self, img_path_list):
        '''
        Initialization method. Get all the SIFT features. Do not call this.
        '''


        all_descriptors = list()
        for img_path in tqdm(list(img_path_list)):

            # get feature
            des = self.describe_SP(img_path, self.model, self.device)

            # get the name
            npy_name = (img_path.name)

            self.SP_dict[npy_name] = des
            all_descriptors.extend(des)


        # flatten all the SIFT feature into a big list
        return np.asarray(all_descriptors)

    def init_VLAD(self):
        '''
        Initialization method. Get VLAD feature for every image based on the codebook. Do not call this.
        '''

        database_VLAD = list()
        database_name = list()
        for SP_name in tqdm(self.SP_dict.keys()):
            des = self.SP_dict[SP_name]
            VLAD = self.get_VLAD(des, self.codebook)
            database_VLAD.append(VLAD)
            database_name.append(SP_name)

        database_VLAD = np.asarray(database_VLAD)

        return database_VLAD, database_name


    ############# Helper Functions. Don't Call #############
    def get_VLAD(self, X, codebook):
        '''
        Initialization method. Get VLAD feature for a specific image based on the codebook. Do not call this.
        '''

        predictedLabels = codebook.predict(X)
        centroids = codebook.cluster_centers_
        labels = codebook.labels_
        k = codebook.n_clusters

        m,d = X.shape
        VLAD_feature = np.zeros([k,d])
        #computing the differences

        # for all the clusters (visual words)
        for i in range(k):
            # if there is at least one descriptor in that cluster
            if np.sum(predictedLabels == i) > 0:
                # add the diferences
                VLAD_feature[i] = np.sum(X[predictedLabels==i,:] - centroids[i],axis=0)


        VLAD_feature = VLAD_feature.flatten()
        # power normalization, also called square-rooting normalization
        VLAD_feature = np.sign(VLAD_feature)*np.sqrt(np.abs(VLAD_feature))

        # L2 normalization
        VLAD_feature = VLAD_feature/np.linalg.norm(VLAD_feature)
        return VLAD_feature

    def describe_SP(self, img, model, device):
        model.eval()
        _, inp = self.read_image(img, device)
        pred = model({'image': inp})
        des = pred['descriptors'][0]
        des = torch.transpose(des, 0, 1)
        des = des.cpu().detach().numpy()
        des = des.astype(np.float64)
        return des

    def frame2tensor(self, frame, device):
        return torch.from_numpy(frame/255.).float()[None, None].to(device)

    def read_image(self, path, device):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            return None, None, None

        inp = self.frame2tensor(image, device)
        return image, inp


    ############# Only Call this function #############
    def query(self, query_name, n_results):
        '''
        Return similar image's names given the query_name

        query_name: str, the name of the image you want to find similar images
        I will assume the name is a .png name. But if you want to query with .npy, path, or pure name, let me know

        n_results: int, the number of similar images you want to query
        '''

        name_list = list()
        q_des = self.SP_dict[query_name]
        query_VLAD = self.get_VLAD(q_des, self.codebook).reshape(1, -1)

        # we only want the cloest one
        dist, index = self.tree.query(query_VLAD, n_results)
        # index is an array of array of 1
        for idx in index[0]:
            value_name = self.VLAD_name[idx]
            name_list.append(value_name)

        return name_list
