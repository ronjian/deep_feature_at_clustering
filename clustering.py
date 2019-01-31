import os
import sys
import numpy as np
from sklearn.cluster import KMeans
import argparse
from utils import read_json, recreate_dir, thumbnail
import shutil



class Cluster():
    def __init__(self, feature_path, clustering_cnt, target_dir):
        self.feature_path = feature_path
        assert os.path.exists(self.feature_path)
        self.file_path = self.feature_path.split('.')[0] + ".json"
        assert os.path.exists(self.file_path)
        self.clustering_cnt = int(clustering_cnt)
        self.target_dir = target_dir
        self.cluster = self._build()
    
    def _build(self):
        return KMeans(n_clusters = self.clustering_cnt,
                 random_state = 0, 
                 n_jobs = -7, # retain 7 cpu threads
                 verbose = 1, # show detail
                 max_iter = 10000)
    
    def load_feature(self):
        print("loading files list {}".format(self.file_path))
        self.file_list = np.asarray(read_json(self.file_path))
        print("loading deep feature {}".format(self.feature_path))
        feature_tmp = np.load(self.feature_path)
        norm_feature = np.linalg.norm(feature_tmp, axis = 1, keepdims=True)
        self.feature = feature_tmp / norm_feature

        #shuffle dataset
        self.size = self.file_list.shape[0]
        idx = np.asarray(range(self.size))
        np.random.shuffle(idx)
        self.file_list = self.file_list[idx]
        self.feature = self.feature[idx]
        
        

    def train(self):
        print("Cluster fitting")
        self.fitter = self.cluster.fit(self.feature)
    
    def centroid_distance(self):
        centroid_distance = self.fitter.transform(self.feature)
        print("merge image name, class, distance together")
        res_arr = []
        for i in range(self.size):
            image_name = self.file_list[i]
            cls_id = self.fitter.labels_[i]
            res_arr.append([image_name, cls_id, centroid_distance[i][cls_id]])
        self.res_arr = np.asarray(res_arr)
    
    def generate_target(self):
        recreate_dir(self.target_dir)
        for cls_id in range(self.clustering_cnt):
            cls_dir = os.path.join(self.target_dir, str(cls_id))
            recreate_dir(cls_dir)
            cls_arr = self.res_arr[self.res_arr[:, 1] == str(cls_id)]
            # sort as distance ascending order
            asc_idx = cls_arr[:, 2].argsort()
            # verify ascending
            if len(cls_arr[asc_idx][:, 2]) > 5:
                assert cls_arr[asc_idx][:, 2][0] < cls_arr[asc_idx][:, 2][5]
            sort_num = 0
            for img_path in cls_arr[asc_idx][:, 0]:
                sort_num += 1
                shutil.copy2(img_path, os.path.join(cls_dir, "{}_{}".format(sort_num, img_path.split('/')[-1])))

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Clustering deep feature')
    parser.add_argument('--feature_path', required=True)
    parser.add_argument('--clustering_cnt', required=True)
    parser.add_argument('--target_dir', required=True)
    
    args = parser.parse_args()
    
    C = Cluster(args.feature_path, args.clustering_cnt, args.target_dir)
    C.load_feature()
    C.train()
    C.centroid_distance()
    C.generate_target()
    
    thumbnail(args.target_dir)
