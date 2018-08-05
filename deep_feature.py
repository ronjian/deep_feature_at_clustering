from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Activation, Input
from keras import backend as K
import numpy as np
import os
import sys
from skimage.io import imread
from skimage.transform import resize
import argparse
from utils import save_json, data_generator

EXPECT_SIZE = (299, 299, 3)
EXTRACT_LAYER = 132
SAVE_EVERY = 20

        

            

def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
    activations = get_activations([X_batch,0])
    return activations



class Extracter():
    def __init__(self, data_dir, batch_size, weights):
        self.weights = weights
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.model = self._build()
        if self.weights != 'imagenet':
            print("loading weights {}".format(self.weights))
            self.model.load_weights(self.weights, by_name=True, skip_mismatch=True)
        
    def _build(self):
        print("building Xception model")
        base_model = Xception(weights='imagenet', include_top=False, input_tensor=Input(shape = EXPECT_SIZE))
        x = base_model.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        model = Model(inputs = base_model.input, outputs = x)
        return model
    

        
    def _save(self):
        feature_path = "deep_feature/{}.npy".format(self.base_name)
        print("saving feature to {}".format(feature_path))
        np.save(feature_path, np.concatenate(self.feature_l, axis = 0))
        files_name_path = "deep_feature/{}.json".format(self.base_name)
        print("saving files name to {}".format(files_name_path))
        save_json(files_name_path, self.image_l)
                
    def extract(self):
        self.base_name = os.path.basename(self.data_dir)
        data_set = [os.path.join(self.data_dir, file) for file in next(os.walk(self.data_dir))[2] if file.endswith('.jpg')]
        self.feature_l, self.image_l = [], []
        cnt = 0
        total_size = len(data_set)
        data_g = data_generator(data_set, batch_size = self.batch_size)
        save_cnt = 0
        while cnt < total_size:
            print("extracting {} of {}".format(cnt, total_size))
            X, task_l = next(data_g)
            self.image_l.extend(task_l)
            my_featuremaps = get_activations(self.model, EXTRACT_LAYER, X)
            self.feature_l.append(my_featuremaps[0])
            cnt += self.batch_size
            save_cnt += 1
            if save_cnt % SAVE_EVERY == 0:
                self._save()
                save_cnt = 0
        self._save()

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='To extract deep feature')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--gpu', required=False, default='0') 
    parser.add_argument('--batch_size', required=False, default=128)
    parser.add_argument('--weight_path', required=False, default='imagenet')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    ext = Extracter(args.data_dir, args.batch_size, args.weight_path)
    ext.extract()

    
    