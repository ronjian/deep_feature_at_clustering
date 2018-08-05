import shutil
import json
import numpy as np
import os
from skimage.io import imread, imsave
from skimage.transform import resize
import cv2



def read_json(file):
    with open(file, 'r',encoding='utf-8') as f:  
        return json.load(f)

def recreate_dir(path):
    print("re-create directory: ", path)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    
def save_json(f_p, dictionary):
    with open(f_p, 'w', encoding='utf-8') as f:  
        json.dump(dictionary, f) 
        
def thumbnail(root_dir):
    row_l = []
    top_k = 20
    for cur_dir, _, files in os.walk(root_dir):
        if cur_dir != root_dir:
            print("generating thumbnail of {}".format(cur_dir))
            files = [[x, int(x.split('_')[0])] for x in files]
            files = sorted(files, key=lambda x: x[1])
            seq_image = np.zeros((300,300,3),np.uint8)
            cv2.putText(seq_image, str(os.path.basename(cur_dir)), (120,80),cv2.FONT_HERSHEY_PLAIN,5,(255,255,255), 10)
            seq_image = _square_resize(seq_image, min_dim=50, max_dim=50)
            col_l = [seq_image]
            for file, _ in files[:top_k]:
                img_arr = imread(os.path.join(cur_dir, file))
                col_l.append(_square_resize(img_arr, min_dim=50, max_dim=50))
            col_cnt = len(col_l) - 1
            while col_cnt < top_k:
                img_arr = np.asarray([[[0, 0, 0]]])
                col_l.append(_square_resize(img_arr, min_dim=50, max_dim=50))
                col_cnt += 1
            row_l.append(np.concatenate(col_l, axis = 1))
    imsave(os.path.join(root_dir, "..", "{}_thumbnail.jpg".format(os.path.basename(root_dir))), np.concatenate(row_l, axis = 0))

    
def _square_resize(image, min_dim=None, max_dim=None, min_scale=None):
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(
            image, (round(h * scale), round(w * scale)),
            order=1, mode="constant", preserve_range=True)

    # Need padding or cropping?
    # Get new height and width
    h, w = image.shape[:2]
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)

    return image.astype(image_dtype)


def data_generator(dataset, batch_size):
    input_l, b, task_l = [], 0, []
    idx = 0
    data_size = len(dataset)
    while True:
        # shuffle at start of epoch
        if idx == 0:
            np.random.shuffle(dataset)
        # compose batch set
        img = imread(dataset[idx])
        task_l.append(dataset[idx])
        
        new_img = resize(img, EXPECT_SIZE, mode='constant')
    
        input_l.append(new_img)
        b += 1
        idx += 1
        
        if b == batch_size:
            yield np.asarray(input_l), task_l
            # re-init
            input_l, b, task_l = [], 0, []
            
        # here is end of epoch
        if idx == data_size:
            idx = 0