This project inspired by [CNN features are also great at unsupervised
classification](https://arxiv.org/pdf/1707.01700.pdf), applying [K-Means](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) on the deep feature extracted from [Xception](https://keras.io/applications/#xception).


### Prerequisites

```
scikit-learn=0.19.1
scikit-image=0.13.1
cv2=3.4.1
keras=2.2.0
```

### Useage:

- Extract deep feature of Xception with pre-trained imagenet weights.
Deep feature and files name will be generated under ```./deep_feature``` folder.

```
python3 deep_feature.py --data_dir="dataset/test_data"
```

- Cluster by K-Means on the extracted deep feature.
Clustered images will be generated under ```./dataset``` folder. Sort number (distance to centroid in ascending order) will be 
inserted before the original file name. Top K thumbnail of each cluster will be generated under ```./dataset``` folder.

```
python3 clustering.py\
  --feature_path="deep_feature/test_data.npy"\
  --clustering_cnt=500\
  --target_dir="dataset/test_data_cluster"
```

### Clustered thumbnail sample:

![thumbnail_sample](assets/thumbnail.jpg)


