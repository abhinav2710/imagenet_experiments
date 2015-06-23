import numpy as np
import sys
import matplotlib.pyplot as plt
import heapq
import os

caffe_root = '/home/abhinav/Documents/Softwares/caffe/'
sys.path.insert(0, caffe_root + 'python')

base_path = sys.argv[1]
file_src = sys.argv[2]
log_file = sys.argv[3]

import caffe

MODEL_FILE = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMAGE_FILE = 'images/cat.jpg'

fh = open(file_src, 'r')
lines = fh.read().split('\n')
log = open(log_file, 'w')

caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
#net.set_mode_gpu()

images = []
image_names = []
image_count = 0
image_labels = []
K = 5

for l in lines:
    lsplit = l.split(' ')
    file_name = lsplit[0]
    correct_label = lsplit[1]
    image = caffe.io.load_image(base_path + '/' + file_name)
    image_names.append(file_name)
    images.append(image)
    image_labels.append(correct_label)
    image_count = image_count + 1
    
    if image_count % 10 == 0:
        predictions = net.predict(images)
        for i in range(10):
            preds = predictions[i]
            top_k = heapq.nlargest(K, range(len(preds)), preds.take);
            log.write(image_names[i] + ' ' + str(top_k) + ' ' + (image_labels[i]))
            print(image_names[i] + ' ' + str(top_k) + ' ' + (image_labels[i]))
        images = []
        image_names = []
        image_labels = []
        log.flush()
        os.fsync(log)
log.close()


