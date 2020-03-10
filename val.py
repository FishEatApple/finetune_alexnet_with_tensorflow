import numpy as np

import cv2
import matplotlib.pyplot as plt

from alexnet import AlexNet
from caffe_classes import class_names
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
imagenet_mean=np.array([104.,117.,124.],dtype=np.float32)
current=os.getcwd()
print(current)
image_dir=os.path.join(current,'images')
img_files=[os.path.join(image_dir,f)for f in os.listdir(image_dir) if f.endswith('.jpeg')]
imgs=[]
for f in img_files:
    imgs.append(cv2.imread(f))
fig=plt.figure(figsize=(15,6))
# for i,img in enumerate(imgs):
#     fig.add_subplot(1,3,i+1)
#     # plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2GRB))
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
# plt.show()

x=tf.placeholder(tf.float32,[1,227,227,3])
keep_prob=tf.placeholder(tf.float32)
model=AlexNet(x,keep_prob,1000,[])
score=model.fc8
softmax=tf.nn.softmax(score)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    model.load_initial_weights(sess)
    fig2=plt.figure(figsize=(15,6))
    for i,image in enumerate(imgs):
        # cv2.imshow(str(i),image)
        img=cv2.resize(image.astype(np.float32),(227,227))
        img-=imagenet_mean
        img=img.reshape((1,227,227,3))
        probs=sess.run(softmax,feed_dict={x:img,keep_prob:1})
        class_name=class_names[np.argmax(probs)]
        maxVal=probs.max()
        cv2.imwrite('./output/'+class_name+'_'+str(maxVal)+'.jpeg',image)