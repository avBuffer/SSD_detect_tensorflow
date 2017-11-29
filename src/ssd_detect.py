# coding: utf-8
"""
Author: Jay Meng
E-mail: jalymo@126.com
Wechat：345238818
"""

import os
import tensorflow as tf
import time
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from config import cfg
from ssdNet import SSDNet
from ssdNet import DEVOC_LABELS


def main(_):       
    print('BEGIN SSD detect ...')
    
    # Saving detect results
    path = cfg.result_dir + '/detect_result.txt'
    if not os.path.exists(cfg.result_dir):
        os.mkdir(cfg.result_dir)
    elif os.path.exists(path):
        os.remove(path)
    
    fd_results = open(path, 'w')
    fd_results.write('Index,Image,Class,Score,Boxes\n')      
            
    # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options) 
    isess = tf.InteractiveSession(config=config)    
    
    ssdNet = SSDNet(isess)
    ckpt_filename = cfg.model_dir + 'VGG_VOC0712_SSD_300x300_iter_120000.ckpt'
    ssdNet.build_arch(ckpt_filename)
    
    # Test some images and output detected results.
    data_path = cfg.data_dir
    image_names = sorted(os.listdir(data_path))
    
    idx = 0
    for image_name in image_names:
        startTime = time.time()
        img = mpimg.imread(data_path + image_name)        
        if len(img) <= 0:
            print('image_name=%s open error!', (data_path + image_name))
            continue  
              
        rclasses, rscores, rbboxes = ssdNet.detect_image(img)
        
        class_type = len(rclasses)
        print('idx=%d image_name=%s class_type=%d' % (idx, (data_path + image_name), class_type))
        if class_type > 0:
            for i in range(0, class_type): 
                class_name = DEVOC_LABELS[rclasses[i]][1] + ':' + DEVOC_LABELS[rclasses[i]][0]    
                endTime = time.time()
                costTime = (endTime - startTime)*1000                
                print('class=%s score=%f boxes=[%f,%f,%f,%f] costTime=%.2fms' % 
                      (class_name, rscores[i], rbboxes[i][0], rbboxes[i][1], rbboxes[i][2], rbboxes[i][3], costTime))
                
                fd_results.write(str(idx) + ',' + (data_path + image_name) + ',' + 
                                 class_name + ',' + str(rscores[i]) + ',[' + 
                                 str(rbboxes[i][0]) + ',' + str(rbboxes[i][1]) + ',' + 
                                 str(rbboxes[i][2]) + ',' + str(rbboxes[i][3]) + '],' + str(costTime) + '\n')
                fd_results.flush()  
        else:
            print('idx=%d image_name=%s without any detected objects!' % (idx, (data_path + image_name)))
              
        idx = idx + 1        
        print('\n')        
        
    fd_results.close()            
    tf.logging.info('Training done') 
    print('END SSD detect')
    
if __name__ == "__main__":
    tf.app.run()
