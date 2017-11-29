# coding: utf-8
"""
Author: Jay Meng
E-mail: jalymo@126.com
Wechat：345238818
"""

import tensorflow as tf
from config import cfg

from nets import ssd_vgg_300, np_methods
from preprocessing import ssd_vgg_preprocessing


VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}

DEVOC_LABELS = {
    0:('none', 'Background'),
    1:('aeroplane', 'Vehicle'),
    2:('bicycle' 'Vehicle'),
    3:('bird', 'Animal'),
    4:('boat', 'Vehicle'),
    5:('bottle', 'Indoor'),
    6:('bus', 'Vehicle'),
    7:('car', 'Vehicle'),
    8:('cat', 'Animal'),
    9:('chair', 'Indoor'),
    10:('cow', 'Animal'),
    11:('diningtable', 'Indoor'),
    12:('dog', 'Animal'),
    13:('horse', 'Animal'),
    14:('motorbike', 'Vehicle'),
    15:('person', 'Person'),
    16:('pottedplant', 'Indoor'),
    17:('sheep', 'Animal'),
    18:('sofa', 'Indoor'),
    19:('train', 'Vehicle'),
    20:('tvmonitor', 'Indoor'),
}

class SSDNet(object): 
    def __init__(self, isess):  
        self.isess = isess 
        # ## SSD 300 Model        
        # The SSD 300 network takes 300x300 image inputs. In order to feed any image, 
        # the latter is resize to this input shape (i.e.`Resize.WARP_RESIZE`). 
        # Note that even though it may change the ratio width / height, the SSD model 
        # performs well on resized images (and it is the default behaviour in the original Caffe implementation).
        # 
        # SSD anchors correspond to the default bounding boxes encoded in the network. 
        # The SSD net output provides offset on the coordinates and dimensions of these anchors.
        
        # Input placeholder.
        #self.net_shape = (cfg.net_width, cfg.net_height)       
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, cfg.num_channels))
        
        # Evaluation pre-processing: resize to SSD net shape.
        image_pre, _, _, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            self.img_input, None, None, (cfg.net_width, cfg.net_height), cfg.data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        self.image_4d = tf.expand_dims(image_pre, 0)
        
        tf.logging.info('Seting up the main structure')


    def build_arch(self, ckpt_filename):
        # Define the SSD model.
        slim = tf.contrib.slim
        reuse = True if 'ssd_net' in locals() else None
        ssd_net = ssd_vgg_300.SSDNet()
        with slim.arg_scope(ssd_net.arg_scope(data_format=cfg.data_format)):
            self.predictions, self.localisations, _, _ = ssd_net.net(self.image_4d, is_training=False, reuse=reuse)
        
        # Restore SSD model.
        self.isess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.isess, ckpt_filename)
        
        # SSD default anchor boxes.
        self.ssd_anchors = ssd_net.anchors((cfg.net_width, cfg.net_height))
    
    
    # ## Post-processing pipeline
    # The SSD outputs need to be post-processed to provide proper detections. 
    # Namely, we follow these common steps:
    # * Select boxes above a classification threshold;
    # * Clip boxes to the image shape;
    # * Apply the Non-Maximum-Selection algorithm: fuse together boxes whose Jaccard score > threshold;
    # * If necessary, resize bounding boxes to original image shape.    
    
    # Main image detecting routine.
    def detect_image(self, img):
        # Run SSD network.
        _, rpredictions, rlocalisations, rbbox_img = self.isess.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],
                                                                        feed_dict={self.img_input: img})
        
        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                rpredictions, rlocalisations, self.ssd_anchors,
                select_threshold=cfg.select_threshold, img_shape=(cfg.net_width, cfg.net_height), 
                        num_classes=cfg.num_classes, decode=True)
        
        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, 
                nms_threshold=cfg.nms_threshold)
        
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes
        
        