# coding: utf-8
"""
Author: Jay Meng
E-mail: jalymo@126.com
Wechat：345238818
"""

import tensorflow as tf

flags = tf.app.flags

############################
#    environment setting   #
############################
flags.DEFINE_string('model_dir', '../checkpoints/', 'path for saving models')
flags.DEFINE_string('data_dir', '../data/', 'path for mnist dataset')
flags.DEFINE_string('result_dir', '../results/', 'path for saving results')


############################
#    hyper parameters      #
############################
flags.DEFINE_integer('net_width', 300, 'net shape width')
flags.DEFINE_integer('net_height', 300, 'net shape height')
flags.DEFINE_integer('num_channels', 3, 'image channel')

flags.DEFINE_integer('num_classes', 21, 'num classes')
flags.DEFINE_string('data_format', 'NHWC', 'SSD data fomart')

flags.DEFINE_float('select_threshold', 0.5, 'select threshold')
flags.DEFINE_float('nms_threshold', 0.4, 'nms threshold')


cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
