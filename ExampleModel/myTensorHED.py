from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf



class HEDLAYERS:

    def __init__(self, Hed_graph):
        #load weight from HED net
        with Hed_graph.as_default():
				    meta_path_restore = '/home/eric/disk/fcnForSpallingCrack/hed/holy-edge-master/save1128/models/hed-model-14900.meta'
				    model_path_restore = '/home/eric/disk/fcnForSpallingCrack/hed/holy-edge-master/save1128/models/hed-model-14900'
				    #with tf.Session() as sess:
				    saver_restore = tf.train.import_meta_graph(meta_path_restore)
				    
				    with tf.Session(graph = Hed_graph) as sess:
				      saver_restore.restore(sess,model_path_restore)
				      graph_restore = tf.get_default_graph()
				      side_1_f = graph_restore.get_tensor_by_name("side_1/Variable:0")
				      side_1_b = graph_restore.get_tensor_by_name("side_1/Variable_1:0")
				      side_1_d = graph_restore.get_tensor_by_name("side_1/Variable_2:0")
				      side_2_f = graph_restore.get_tensor_by_name("side_2/Variable:0")
				      side_2_b = graph_restore.get_tensor_by_name("side_2/Variable_1:0")
				      side_2_d = graph_restore.get_tensor_by_name("side_2/Variable_2:0")
				      side_3_f = graph_restore.get_tensor_by_name("side_3/Variable:0")
				      side_3_b = graph_restore.get_tensor_by_name("side_3/Variable_1:0")
				      side_3_d = graph_restore.get_tensor_by_name("side_3/Variable_2:0")
				      side_4_f = graph_restore.get_tensor_by_name("side_4/Variable:0")
				      side_4_b = graph_restore.get_tensor_by_name("side_4/Variable_1:0")
				      side_4_d = graph_restore.get_tensor_by_name("side_4/Variable_2:0")
				      side_5_f = graph_restore.get_tensor_by_name("side_5/Variable:0")
				      side_5_b = graph_restore.get_tensor_by_name("side_5/Variable_1:0")
				      side_5_d = graph_restore.get_tensor_by_name("side_5/Variable_2:0")
				      self.side_1_f = sess.run(side_1_f)
				      self.side_1_b = sess.run(side_1_b)
				      self.side_1_d = sess.run(side_1_d)
				      self.side_2_f = sess.run(side_2_f)
				      self.side_2_b = sess.run(side_2_b)
				      self.side_2_d = sess.run(side_2_d)
				      self.side_3_f = sess.run(side_3_f)
				      self.side_3_b = sess.run(side_3_b)
				      self.side_3_d = sess.run(side_3_d)
				      self.side_4_f = sess.run(side_4_f)
				      self.side_4_b = sess.run(side_4_b)
				      self.side_4_d = sess.run(side_4_d)
				      self.side_5_f = sess.run(side_5_f)
				      self.side_5_b = sess.run(side_5_b)
				      self.side_5_d = sess.run(side_5_d)

    def _get_parameter(self):

        return self.side_1_f, self.side_1_b, self.side_1_d, self.side_2_f, self.side_2_b, self.side_2_d, self.side_3_f, self.side_3_b, self.side_3_d, self.side_4_f, self.side_4_b, self.side_4_d, self.side_5_f, self.side_5_b, self.side_5_d
