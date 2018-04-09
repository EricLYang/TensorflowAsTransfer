### A simple demo to obtain layer parameters using Tensorflow

Tell tensorflow the path where you saved the MetaGraph:

> - meta_path_restore = '/home/eric/disk/fcnForSpallingCrack/hed/holy-edge-master/save1128/models/hed-model-14900.meta'

Just follow the checkpoint form


### Then restore the MetaGraph, be sure you know attribute of each layer


> - saver_restore = tf.train.import_meta_graph(meta_path_restore)


### finally you can obtain the parameters under a predefined graph

In tensorflow, each model should run under independent graph, thus: 

> -  with tf.Session(graph = Hed_graph) as sess:

Then you can safely restore the parameters

> - saver_restore.restore(sess,model_path_restore)

In order to get the parameter according to attributes, you need to obtain the current graph:

> - graph_restore = tf.get_default_graph()

