import tensorflow as tf
from models.wavenet import WaveNet



with tf.Session() as sess:
    wavenet = WaveNet(dilations=[1,2,4,8,16])

    conv2 = wavenet.full_network(tf.ones([1, 50, 1]))
    writer = tf.summary.FileWriter("log/", sess.graph)

    writer.close()
