import tensorflow as tf
from models.wavenet import WaveNet

wavenet = WaveNet()

wavenet.full_network(tf.ones([1, 50, 1]))
