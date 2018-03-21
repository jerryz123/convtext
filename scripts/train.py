import tensorflow as tf
from models.wavenet import WaveNet
from utils.loss_fns import one_hot_character_loss
from datasets.nltk_shakespeare import Corpus


corpus = Corpus()
id_to_char, raw = corpus.one_hot_raw()

wavenet = WaveNet(input_channels=8)
conv2 = wavenet.full_network(tf.ones([1, 50, 8]))
loss = one_hot_character_loss(conv2, tf.ones([1, 50, 8]), l2_norm=1)

merge = tf.summary.merge_all()
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()


with tf.Session() as sess:
    writer = tf.summary.FileWriter("./log", sess.graph)
    sess.run(init_op)
    #saver.restore(sess, "log/model.ckpt")
    output = sess.run(loss)
    print(output)
    #saver.save(sess=sess, save_path="log/model.ckpt")
    writer.close()
