import tensorflow as tf
from models.wavenet import WaveNet
from utils.loss_fns import one_hot_character_loss
from utils.encodings import to_one_hot
from datasets.nltk_shakespeare import Corpus
import random
import os
import numpy as np

#TODO: Load these from config
BATCH_SIZE = 1
SEQ_LENGTH = 1024
USE_BIASES = False
DILATIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256,
             1, 2, 4, 8, 16, 32, 64, 128, 256]
VOCAB_SIZE = 75
LEARNING_RATE = 0.001
L2_REGULARIZATION = 0.01
NUM_EPOCHS = 1
SAVE_DIR = "./model_ckpts/shakespeare"

corpus = Corpus()
batches = corpus.create_batches_one_hot(batch_size=BATCH_SIZE,
                                        seq_length=SEQ_LENGTH,
                                        vocab_size=VOCAB_SIZE)
random.shuffle(batches)
wavenet = WaveNet(input_channels=VOCAB_SIZE,
                  batch_size=BATCH_SIZE,
                  use_biases=USE_BIASES,
                  dilations=DILATIONS)

input_data = tf.placeholder(tf.int32, [BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE])
conv2 = wavenet.full_network(input_data)
loss = one_hot_character_loss(conv2, input_data, l2_norm=L2_REGULARIZATION)

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
trainable = tf.trainable_variables()
optim = optimizer.minimize(loss, var_list=trainable)

merge = tf.summary.merge_all()
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
    writer = tf.summary.FileWriter("./log", sess.graph)
    sess.run(init_op)
    saver.restore(sess, ckpt.model_checkpoint_path)


    init = batches[0]
    init = "".join([corpus.id2char[np.argmax(j)] for j in init[0]])
    output_str = init
    init = (SEQ_LENGTH - len(init)) * " " + init

    chars = np.array([corpus.char2id[c] for c in init])
    one_hot = to_one_hot(corpus.char2id, chars)

    while True:
        feed = {input_data:np.expand_dims(one_hot, 0)}
        output, = sess.run([conv2], feed)
        out_id = np.argmax(output[0,-1])
        out_char = corpus.id2char[out_id]
        output_str += out_char
        t = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
        t[:-1] = one_hot[1:]
        t[-1][out_id] = 1
        one_hot = t
        print("".join([corpus.id2char[np.argmax(j)] for j in one_hot]))
