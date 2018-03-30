import random
import os
from importlib.machinery import SourceFileLoader

import tensorflow as tf
from tensorflow.python.platform import flags
from models.wavenet import WaveNet
from utils.loss_fns import one_hot_character_loss
from utils.encodings import *
from datasets.nltk_shakespeare import Corpus

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('config', '', 'path to configuration file')
def main():
    hyperparams = SourceFileLoader('hyperparams', FLAGS.config).load_module()
    conf = hyperparams.configuration

    BATCH_SIZE = conf['BATCH_SIZE']
    SEQ_LENGTH = conf['SEQ_LENGTH']
    VOCAB_SIZE = conf['VOCAB_SIZE']
    LEARNING_RATE = conf['LEARNING_RATE']
    L2_REGULARIZATION = conf['L2_REGULARIZATION']
    DEBUG_STEP = conf['DEBUG_STEP']
    NUM_EPOCHS = conf['NUM_EPOCHS']
    VALIDATION_SIZE = conf['VALIDATION_SIZE']
    SAVE_DIR = conf['SAVE_DIR']

    corpus = Corpus()
    batches = corpus.create_batches_one_hot(batch_size=BATCH_SIZE,
                                            seq_length=SEQ_LENGTH,
                                            vocab_size=VOCAB_SIZE)
    validation = batches[:VALIDATION_SIZE]
    batches = batches[VALIDATION_SIZE:]


    wavenet = WaveNet(conf)
    input_data = tf.placeholder(tf.int32, [BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE])
    conv2 = wavenet.full_network(input_data)
    loss = one_hot_character_loss(conv2, input_data, l2_norm=L2_REGULARIZATION,
                                  edge_length=SEQ_LENGTH//2)

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
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for e in range(NUM_EPOCHS):
            random.shuffle(batches)
            for bi, b in enumerate(batches):
                feed = {input_data:b}
                eval_loss, _, conv_op = sess.run([loss, optim, conv2], feed)
                if bi % DEBUG_STEP == 0:
                    t_loss = 0
                    for v in validation:
                        feed = {input_data:v}
                        eval_loss, _, conv_op = sess.run([loss, optim, conv2], feed)
                        t_loss += eval_loss
                    t_loss = t_loss / len(validation)

                    print(str(bi) + " Loss " + str(t_loss))
                    # print("".join(one_hot_to_char(corpus.id2char, b[0][-10:])))
                    # print("".join(one_hot_to_char(corpus.id2char, conv_op[0][-10:])))

            checkpoint_path = os.path.join(SAVE_DIR, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=e*len(batches))
            print("Model saved")


        writer.close()

if __name__ == '__main__':
    main()