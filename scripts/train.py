import random
import os
from importlib.machinery import SourceFileLoader

import tensorflow as tf
from tensorflow.python.platform import flags
from models.wavenet import WaveNet
from utils.loss_fns import one_hot_character_loss


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('config', '', 'path to configuration file')
def main():
    hyperparams = SourceFileLoader('hyperparams', FLAGS.config).load_module()
    conf = hyperparams.configuration

    BATCH_SIZE = conf['BATCH_SIZE']
    SEQ_LENGTH = conf['SEQ_LENGTH']
    LEARNING_RATE = conf['LEARNING_RATE']
    L2_REGULARIZATION = conf['L2_REGULARIZATION']
    DEBUG_STEP = conf['DEBUG_STEP'] if 'DEBUG_STEP' in conf else 100
    SAVE_STEP = conf['SAVE_STEP'] if 'SAVE_STEP' in conf else 1000
    NUM_EPOCHS = conf['NUM_EPOCHS']
    VALIDATION_SIZE = conf['VALIDATION_SIZE']
    SAVE_DIR = conf['SAVE_DIR']

    if 'QUEUE_LOADER' in conf:
        from datasets.read_text_data import read_text_dataset
        batch_queue = read_text_dataset(conf)
        input_data = batch_queue.get_next()

        validation_queue = read_text_dataset(conf, is_training=False)
        val_data = validation_queue.get_next()
        wavenet = WaveNet(conf)

        conv2 = wavenet.full_network(input_data)

        loss = one_hot_character_loss(conv2, input_data, l2_norm=L2_REGULARIZATION,
                                          edge_length=SEQ_LENGTH // 2)

        val_conv2 = wavenet.full_network(val_data)
        val_loss = one_hot_character_loss(val_conv2, val_data, l2_norm=L2_REGULARIZATION,
                                              edge_length=SEQ_LENGTH // 2)

    else:
        VOCAB_SIZE = conf['VOCAB_SIZE']
        from datasets.nltk_shakespeare import Corpus
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

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(vars, max_to_keep  =0)

    with tf.Session() as sess:
        if 'QUEUE_LOADER' in conf:
            sess.run([batch_queue.initializer, validation_queue.initializer])

        ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
        writer = tf.summary.FileWriter(SAVE_DIR, graph = sess.graph, flush_secs=10)

        sess.run(init_op)
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)

        if 'QUEUE_LOADER' in conf:
            #technically not correct but treat num_epochs as num_iters
            for i in range(NUM_EPOCHS):
                if i % DEBUG_STEP == 0:
                    m_loss, v_loss, _ = sess.run([loss, val_loss, optim])
                    print("Iter:", i, "Model Loss:", m_loss, "Val Loss:",v_loss)

                    iter_summary = tf.Summary()
                    iter_summary.value.add(tag="validation/loss", simple_value = v_loss)
                    iter_summary.value.add(tag = "train/loss", simple_value = m_loss)
                    writer.add_summary(iter_summary, i)
                else:
                    sess.run([optim])

                if i % SAVE_STEP == 0 and i > 0:
                    checkpoint_path = os.path.join(SAVE_DIR, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=i)
                    print("Model saved")
        else:
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

                        print("iter:" + str(bi) + " Loss " + str(t_loss))
                        # print("".join(one_hot_to_char(corpus.id2char, b[0][-10:])))
                        # print("".join(one_hot_to_char(corpus.id2char, conv_op[0][-10:])))

                checkpoint_path = os.path.join(SAVE_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=e*len(batches))
                print("Model saved")


        writer.close()

if __name__ == '__main__':
    main()