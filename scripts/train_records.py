import os
import tensorflow as tf
from datasets.yelp_dataset.read_tfrecords import build_record_reader
from tensorflow.python.platform import flags
from importlib.machinery import SourceFileLoader
from eval_records import token_lookup
import numpy as np
import pickle as pkl
if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('config', '', 'path to configuration file')
    flags.DEFINE_string('pretrained', '', 'pretrained path')

def main():
    hyperparams = SourceFileLoader('hyperparams', FLAGS.config).load_module()
    conf = hyperparams.configuration

    word_lookup_tabel = pkl.load(open(os.path.join(conf['data_dir'], 'word_tabels.pkl'), 'rb'))['word_lookup']
    title_word_lookup_tabel = pkl.load(open(os.path.join(conf['data_dir'], 'b_name_word_tabels.pkl'), 'rb'))['word_lookup']

    with tf.variable_scope('model', reuse = None) as training_scope:
        train_title, train_star, train_text, train_valid = build_record_reader(conf)
        train_model = conf['model'](conf, train_title, train_star, train_text, train_valid)
        train_model.build(is_Train = True)

    with tf.variable_scope('val_model', reuse = None):
        val_title, val_star, val_text, val_valid = build_record_reader(conf, isTrain = False)

        with tf.variable_scope(training_scope, reuse=True):
            val_model = conf['model'](conf, val_title, val_star, val_text, val_valid)
            val_model.build(is_Train = True)
    
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(train_model.loss))
    if 'clip_grad' in conf:
        gradients, _ = tf.clip_by_global_norm(gradients, conf['clip_grad'])
    train_operation = optimizer.apply_gradients(zip(gradients, variables))

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(vars, max_to_keep=0)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    SAVE_DIR = conf['model_dir']
    start_iter = 0

    if len(FLAGS.pretrained) == 0:
        print("saving to:", SAVE_DIR)
        if os.path.exists(SAVE_DIR):
            print("ERROR SAVE DIR EXISTS")
            exit(-1)
        os.mkdir(SAVE_DIR)
    else:
        load_dir = os.path.join(SAVE_DIR, FLAGS.pretrained)
        print("loading pretrained model:", load_dir)
        saver.restore(sess, load_dir)
        start_iter = int(FLAGS.pretrained.split('model')[1]) + 1
    
    eval_out = open(os.path.join(SAVE_DIR, "eval.txt"), "a+")

    writer = tf.summary.FileWriter(SAVE_DIR, graph = sess.graph, flush_secs=10)

    for i in range(start_iter, conf.get('n_iters', 80000)):
        print('on iter {}'.format(i), end='\r')

        if 'learning_rate' in conf:
            iter_rate = conf['learning_rate']
            if i > 25500: #should start cooling roughly about here
                iter_rate /= np.sqrt(i)
            f_dict = {learning_rate : iter_rate}
        else:
            step_num = i + 1
            step_term = min(1. / np.sqrt(step_num), step_num * np.power(4000, -1.5))
            print(1. / np.sqrt(conf['d_embed']) * step_term)
            f_dict = {learning_rate : 1. / np.sqrt(conf['d_embed']) * step_term}
        
        if i % conf.get('debug_step', 100) == 0:
            m_loss, v_loss, _ = sess.run([train_model.norm_loss, val_model.norm_loss, train_operation], feed_dict=f_dict)
            print('At iter {}, model loss: {}, val model loss: {}\n'.format(i, m_loss, v_loss))

            iter_summary = tf.Summary()
            iter_summary.value.add(tag="validation/loss", simple_value = v_loss)
            iter_summary.value.add(tag = "train/loss", simple_value = m_loss)
            writer.add_summary(iter_summary, i)

            if i % conf.get('eval_step', 1000) == 0:
                g_truth, logits, _, input_title, input_stars = sess.run([val_model.input_text, val_model.logits, train_operation, val_model.input_title, val_model.raw_stars], feed_dict=f_dict)
                gen_tokens = np.argmax(logits[0], axis = -1)
                eval_out.write('GENERATING REVIEW AT ITER: {}'.format(i))
                eval_out.write('\n')
                eval_out.write('b_name: ' + ' '.join(token_lookup(input_title[0], title_word_lookup_tabel, conf['n_title_words'])))
                eval_out.write('\n')
                eval_out.write('stars: {}'.format(input_stars[0, 0]))
                eval_out.write('\n \n')
                eval_out.write('GROUND TRUTH VS GEN')
                eval_out.write('\n')
                g_truth_words, gen_words = token_lookup(g_truth[0], word_lookup_tabel, conf['n_words'] - 1), token_lookup(gen_tokens, word_lookup_tabel, conf['n_words'] - 1)
                eval_out.write("REAL")
                eval_out.write('\n')
                eval_out.write(' '.join(g_truth_words))
                eval_out.write('\n')
                eval_out.write("GEN")
                eval_out.write('\n')
                eval_out.write(' '.join(gen_words))
                eval_out.write('\n\n')
                eval_out.flush()
        else:
            sess.run(train_operation, feed_dict=f_dict)
        
        if i % conf.get('save_step', 1000) == 0 and i > 0:
            checkpoint_path = os.path.join(SAVE_DIR, 'model{}'.format(i))
            saver.save(sess, checkpoint_path)
    
    checkpoint_path = os.path.join(SAVE_DIR, 'modelfinal')
    saver.save(sess, checkpoint_path)     

    sess.close()

if __name__ == '__main__':
    main()
