import os
import tensorflow as tf
from datasets.yelp_dataset.read_tfrecords import build_record_reader
from tensorflow.python.platform import flags
from importlib.machinery import SourceFileLoader
import pickle as pkl
import numpy as np

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('config', '', 'path to configuration file')
    flags.DEFINE_string('model', '', 'model to evaluate')
def token_lookup(tokens, lookup, THRESH):
    
    words = []
    for t in tokens.squeeze():
        if t >= THRESH:
            words.append('<UNK>')
        else:
            words.append(lookup[t])

    
    return words

def main():
    model_name = FLAGS.model
    hyperparams = SourceFileLoader('hyperparams', FLAGS.config).load_module()
    conf = hyperparams.configuration
    conf['batch_size'] = 1

    word_lookup_tabel = pkl.load(open(os.path.join(conf['data_dir'], 'word_tabels.pkl'), 'rb'))['word_lookup']
    title_word_lookup_tabel = pkl.load(open(os.path.join(conf['data_dir'], 'b_name_word_tabels.pkl'), 'rb'))['word_lookup']

    with tf.variable_scope('model', reuse = None) as training_scope:
        train_title, train_star, train_text = build_record_reader(conf)
        train_model = conf['model'](conf, train_title, train_star, train_text)
        train_model.build()

    with tf.variable_scope('val_model', reuse = None):
        val_title, val_star, val_text = build_record_reader(conf, isTrain = False)

        with tf.variable_scope(training_scope, reuse=True):
            val_model = conf['model'](conf, val_title, val_star, val_text)
            val_model.build()

    optimizer = tf.train.AdamOptimizer()
    gradients, variables = zip(*optimizer.compute_gradients(train_model.loss))
    gradients, _ = tf.clip_by_global_norm(gradients, conf.get('clip_grad', 1.0))
    train_operation = optimizer.apply_gradients(zip(gradients, variables))

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(vars, max_to_keep=0)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    MODEL_DIR = os.path.join(conf['model_dir'], model_name)
    print("opening:", MODEL_DIR)

    saver.restore(sess, MODEL_DIR)   

    g_truth, logits, loss, input_title, input_stars = sess.run([val_model.input_text, val_model.logits, val_model.loss, val_model.input_title, val_model.input_stars])
    gen_tokens = np.argmax(logits, axis = 2)
    
    print(g_truth.shape, gen_tokens.shape)
    print('GENERATING REVIEW')
    print('b_name', ' '.join(token_lookup(input_title[0], title_word_lookup_tabel, conf['n_title_words'])))
    print('stars', input_stars[0, 0])
    print()
    print('GROUND TRUTH VS GEN')
    g_truth_words, gen_words = token_lookup(g_truth[0], word_lookup_tabel, conf['n_words'] - 1), token_lookup(gen_tokens, word_lookup_tabel, conf['n_words'] - 1)
    print("REAL")
    print(' '.join(g_truth_words))
    print("GEN")
    print(' '.join(gen_words))
    
    
    sess.close()

if __name__ == '__main__':
    main()