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

class WordTranslator:
    def __init__(self, conf, pkl_path):
        dicts = pkl.load(open(os.path.join(conf['data_dir'], pkl_path), 'rb'))
        self.w_lookup = dicts['word_lookup']
        self.n_lookup = dicts['word_to_token']
    def tokenize_string(self, string):
        splits = string.lower().split()
        return np.array([self.n_lookup[s] for s in splits]).reshape((1, -1))
    def word_lookup(self, num_list):
        return [self.w_lookup[s] for s in np.squeeze(num_list)]
    def num_lookup(self, word_list):
        return np.array([self.n_lookup[w] for w in word_list]).reshape((1, -1))
def main():
    model_name = FLAGS.model
    hyperparams = SourceFileLoader('hyperparams', FLAGS.config).load_module()
    conf = hyperparams.configuration
    conf['batch_size'] = 1

    MAX_GEN = 300
    TARGET_BUSINESS = "Taco Bell"
    TARGET_STARS = 5

    print("generating", TARGET_STARS, "review of", TARGET_BUSINESS)

    title_trans = WordTranslator(conf, 'b_name_word_tabels.pkl')
    text_trans = WordTranslator(conf, 'word_tabels.pkl')

    print(title_trans.tokenize_string(TARGET_BUSINESS))

    with tf.variable_scope('model', reuse = None) as training_scope:
        train_title, train_star, train_text, train_valid = build_record_reader(conf)
        train_model = conf['model'](conf, train_title, train_star, train_text, train_valid)
        train_model.build(is_Train = True)

    with tf.variable_scope('val_model', reuse = None):
        val_title = tf.placeholder(tf.int64, shape = [1, None])
        val_star = tf.placeholder(tf.int64, shape = [1, 1])
        val_text = tf.placeholder(tf.int64, shape = [1, None])
        val_valid = tf.ones_like(val_text, dtype=tf.float32)

        with tf.variable_scope(training_scope, reuse=True):
            val_model = conf['model'](conf, val_title, val_star, val_text, val_valid)
            val_model.build(is_Train = False)

    optimizer = tf.train.AdamOptimizer()
    gradients, variables = zip(*optimizer.compute_gradients(train_model.loss))
    gradients, _ = tf.clip_by_global_norm(gradients, conf.get('clip_grad', 1.0))
    train_operation = optimizer.apply_gradients(zip(gradients, variables))

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(vars, max_to_keep=0)
    config = tf.ConfigProto(device_count = {'GPU': 0})
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())

    MODEL_PATH = os.path.join(conf['model_dir'], model_name)
    print("opening:", MODEL_PATH)

    saver.restore(sess, MODEL_PATH)   
    
    sentence = ['<START>']
    while len(sentence) <= 300 and sentence[-1] != '<END>':

    sess.close()

if __name__ == '__main__':
    main()