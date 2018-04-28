import os
import tensorflow as tf
from datasets.yelp_dataset.read_tfrecords import build_record_reader
from tensorflow.python.platform import flags
from importlib.machinery import SourceFileLoader

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('config', '', 'path to configuration file')

def main():
    hyperparams = SourceFileLoader('hyperparams', FLAGS.config).load_module()
    conf = hyperparams.configuration

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

    SAVE_DIR = conf['model_dir']
    print("saving to:", SAVE_DIR)
    if os.path.exists(SAVE_DIR):
        print("ERROR SAVE DIR EXISTS")
        exit(-1)
    os.mkdir(SAVE_DIR)

    writer = tf.summary.FileWriter(SAVE_DIR, graph = sess.graph, flush_secs=10)

    for i in range(conf.get('n_iters', 80000)):
        if i % conf.get('debug_step', 500) == 0:
            m_loss, v_loss, _ = sess.run([train_model.loss, val_model.loss, train_operation])
            print('At iter {}, model loss: {}, val model loss: {}'.format(i, m_loss, v_loss))

            iter_summary = tf.Summary()
            iter_summary.value.add(tag="validation/loss", simple_value = v_loss)
            iter_summary.value.add(tag = "train/loss", simple_value = m_loss)
            writer.add_summary(iter_summary, i)
        else:
            sess.run(train_operation)
        
        if i % conf.get('save_set', 1000) == 0 and i > 0:
            checkpoint_path = os.path.join(SAVE_DIR, 'model{}.ckpt'.format(i))
            saver.save(sess, checkpoint_path)
    
    checkpoint_path = os.path.join(SAVE_DIR, 'modelfinal.ckpt')
    saver.save(sess, checkpoint_path)     

    sess.close()

if __name__ == '__main__':
    main()