import tensorflow as tf

def read_text_dataset(conf, is_training = True):
    if is_training:
        filenames = conf['TRAIN_FILES']
    else:
        filenames = conf['TEST_FILES']

    conf['VOCAB_SIZE'] = 256

    seq_length = conf['SEQ_LENGTH']
    batch_size = conf['BATCH_SIZE']
    def _map_line(line):
        split = tf.string_split([line], delimiter='').values
        numeric_chars = tf.decode_raw(split, tf.uint8)[:,0]
        return tf.data.Dataset.from_tensor_slices(tf.one_hot(numeric_chars, 256)).batch(seq_length)
    dataset = tf.data.TextLineDataset(filenames).shuffle(1000).repeat().flat_map(_map_line).batch(batch_size)

    return dataset.make_initializable_iterator()

if __name__ == '__main__':
    import numpy as np
    sess = tf.Session()
    conf = {'BATCH_SIZE' : 2, "SEQ_LENGTH" : 10, 'TRAIN_FILES' : ['corpus.txt']}
    datum = read_text_dataset(conf)
    sess.run(datum.initializer)
    #print(sess.run(tf.string_split(datum.get_next(), delimiter='').values))
    data = sess.run(datum.get_next())
    print('mapped', data.shape, data.dtype)
    print(np.where(data[0, 0, :] == 1))