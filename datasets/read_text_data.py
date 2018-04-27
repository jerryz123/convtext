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
        numeric_chars = tf.squeeze(tf.decode_raw(split, tf.uint8))
        one_hot = tf.one_hot(numeric_chars, 256)
        return tf.data.Dataset.from_tensor_slices(one_hot)

    dataset = tf.data.TextLineDataset(filenames)#.shuffle(1000).repeat().flat_map(_map_line).batch(batch_size)
    dataset = dataset.repeat()
    #tokenize the dataset into onehot character distributions
    dataset = dataset.flat_map(_map_line)
    #group distributions into seq_length long batches
    dataset = dataset.batch(seq_length).shuffle(1000)
    #create batch_size number of seq_length sequences
    dataset = dataset.batch(batch_size)
    return dataset.make_initializable_iterator()

if __name__ == '__main__':
    import numpy as np
    sess = tf.Session()
    conf = {'BATCH_SIZE' : 1, "SEQ_LENGTH" : 1, 'TRAIN_FILES' : ['corpus.txt']}
    datum = read_text_dataset(conf)
    sess.run(datum.initializer)
    #print(sess.run(tf.string_split(datum.get_next(), delimiter='').values))
    next = datum.get_next()
    for i in range(10):
        data = sess.run(next)
        print('i:',i, 'data:', data)