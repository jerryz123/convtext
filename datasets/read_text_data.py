import tensorflow as tf

def read_text_dataset(filenames, batch_size, seq_length):
    def _map_line(line):
        split = tf.string_split([line], delimiter='').values
        numeric_chars = tf.decode_raw(split, tf.uint8)[:,0]
        return tf.data.Dataset.from_tensor_slices(tf.one_hot(numeric_chars, 256)).batch(seq_length)
    dataset = tf.data.TextLineDataset(filenames).shuffle(1000).repeat().flat_map(_map_line).batch(batch_size)

    return dataset.make_initializable_iterator()

if __name__ == '__main__':
    import numpy as np
    sess = tf.Session()
    datum = read_text_dataset(['corpus.txt'],2, 10)
    sess.run(datum.initializer)
    #print(sess.run(tf.string_split(datum.get_next(), delimiter='').values))
    data = sess.run(datum.get_next())
    print('mapped', data.shape, data.dtype)
    print(np.where(data[0, 0, :] == 1))