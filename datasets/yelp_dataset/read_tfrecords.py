import tensorflow as tf
import pickle as pkl
import glob
import os
import random 

def build_record_reader(conf, isTrain = True):
    filenames = glob.glob(os.path.join(conf['data_dir'], '*.tfrecords'))
    
    train_val_split = conf.get('train_val_split', 0.95)
    if isTrain:
        filenames = filenames[:int(len(filenames) * train_val_split)]
    else:
        filenames = filenames[int(len(filenames) * train_val_split):]

    def _parse_function(ex):
        context_feat = {"star": tf.FixedLenFeature([], dtype=tf.int64)}
        sequence_feat = {"text": tf.FixedLenSequenceFeature([], dtype=tf.int64), 
                            "title": tf.FixedLenSequenceFeature([], dtype=tf.int64)}
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized=ex,
                                        context_features=context_feat,sequence_features=sequence_feat)
        
        star_tensor = tf.expand_dims(context_parsed["star"], 0)
        text_tensor, title_tensor = sequence_parsed["text"], sequence_parsed["title"]
        return title_tensor, star_tensor, text_tensor
    
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat()
    dataset = dataset.padded_batch(conf.get('batch_size', 1), ([None], [None], [None]))
    iterator = dataset.make_one_shot_iterator()

    title_tensor, star_tensor, text_tensor = iterator.get_next()
    return  title_tensor, star_tensor, text_tensor


def main():
    conf = {'data_dir' : 'records', 'batch_size' : 8}

    title, star, text = build_record_reader(conf)

    sess = tf.Session()
    title_npy, star_npy, text_npy = sess.run([title, star, text])

    print('title', title_npy.shape)
    print('star', star_npy.shape)
    print('text', text_npy.shape)

    word_lookup = pkl.load(open('records/word_tabels.pkl', 'rb'))['word_lookup']
    word_lookup_title = pkl.load(open('records/b_name_word_tabels.pkl', 'rb'))['word_lookup']

    for b in range(conf.get('batch_size', 1)):
        title = ' '.join([word_lookup_title[t] for t in title_npy[b]])
        stars = star_npy[b][0]
        text = ' '.join([word_lookup[t] for t in text_npy[b]])
        print('business:', title)
        print('review stars', stars)
        print('review text', text)

if __name__ == '__main__':
    main()