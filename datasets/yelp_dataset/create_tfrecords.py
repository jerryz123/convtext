import tensorflow as tf
import multiprocessing as mp
import glob
import pickle as pkl
import string
import os


def _map_record_save(assignments_and_word_vec):
    business_assignments, word_to_token, title_word_to_token = assignments_and_word_vec
    print('got', len(business_assignments), 'assignments')
    review_counter = 0

    for b_fname in business_assignments:
        record_name = os.path.join('records/', '{}.tfrecords'.format(b_fname.split('/')[-1].split('.csv')[0]))
        writer = tf.python_io.TFRecordWriter(record_name)

        name_toks = None

        with open(b_fname, 'r') as f:
            for line in f:
                split_line = line.split(',')
                
                if name_toks is None:
                    name_toks = split_line[0].split()
                
                try:
                    stars = int(split_line[1])
                except ValueError:
                    print(b_fname, split_line[1])
                    
                stars, tokens = int(split_line[1]), split_line[2].split()
                
                ex = tf.train.SequenceExample()
                ex.context.feature["star"].int64_list.value.append(stars)
                
                text_tokens = ex.feature_lists.feature_list["text"]
                for t in tokens:
                    text_tokens.feature.add().int64_list.value.append(word_to_token[t])
                
                title_tokens = ex.feature_lists.feature_list["title"]
                for t in name_toks:
                    title_tokens.feature.add().int64_list.value.append(title_word_to_token[t])
                
                review_counter += 1
                writer.write(ex.SerializeToString())
        writer.close()
    return review_counter


def _map_wc(business_assignments):
    words, title_words = {}, {}
    print('got', len(business_assignments), 'assignments')
    for b_fname in business_assignments:
        with open(b_fname, 'r') as f:
            for n, line in enumerate(f):
                split_line = line.split(',')
                if len(split_line) < 3:
                    print(b_fname)
                    print(line)
                    exit(-1)
                if n == 0:
                    title_tokens = split_line[0].split()
                    for t in title_tokens:
                        if t in title_words:
                            title_words[t] += 1
                        else:
                            title_words[t] = 1
                        
                tokens = split_line[2].split()

                for t in tokens:
                    if t in words:
                        words[t] += 1
                    else:
                        words[t] = 1
    return words, title_words

def _reduce_wc(dict_list):
    unique_words, unique_title_words = set(dict_list[0][0]), set(dict_list[0][1])
    for i in range(1, len(dict_list)):
        unique_words, unique_title_words = unique_words | set(dict_list[i][0]), unique_title_words | set(dict_list[i][1])

    return {k: sum([d[0].get(k, 0) for d in dict_list]) for k in unique_words}, {k: sum([d[1].get(k, 0) for d in dict_list]) for k in unique_title_words}

def init_tabels(unique_words, sorted_words):
    word_to_token, word_lookup = {}, ['<END>', '<START>', '<PERIOD>']
    word_to_token['<END>'] = 0
    word_to_token['<START>'] = 1
    word_to_token['<PERIOD>'] = 2

    w_counter = 3
    for w in sorted_words:
        if w not in ['<END>', '<START>', '<PERIOD>']:
            word_lookup.append(w)
            word_to_token[w] = w_counter
            w_counter += 1
    return word_to_token, word_lookup

def init_title_tabels(unique_words, sorted_words):
    word_to_token, word_lookup = {}, ['<UNK>']
    word_to_token['<UNK>'] = 0

    w_counter = 1
    for w in sorted_words:
        word_lookup.append(w)
        word_to_token[w] = w_counter
        w_counter += 1
    return word_to_token, word_lookup
def main():
    b_files = glob.glob('processed/*.csv')
    N_BUSINESSES = len(b_files)
    CORES = 8
    N_TOKENIZE = 30000

    pool = mp.Pool(CORES)
    spacing = N_BUSINESSES // CORES
    jobs = [b_files[i * spacing: (i + 1) * spacing] for i in range(CORES - 1)]
    jobs.append(b_files[(CORES - 1) * spacing:])
    wc_dicts = pool.map(_map_wc, jobs)
    
    unique_words, unique_title_words = _reduce_wc(wc_dicts)
    sorted_words = sorted(unique_words.keys(), key = lambda x : -unique_words[x])
    sorted_title_words = sorted(unique_title_words.keys(), key = lambda x : -unique_title_words[x])

    print('there are', len(unique_words.keys()), 'unique words')
    print()
    print('TOP FIFTY WORDS')
    for w in sorted_words[:50]:
        print('word', w, 'has count', unique_words[w])
    print()
    print('there are', len(unique_title_words.keys()), 'unique title words')
    print('TOP FIFTY TITLE WORDS')
    for w in sorted_title_words[:50]:
        print('title word', w, 'has count', unique_title_words[w])
    print()
    
    word_to_token, word_lookup = init_tabels(unique_words, sorted_words)
    pkl.dump({'word_to_token' : word_to_token, 'word_lookup' : word_lookup}, open( "records/word_tabels.pkl", "wb" ))

    title_word_to_token, title_word_lookup = init_title_tabels(unique_title_words, sorted_title_words)
    pkl.dump({'word_to_token' : title_word_to_token, 'word_lookup' : title_word_lookup}, open( "records/b_name_word_tabels.pkl", "wb" ))

    jobs = [(j, word_to_token, title_word_to_token) for j in jobs]
    num_printed = pool.map(_map_record_save, jobs)
    print('saved', sum(num_printed), 'reviews')
    pool.close()

if __name__ =='__main__':
    main()