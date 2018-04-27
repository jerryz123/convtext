import json
import string
import multiprocessing as mp
import random
from nltk.tokenize import sent_tokenize, word_tokenize

import fcntl

N_PER_FILE = 1024
def merge_dicts(dict_list):
    set_user, set_business, set_word = set(dict_list[0][0]), set(dict_list[0][1]), set(dict_list[0][2])
    for i in range(1, len(dict_list)):
        set_user, set_business, set_word = set_user | set(dict_list[i][0]), set_business | set(dict_list[i][1]), set_word | set(dict_list[i][2])

    total_users = {k: sum([d[0].get(k, 0) for d in dict_list]) for k in set_user}
    def_list = [0 for _ in range(5)]
    total_business = {k: [sum([d[1].get(k, def_list)[i] for d in dict_list]) for i in range(5)] for k in set_business}
    total_word = {k: sum([d[2].get(k, 0) for d in dict_list]) for k in set_word}
    return total_users, total_business, total_word
def dict_increment(dict, entry):
    if entry in dict:
        dict[entry] += 1
    else:
        dict[entry] = 1

def print_sorted(dict, top_print = 50):
    sorted_keys = sorted(dict.keys(), key = lambda x : -dict[x])
    for i in range(top_print):
        print(sorted_keys[i], 'has count', dict[sorted_keys[i]])

trantab = str.maketrans({i : None for i in string.punctuation})
punc_newline_transtab = str.maketrans({i : None for i in string.punctuation + '\n'+'\r'+'\t'})


def process_wrapper(line_assignments):
    user_counts = {}
    business_counts = {}
    wd_counts = {}

    start, end = line_assignments
    print('processing', start, end)
    with open('review.json', 'r', 8192) as f:
        l_ctr = 0
        for line in f:
            if l_ctr >= start and l_ctr < end:
                review_entry = json.loads(line)
                business, user = review_entry['business_id'], review_entry['user_id']
                # all lower-case, cleans punctation, and splits by space
                text, stars = review_entry['text'].lower().translate(trantab).split(), int(review_entry['stars']) - 1

                [dict_increment(wd_counts, t) for t in text]
                if business in business_counts:
                    business_counts[business][stars] += 1
                else:
                    star_list = [0 for _ in range(5)]
                    star_list[stars] = 1
                    business_counts[business] = star_list

                #dict_increment(business_counts, business)
                dict_increment(user_counts, user)
            l_ctr += 1
    return user_counts, business_counts, wd_counts


def _mapped_saver(line_assignments):
    worker_num, valid_businesses = line_assignments
    print('thread', worker_num, 'processing', len(valid_businesses.keys()), 'businesses')

    write_list = []
    total_write = 0

    with open('review.json', 'r', 8192) as f:
        for line in f:
            review_entry = json.loads(line)
            business = review_entry['business_id']
            if business in valid_businesses:
                stars = int(review_entry['stars'])
                sentences = [s.translate(punc_newline_transtab).lower() for s in sent_tokenize(review_entry['text'])]
                tokens_out = '<START> ' + ' <PERIOD> '.join(sentences) + ' <END>'
                b_name, b_ctr = valid_businesses[business]

                write_list.append({'b_name' : b_name, 'b_id':b_ctr, 'text' : tokens_out, 'stars' : stars})

            if len(write_list) >= N_PER_FILE :
                for w in write_list:
                    out_str = 'business_{}.csv'.format(w['b_id'])

                    f2 = open('processed/'+out_str, 'a+')
                    fcntl.flock(f2, fcntl.LOCK_EX)
                    f2.write('{},{},{}\n'.format(w['b_name'], w['stars'], w['text']))
                    f2.flush()
                    fcntl.flock(f2, fcntl.LOCK_UN)
                    f2.close()

                total_write += len(write_list)
                write_list = []

    if len(write_list) > 0:
        for w in write_list:
            out_str = 'business_{}.csv'.format(w['b_id'])

            f2 = open('processed/' + out_str, 'a+')
            fcntl.flock(f2, fcntl.LOCK_EX)
            f2.write('{},{},{}\n'.format(w['b_name'], w['stars'], w['text']))
            f2.flush()
            fcntl.flock(f2, fcntl.LOCK_UN)
            f2.close()

        total_write += len(write_list)

    return total_write


#init objects
CORES = 8
pool = mp.Pool(8)
jobs = []

n_lines = 5261669 #line count calculated by wc
spacing = n_lines // CORES
for i in range(CORES):
    jobs.append([i * spacing, (i + 1) * spacing])
jobs[-1][1] = n_lines

ret_dicts = pool.map(process_wrapper, jobs)


user_counts, business_counts, wd_counts = merge_dicts(ret_dicts)
print('There are', len(user_counts.keys()), 'unique users')
print('There are', len(business_counts.keys()), 'unique businesses')
print('There are', len(wd_counts.keys()), 'unique words')

print('Top 50 words are')
print()
print_sorted(wd_counts)

print()
print("Fifty random businesses are")
b_keys = list(business_counts.keys())
for _ in range(50):
    b = random.choice(b_keys)
    print('Business', b, 'has stars', business_counts[b])

print('')
b_key_names = {}
write_counter = 0
for b in business_counts.keys():
    b_list = business_counts[b]
    if b_list[0] + b_list[1] > 0 and b_list[2] + b_list[3] + b_list[4] > 0:
        b_key_names[b] = None
        write_counter += sum(b_list)

with open('business.json', 'r', 8192) as f:
    for line in f:
        business_entry = json.loads(line)
        b_id, b_name = business_entry['business_id'], business_entry['name']
        b_name = b_name.translate(punc_newline_transtab).lower()
        if b_id in b_key_names:
            b_key_names[b_id] = b_name


b_keys = list(b_key_names.keys())

spacing = len(b_keys) // CORES
jobs = [(i, b_keys[i * spacing : (i + 1) * spacing]) for i in range(CORES - 1)]
jobs.append((CORES - 1, b_keys[(CORES - 1) * spacing :]))

counter = 0
for i in range(CORES):
    _, job_assignment = jobs[i]
    assignment_dict = {}
    for b in job_assignment:
        assignment_dict[b] = (b_key_names[b], counter)
        counter += 1
    jobs[i] = (i, assignment_dict)

ret = sum(pool.map(_mapped_saver, jobs))
print('to_write', write_counter, 'num written', ret)
pool.close()