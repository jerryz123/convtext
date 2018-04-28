import tensorflow as tf

def create_embed(name, N, DIM):
    return tf.get_variable(name, shape=[N, DIM],
           initializer=tf.contrib.layers.xavier_initializer())
class Generator:
    N_STARS = 5

    def __init__(self, conf, title, stars, text):
        self.raw_title, self.raw_stars, self.raw_text = title, stars, text
        self.conf = conf

        self.N_TITLE_WORDS = self.conf['n_title_words']
        self.N_WORDS, self.D_EMBED = self.conf['n_words'], self.conf['d_embed']

        self.title_embed = create_embed('title_embed', self.N_TITLE_WORDS, self.D_EMBED)
        self.stars_embed = create_embed('star_embed', self.N_STARS, self.D_EMBED)
        self.text_embed = create_embed('text_embed', self.N_WORDS, self.D_EMBED)
        
        
        self.input_text = tf.cast(tf.clip_by_value(self.raw_text, 0, self.N_WORDS - 1), self.raw_text.dtype) #only consider top N_WORDS, rest go to -> <UNK>
        self.input_title = tf.cast(self.raw_title < self.N_TITLE_WORDS, self.raw_title.dtype) * self.raw_title #only consider top N_TITLE_WORDS, rest go to -> <UNK>
        self.input_stars = self.raw_stars
    
    def build(self):
        raise NotImplementedError('Generator is Abstract')

class LSTMGenerator(Generator):
    def build(self):
        title_vec = tf.nn.embedding_lookup(self.title_embed, self.input_title) # (B, T_1, DIM)
        text_vec = tf.nn.embedding_lookup(self.text_embed, self.input_text) # (B, T_2, DIM)
        star_vec = tf.nn.embedding_lookup(self.stars_embed, self.input_stars) # (B, 1, DIM)

        B, T_1, T_2 = tf.shape(star_vec)[0], tf.shape(title_vec)[1], tf.shape(text_vec)[1]

        print(B, T_1, T_2)
        print(1 / 0)
        self.loss = tf.reduce_sum(title_vec - title_vec)

        