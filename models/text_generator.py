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

        encoder_in = tf.concat((star_vec, title_vec), 1)

        forward_cell = tf.nn.rnn_cell.BasicLSTMCell(self.conf['encoder_units'])
        backward_cell = tf.nn.rnn_cell.BasicLSTMCell(self.conf['encoder_units'])
        
        bi_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, encoder_in, dtype = tf.float32)
        encoder_outputs = tf.concat(bi_outputs, -1)

        encoder_state = []
        for layer_id in range(2):
            encoder_state.append(bi_encoder_state[0][layer_id])  # forward
            encoder_state.append(bi_encoder_state[1][layer_id])  # backward
        encoder_state = tuple(encoder_state)


        attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.conf['encoder_units'], encoder_outputs)
        helper = tf.contrib.seq2seq.TrainingHelper(text_vec, [T_2-1 for _ in range(self.conf['batch_size'])])

        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.conf['encoder_units'])
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,attention_layer_size=self.conf['encoder_units'])
        initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.conf['batch_size'])
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state, output_layer= tf.layers.Dense(self.N_WORDS, use_bias=False))

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        self.logits = outputs.rnn_output
        
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_text[:, 1:], logits=self.logits)

        self.loss = tf.reduce_sum(crossent) / tf.cast(B, tf.float32)


        