import tensorflow as tf
import numpy as np

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
        self.input_stars = self.raw_stars - 1
    
    def build(self, is_Train = True):
        raise NotImplementedError('Generator is Abstract')

class LSTMGenerator(Generator):
    def build(self, is_Train = True):
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

class TransformerGenerator(Generator):
    def _create_pos_encoding(self, inputs):
        T_input, dim = tf.shape(inputs)[1], self.conf['d_embed']
        
        freq_dim_arg = tf.exp(tf.linspace(0., np.log(10000) * (- 2 * (dim - 1) / dim), dim))
        freq_T_arg = tf.linspace(0., tf.cast(T_input - 1, tf.float32), T_input)
        freq_arg = tf.reshape(freq_T_arg, (-1, 1)) * tf.reshape(freq_dim_arg, (1, -1))
        freq_arg = tf.reshape(freq_arg, (-1, dim // 2, 2))

        sin_args, cos_args = tf.sin(freq_arg[:, :, 0]), tf.cos(freq_arg[:, :, 1])

        pos_encoding = tf.concat((tf.expand_dims(sin_args, -1), tf.expand_dims(cos_args, -1)), -1)

        return tf.stop_gradient(tf.expand_dims(tf.reshape(pos_encoding, (-1, dim)), 0))
    def _multihead_attention(self, Q_in, K_in, causal_mask = False, is_training = False):
        """

        Credit to kyubyong park for a very helpful reference implementation https://github.com/Kyubyong/transformer/blob/master/modules.py
        """
        d_head = self.conf['sub_head_dim']
        num_heads = self.conf['d_embed'] // d_head
        assert d_head * num_heads == self.conf['d_embed'], 'INPUT DIM MUST EQUAL D * N_HEADS FOR RESIDUAL LAYER'
        b_size = tf.shape(Q_in)[0]
        
        Q = tf.layers.dense(Q_in, d_head * num_heads) #(B, T_q, H * D) 
        K = tf.layers.dense(K_in, d_head * num_heads) #(B, T_k, H * D)
        V = tf.layers.dense(K_in, d_head * num_heads) #(B, T_k, H * D)

        Q_heads = tf.concat(tf.split(Q, num_heads, axis = 2), axis = 0) #(B * H, T_q,  D)
        K_heads = tf.concat(tf.split(K, num_heads, axis = 2), axis = 0) #(B * H, T_k,  D)
        V_heads = tf.concat(tf.split(V, num_heads, axis = 2), axis = 0) #(B * H, T_k,  D)

        scaled_dot_attention = tf.matmul(Q_heads, tf.transpose(K_heads, [0, 2, 1])) / np.sqrt(d_head) #(B * H, T_q, T_k)
        
        if causal_mask:
            c_mask = tf.ones_like(scaled_dot_attention[0, :, :])
            c_mask = tf.matrix_band_part(c_mask, -1, 0)
            c_mask = tf.tile(tf.expand_dims(c_mask, 0), [b_size * num_heads, 1, 1])

            neg_inf = tf.ones_like(c_mask) * (-2 ** 32 + 1)
            scaled_dot_attention = tf.where(tf.equal(c_mask, 0), neg_inf, scaled_dot_attention)
        
        
        attention_weights = tf.nn.softmax(scaled_dot_attention)

        if 'dropout' in self.conf:
            attention_weights = tf.layers.dropout(attention_weights, rate=self.conf['dropout'], training=tf.convert_to_tensor(is_training))
        
        output_heads = tf.matmul(attention_weights, V_heads) #(B * H, T_q, D)
        outputs = tf.concat(tf.split(output_heads, num_heads, axis = 0), axis = 2) + Q_in #(B, T_q, D * H)

        return tf.contrib.layers.layer_norm(outputs)

    def _feedforward_layer(self, inputs):
        num_inner, num_out = self.conf['feedforward_dim'], self.conf['d_embed']

        inner_mult = tf.layers.conv1d(inputs, num_inner, 1, activation=tf.nn.relu)
        outer_mult = tf.layers.conv1d(inner_mult, num_out, 1)

        return tf.contrib.layers.layer_norm(outer_mult + inputs)
    
    def build(self, is_Train = True):
        title_vec = tf.nn.embedding_lookup(self.title_embed, self.input_title) # (B, T_title, DIM)
        text_vec = tf.nn.embedding_lookup(self.text_embed, self.input_text) # (B, T_text, DIM)
        star_vec = tf.nn.embedding_lookup(self.stars_embed, self.input_stars) # (B, 1, DIM)

        encoder_inputs = tf.concat((star_vec, title_vec), 1)
        shifted_decoder_in = text_vec[:, :-1, :]

        pos_encoder_inputs = encoder_inputs + self._create_pos_encoding(encoder_inputs)
        pos_decoder_inputs = shifted_decoder_in + self._create_pos_encoding(shifted_decoder_in)
        
        prev_enc_out = pos_encoder_inputs
        prev_dec_out = pos_decoder_inputs
        
        for i in range(self.conf['num_repeats']):
            #encoder cell
            enc_self_attention = self._multihead_attention(prev_enc_out, prev_enc_out, is_training=is_Train)
            enc_out = self._feedforward_layer(enc_self_attention)

            #decoder cell
            dec_masked_self_attention = self._multihead_attention(prev_dec_out, prev_dec_out, causal_mask=True, is_training=is_Train)
            dec_enc_attention = self._multihead_attention(dec_masked_self_attention, enc_out, is_training=is_Train)
            dec_out = self._feedforward_layer(dec_enc_attention)

            prev_dec_out = dec_out
            prev_enc_out = enc_out
        
        self.logits = tf.layers.dense(prev_dec_out, self.N_WORDS)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_text[:, 1:], logits=self.logits)

        self.loss = tf.reduce_sum(crossent) / tf.cast(tf.shape(self.logits)[0], tf.float32) / tf.cast(tf.shape(self.logits)[1], tf.float32)
        
        