import tensorflow as tf

def one_hot_character_loss(self, net_outputs, inputs, l2_norm=0):
    """
    Returns loss when inputs and outputs are one-hot encoded character sequences
    """
    quantization_channels = tf.shape(inputs)[2]

    # Shift input so network learns to predict next output
    shifted = tf.slice(encoded, [0, 1, 0],
                       [-1, tf.shape(encoded)[1] - 1, -1])
    shifted = tf.pad(shifted, [(0, 0), (0, 1), (0, 0)])

    shifted = tf.reshape(shifted, [-1. quantization_channels])
    prediction = tf.reshape(net_outputs, [-1, quantization_channels])

    loss = tf.nn.softmax_cross_entropy_with_logits(prediction,
                                                   shifted)
    loss = tf.reduce_mean(loss)

    
    tf.scalar_summary('true_one_hot_character_loss', loss)

    l2_loss = l2_norm * tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() \
                                if 'bias' not in v.name])

    tf.scalar_summary('l2_loss', l2_loss)
    total_loss = loss + l2_loss
    tf.scalar_summary('total_loss', total_loss)
    return total_loss
    
    
