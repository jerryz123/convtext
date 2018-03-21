import tensorflow as tf

def one_hot_character_loss(net_outputs, inputs, l2_norm=0):
    """
    Returns loss when inputs and outputs are one-hot encoded character sequences
    """
    with tf.name_scope("one_hot_character_loss"):
        quantization_channels = tf.shape(inputs)[2]

        # Shift input so network learns to predict next output
        shifted = tf.slice(inputs, [0, 1, 0],
                           [-1, tf.shape(inputs)[1] - 1, -1])
        shifted = tf.pad(shifted, [(0, 0), (0, 1), (0, 0)])
        shifted = tf.reshape(shifted, [-1, quantization_channels])
        prediction = tf.reshape(net_outputs, [-1, quantization_channels])

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction,
                                                       labels=shifted)
        loss = tf.reduce_mean(loss)


        tf.summary.scalar('true_one_hot_character_loss', loss)

        l2_loss = l2_norm * tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() \
                                    if 'bias' not in var.name])

        tf.summary.scalar('l2_loss', l2_loss)
        total_loss = loss + l2_loss
        tf.summary.scalar('total_loss', total_loss)
    return total_loss
    
    
