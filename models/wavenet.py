import tensorflow as tf

def causal_conv(inputs, weights, biases, dilation, scope):
    """
    Inputs of shape [batch_size, input_length, input_channels]
    Weights of shape [filter_width, input_channels, output_channels]
    Biases of shape [output_channels]
        Set Biases to None for no Bias

    Performs causal convolution
    """

    with tf.name_scope(scope):
        input_length = tf.shape(inputs)[1]
        filter_width = tf.shape(weights)[0]
        padded_inputs = tf.pad(inputs,
                               [(0, 0), ((filter_width - 1) * dilation, 0), (0, 0)])

        # The convolution filter will stop at the right edge of input stream
        # Need to pad off the left edge of the temporal dimesion
        output = tf.nn.convolution(input=padded_inputs,
                                   filter=weights,
                                   padding='VALID',
                                   strides=[1],
                                   dilation_rate=[dilation])

        output += biases
        return output



class WaveNet(object):
    """
    General WaveNet

    Usage:
    - dilations: list with dilation factor for each layer.
          Ex: [1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32]
    - batch_size: best set to 1
    - filter_width: Width of each 1d conv filter. Usually set to 2
    - residual_channels: Number of filters for each residual layer
    - dilation_channels: Number of filters for each dilation layer
    - input_channels: Dimension of one elemnet of input stream.
          Ex: 256 for one-hot ASCII, 1 for ASCII raw
    - skip_channels: Number of filters for skip-connections
    - use_biases: Sets if bias is added to convolutions. Usually set to False
    """
    def __init__(self,
                 batch_size=1,
                 dilations=[1, 2, 4, 8, 16, 32, 64,
                            1, 2, 4, 8, 16, 32, 64,
                            1, 2, 4, 8, 16, 32, 64],
                 filter_width=2,
                 residual_channels=32,
                 dilation_channels=32,
                 input_channels=1,
                 skip_channels=256,
                 use_biases=False):
        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.input_channels = input_channels
        self.output_channels = input_channels
        self.skip_channels = 16
        self.use_biases = use_biases

        self.variables = dict()
        print("Receptive field is " + str((filter_width - 1) * sum(dilations) + 1))

        # Generate all variables
        with tf.variable_scope('wavenet_vars'):
            # Generate weights and biases for input layer
            with tf.variable_scope('start_layer'):
                filter = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[self.filter_width,
                                                                                    self.input_channels,
                                                                                    self.residual_channels]),
                                      name='start_filter')
                filter_bias = tf.Variable(tf.zeros([self.residual_channels]),
                                          name="start_bias")
                self.variables['start_layer'] = {'filter':filter, 'biases':filter_bias}


            # Generate weights, biases for dilated layers
            self.variables['dilated_layers'] = []
            with tf.variable_scope('dilated_layers'):
                for i, dilation in enumerate(self.dilations):
                    with tf.variable_scope('layer{}'.format(i)):
                        filter = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[self.filter_width,
                                                                                           self.residual_channels,
                                                                                           self.dilation_channels]),
                                             name="layer{}_filter".format(i))
                        gate   = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[self.filter_width,
                                                                                           self.residual_channels,
                                                                                           self.dilation_channels]),
                                             name="layer{}_gate".format(i))
                        dense  = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[1,
                                                                                           self.dilation_channels,
                                                                                           self.residual_channels]),
                                             name="layer{}_dense".format(i))
                        skip   = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[1,
                                                                                           self.dilation_channels,
                                                                                           self.skip_channels]),
                                             name="layer{}_skip".format(i))
                        if (self.use_biases):
                            filter_bias = tf.Variable(tf.zeros([self.dilation_channels]),
                                                      name="layer{}_filter_bias".format(i))
                            gate_bias   = tf.Variable(tf.zeros([self.dilation_channels]),
                                                      name="layer{}_gate_bias".format(i))
                            dense_bias  = tf.Variable(tf.zeros([self.residual_channels]),
                                                      name="layer{}_dense_bias".format(i))
                            skip_bias   = tf.Variable(tf.zeros([self.skip_channels]),
                                                      name="layer{}_skip_bias".format(i))
                        else:
                            filter_bias = tf.zeros([self.dilation_channels])
                            gate_bias   = tf.zeros([self.dilation_channels])
                            dense_bias  = tf.zeros([self.residual_channels])
                            skip_bias   = tf.zeros([self.skip_channels])

                        self.variables['dilated_layers'].append({'filter':filter,
                                                                 'gate':gate,
                                                                 'dense':dense,
                                                                 'skip':skip,
                                                                 'filter_bias':filter_bias,
                                                                 'gate_bias':gate_bias,
                                                                 'dense_bias':dense_bias,
                                                                 'skip_bias':skip_bias})
            with tf.variable_scope('post_layer'):
                filter1 = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[1,
                                                                                    self.skip_channels,
                                                                                    self.skip_channels]),
                                         name="postprocess1_filter".format(i))
                bias1 = tf.Variable(tf.zeros([self.skip_channels]),
                                    name="postprocess1_bias")
                filter2 = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[1,
                                                                                    self.skip_channels,
                                                                                    self.output_channels]),
                                      name="postprocess2_filter".format(i))
                bias2 = tf.Variable(tf.zeros([self.output_channels]),
                                    name="postprocess2_bias")
                self.variables['post_layer'] = {'filter1':filter1,
                                                'bias1'  :bias1,
                                                'filter2':filter2,
                                                'bias2'  :bias2}

    def full_network(self, inputs):
        """
        Creates full network, without dilation queues.
        This is more inefficient.
        Inputs should be of shape [self.batch_size, input_length, self.input_channels]
        """

        # Preprocess via initial causal convolution
        with tf.name_scope('wavenet'):
            # Convert to float
            inputs = tf.cast(inputs, tf.float32)
            # Normal convolution for preprocess
            current_tensor = inputs
            current_tensor = causal_conv(inputs   = current_tensor,
                                         weights  = self.variables['start_layer']['filter'],
                                         biases   = self.variables['start_layer']['biases'],
                                         dilation = 1,
                                         scope    = 'start_layer')

            skip_outputs = []
            with tf.name_scope("dilated_layers"):
                for i, dilation in enumerate(self.dilations):
                    with tf.name_scope("layer{}".format(i)):
                        filter_weights = self.variables['dilated_layers'][i]['filter']
                        gate_weights   = self.variables['dilated_layers'][i]['gate']
                        dense_weights  = self.variables['dilated_layers'][i]['dense']
                        skip_weights   = self.variables['dilated_layers'][i]['skip']

                        filter_bias    = self.variables['dilated_layers'][i]['filter_bias']
                        gate_bias      = self.variables['dilated_layers'][i]['gate_bias']
                        dense_bias     = self.variables['dilated_layers'][i]['dense_bias']
                        skip_bias      = self.variables['dilated_layers'][i]['skip_bias']

                        # The following activation function (gated activation) is taken from WaveNet
                        # They found that this works well for audio data. We might want to
                        #   modify this for textual data
                        #
                        #        |-> [gate]   -|        |-> 1x1 conv -> skip output
                        #        |             |-> (*) -|
                        # input -|-> [filter] -|        |-> 1x1 conv -|
                        #        |                                    |-> (+) -> dense output
                        #        |------------------------------------|
                        #
                        # filter receives a tanh activation, gate receives a sigmoid
                        # Skip outputs are sent to end
                        # Residual gets added to 1x1 conv output


                        # Generate tanh(W_{filter} x) + sigmoid(W_{gate} x)
                        filter_output = causal_conv(inputs   = current_tensor,
                                                    weights  = filter_weights,
                                                    biases   = filter_bias,
                                                    dilation = dilation,
                                                    scope    = 'layer{}_filter'.format(i))
                        gate_output   = causal_conv(inputs   = current_tensor,
                                                    weights  = gate_weights,
                                                    biases   = gate_bias,
                                                    dilation = dilation,
                                                    scope    = 'layer{}_gate'.format(i))
                        filter_p_gate = tf.tanh(filter_output) * tf.sigmoid(gate_output)

                        # Generate dense output and add residual
                        current_tensor = current_tensor + tf.nn.conv1d(filter_p_gate,
                                                                       dense_weights,
                                                                       stride  = 1,
                                                                       padding = 'SAME',
                                                                       name    = 'layer{}_dense'.format(i)) \

                        current_tensor += dense_bias

                        # Generate skip_output
                        skip_output = tf.nn.conv1d(filter_p_gate,
                                                   skip_weights,
                                                   stride   = 1,
                                                   padding  = 'SAME',
                                                   name     = 'layer{}_skip'.format(i))
                        skip_output += skip_bias


                        skip_outputs.append(skip_output)


            with tf.name_scope("post_layer"):
                # Perform (+ skip_outputs) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
                # postprocess the output.
                filter1 = self.variables['post_layer']['filter1']
                bias1   = self.variables['post_layer']['bias1']
                filter2 = self.variables['post_layer']['filter2']
                bias2   = self.variables['post_layer']['bias2']

                # Sum the skip connections
                skip_total = sum(skip_outputs)
                relu1 = tf.nn.relu(skip_total)
                conv1 = tf.nn.conv1d(relu1,
                                     filter1,
                                     stride = 1,
                                     padding = 'SAME',
                                     name = 'post_layer_0')
                conv1 += bias1

                relu2 = tf.nn.relu(conv1)
                conv2 = tf.nn.conv1d(relu2,
                                     filter2,
                                     stride = 1,
                                     padding = 'SAME',
                                     name = 'post_layer_1')

                conv2 += bias2

            return conv2
