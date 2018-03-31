DATA_DIR = '../datasets/'
configuration = {
    'BATCH_SIZE': 1,
    'SEQ_LENGTH' : 2048,
    'USE_BIASES' : False,
    'DILATIONS' : [1, 2, 4, 8, 16, 32, 64, 128, 256,
                 1, 2, 4, 8, 16, 32, 64, 128, 256],
    'FILTER_WIDTH' : 2,
    'RESIDUAL_CHANNELS' : 32,
    'DILATION_CHANNELS' : 32,
    'NUM_CHANNELS' : 1,
    'SKIP_CHANNELS' : 16,
    'QUEUE_LOADER' : True,
    'TRAIN_FILES' : [DATA_DIR + 'corpus.txt'],
    'TEST_FILES' : [DATA_DIR + 'corpus_test.txt'],
    'LEARNING_RATE' :  0.0001,
    'L2_REGULARIZATION' :  0,
    'DEBUG_STEP' : 100,
    'NUM_EPOCHS' : 10000,
    'VALIDATION_SIZE' : 100,
    'SAVE_DIR' : "./model_ckpts/shakespeare"
}