configurations = {
    'BATCH_SIZE': 1,
    'SEQ_LENGTH' : 2048,
    'USE_BIASES' : False,
    'DILATIONS' : [1, 2, 4, 8, 16, 32, 64, 128, 256,
                 1, 2, 4, 8, 16, 32, 64, 128, 256],
    'VOCAB_SIZE' : 75,
    'LEARNING_RATE' :  0.0001,
    'L2_REGULARIZATION' :  0,
    'DEBUG_STEP' : 100,
    'NUM_EPOCHS' : 100,
    'VALIDATION_SIZE' : 100,
    'SAVE_DIR' : "./model_ckpts/shakespeare"
}