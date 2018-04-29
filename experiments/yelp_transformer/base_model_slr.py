import os
from models.text_generator import TransformerGenerator
configuration = {
    'data_dir' : os.path.join(os.environ.get('YELP_DATA_DIR', './'), 'records/'),
    'batch_size' : 8,
    'train_val_split' : 0.95, #0.95 by default
    'model' : TransformerGenerator,
    'model_dir' : os.path.join(os.environ.get('SAVE_DIR', './'), 'transformer_model_slr/'),
    'd_embed' : 512,
    'n_words' : 7000,
    'n_title_words' : 7000,
    'sub_head_dim' : 64,
    'dropout' : 0.25,
    'feedforward_dim' : 2048,
    'num_repeats' : 6,
    'clip_grad' : 1.0,
    'learning_rate' : 0.0001,
    'n_iters' : 300000,
    'save_step' : 10000
}
