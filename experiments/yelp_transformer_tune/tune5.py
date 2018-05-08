import os
from models.text_generator import TransformerGenerator

DEFAULT_SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) #saved in experiment folder by default
configuration = {
    'data_dir' : os.path.join(os.environ.get('YELP_DATA_DIR', './'), 'records/'),
    'batch_size' : 16,
    'train_val_split' : 0.95, #0.95 by default
    'model' : TransformerGenerator,
    'model_dir' : os.path.join(os.environ.get('SAVE_DIR', DEFAULT_SAVE_DIR), 'tune5/'),
    'd_embed' : 512,
    'n_words' : 14000,
    'n_title_words' : 14000,
    'sub_head_dim' : 64,
    'dropout' : 0.25,
    'feedforward_dim' : 3096,
    'num_repeats' : 4,
    'n_iters' : 300000,
    'save_step' : 1000
}
