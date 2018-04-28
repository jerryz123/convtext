import os
from models.text_generator import LSTMGenerator
configuration = {
    'data_dir' : os.path.join(os.environ.get('YELP_DATA_DIR', './'), 'records/'),
    'batch_size' : 8,
    'train_val_split' : 0.95, #0.95 by default
    'model' : LSTMGenerator,
    'model_dir' : os.path.join(os.environ.get('SAVE_DIR', './'), 'lstm_model/'),
    'd_embed' : 512,
    'n_words' : 7000,
    'n_title_words' : 7000,
    'encoder_units' : 256
}
