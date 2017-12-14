START_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2
VOCAB_ATT = {'<S>': 0, '</S>': 1, '<UNK>': 2, '0': 3, '1': 4, '2': 5, '3': 6, '4': 7, '5': 8, '6': 9, '7': 10, '8': 11, '9': 12}
VOCAB_ATT_SIZE = len(VOCAB_ATT)

VOCAB_CTC = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
VOCAB_CTC_SIZE = len(VOCAB_CTC)

ATT_EMBED_DIM = 512
BATCH_SIZE = 32
RNN_UNITS = 256
TRAIN_STEP = 1000000
IMAGE_HEIGHT = 32
MAXIMUM__DECODE_ITERATIONS = 20
DISPLAY_STEPS = 100
LOGS_PATH = 'logs_path'
CKPT_DIR = 'save_model'
