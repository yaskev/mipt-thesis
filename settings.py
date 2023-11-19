from utils.typing import ComplexNetworkType

# Launch Monte Carlo or not
USE_DATA_FROM_FILE = True

# Monte Carlo parameters
NUMBER_OF_PATHS = 100
NUMBER_OF_STEPS = 365
CONFIDENCE_INTERVAL = 0.95
USE_ANTITHETIC_VARIATES = True
PLOT_SOME_PATHS = False

START_SHIFT = None # years

# Random number generator seeds
PATH_RANDOM_SEED = 42  # Set None not to fix seed
OPTIONS_PARAMS_RANDOM_SEED = 123  # Set None not to fix seed

# Neural networks
DATASET_SIZE = 100000  # Used only when USE_DATA_FROM_FILE = False
EPOCHS_COUNT = 500
BATCH_SIZE = 256
NETWORK_TYPE = ComplexNetworkType.POSITIVE_NETWORK

# Sigma
SUBTRACT_INTRINSIC_VALUE = False
SIGMA_USE_SCALER = False

USE_PRETRAINED_NET = True
SAVE_TRAINED_NET = False
# POSITIVE_MODEL_PATH = 'pos_new_mc_1000.sav'
POSITIVE_MODEL_PATH = 'pos_with_int_val.sav'
# POSITIVE_MODEL_PATH = 'trained_pos_doubles.sav'
CONVEX_MODEL_PATH = 'conv_lr_decay_19_11_x8.sav'
# CONVEX_MODEL_PATH = 'trained_con_doubles.sav'
SIGMA_MODEL_PATH = 'trained_sigma_shift.sav'

# General
VERBOSE = True
FIXED_AVG_TYPE = None
CALC_GREEKS = False
WITH_CI_STATS = True and USE_PRETRAINED_NET  # Turn off when training
