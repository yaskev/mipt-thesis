from utils.typing import ComplexNetworkType, OptionAvgType

# Launch Monte Carlo or not
USE_DATA_FROM_FILE = False

# Monte Carlo parameters
NUMBER_OF_PATHS = 4000
NUMBER_OF_STEPS = 365
CONFIDENCE_INTERVAL = 0.95
USE_ANTITHETIC_VARIATES = True
PLOT_SOME_PATHS = False

START_SHIFT = None # years

# Random number generator seeds
PATH_RANDOM_SEED = 42  # Set None not to fix seed
OPTIONS_PARAMS_RANDOM_SEED = 123  # Set None not to fix seed

# Neural networks
DATASET_SIZE = 1000  # Used only when USE_DATA_FROM_FILE = False
EPOCHS_COUNT = 3000
BATCH_SIZE = 256
NETWORK_TYPE = ComplexNetworkType.POSITIVE_NETWORK

# Sigma
SUBTRACT_INTRINSIC_VALUE = False
SIGMA_USE_SCALER = False

USE_PRETRAINED_NET = True
SAVE_TRAINED_NET = False
# POSITIVE_MODEL_PATH = 'pos_new_mc_1000.sav'
POSITIVE_MODEL_PATH = 'pos_net_0_95.sav'
# POSITIVE_MODEL_PATH = 'models/pos/pos_new_mc_1000.sav'
# POSITIVE_MODEL_PATH = 'trained_pos_doubles.sav'
CONVEX_MODEL_PATH = '../../models/conv/conv_lr_decay_19_11_x8.sav'
# CONVEX_MODEL_PATH = 'models/conv/conv_lr_decay_19_11_x8.sav'
# CONVEX_MODEL_PATH = 'trained_con_doubles.sav'
SIGMA_MODEL_PATH = 'trained_sigma_shift.sav'
SEMIPOSITIVE_MODEL_PATH = '../../semipos_net_0_95.sav'

# General
VERBOSE = True
FIXED_AVG_TYPE = None
CALC_GREEKS = False
WITH_CI_STATS = False and USE_PRETRAINED_NET  # Turn off when training
