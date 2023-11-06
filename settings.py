from utils.typing import ComplexNetworkType

# Launch Monte Carlo or not
USE_DATA_FROM_FILE = True

# Monte Carlo parameters
NUMBER_OF_PATHS = 40000
NUMBER_OF_STEPS = 365
CONFIDENCE_INTERVAL = 0.95
USE_ANTITHETIC_VARIATES = True
PLOT_SOME_PATHS = False

# Random number generator seeds
PATH_RANDOM_SEED = 123  # Set None not to fix seed
OPTIONS_PARAMS_RANDOM_SEED = 355  # Set None not to fix seed

# Neural networks
DATASET_SIZE = 5000  # Used only when USE_DATA_FROM_FILE = False
EPOCHS_COUNT = 200
BATCH_SIZE = 256
NETWORK_TYPE = ComplexNetworkType.CONVEX_NETWORK

# General
VERBOSE = True
FIXED_AVG_TYPE = None
CALC_GREEKS = False
WITH_CI_STATS = True # Turn off when training

SAVE_TRAINED_NET = False
USE_PRETRAINED_NET = True
POSITIVE_MODEL_PATH = 'trained_pos_doubles_2.sav'
# POSITIVE_MODEL_PATH = 'trained_pos_doubles.sav'
CONVEX_MODEL_PATH = 'trained_convex.sav'
# CONVEX_MODEL_PATH = 'trained_con_doubles.sav'
SIGMA_MODEL_PATH = 'trained_sigma.sav'
