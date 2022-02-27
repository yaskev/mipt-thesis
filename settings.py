from utils.typing import OptionAvgType

# Launch Monte Carlo or not
USE_DATA_FROM_FILE = True

# Monte Carlo parameters
NUMBER_OF_PATHS = 100
NUMBER_OF_STEPS = 365
CONFIDENCE_INTERVAL = 0.95
USE_ANTITHETIC_VARIATES = True

# Random number generator seeds
PATH_RANDOM_SEED = None  # Set None not to fix seed
OPTIONS_PARAMS_RANDOM_SEED = None  # Set None not to fix seed

# Neural networks
DATASET_SIZE = 100000  # Used only when USE_DATA_FROM_FILE = False
EPOCHS_COUNT = 500
BATCH_SIZE = 100
USE_CONVEX_NETWORK = True

# General
VERBOSE = True
FIXED_AVG_TYPE = OptionAvgType.ARITHMETIC
