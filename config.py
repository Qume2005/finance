"""Constants and hyperparameters."""

SEED = 42
WINDOWS = [5, 10, 20, 30, 50, 100, 200]

# Data
N_TRAIN  = 7000
N_TEST   = 3000
N_DAYS   = N_TRAIN + N_TEST + 250  # 250-day warm-up buffer
N_SERIES = 2

# Train / Test series assignment
TRAIN_IDS = [0]
TEST_IDS  = [1]

# GRPO
EPISODE_LEN = 200
G_SAMPLES   = 24
LAMBDA_DD   = 2.0
EPS_CLIP    = 0.2
BETA_KL     = 0.04
N_EPISODES  = 800
LR          = 5e-4

# Output
OUTPUT_DIR = "output"
