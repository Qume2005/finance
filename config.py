"""Constants and hyperparameters."""

SEED = 16
WINDOWS = [5, 10, 20, 30, 50, 100, 200]

# ── 序列数量 ──
N_TRAIN_SERIES = 10000      # 训练序列数
N_TEST_SERIES  = 20        # 测试序列数
N_SERIES = N_TRAIN_SERIES + N_TEST_SERIES

# ── 每条序列的样本长度 ──
N_TRAIN_DAYS = 450      # 每条训练序列的原始天数
N_TEST_DAYS  = 450       # 每条测试序列的原始天数

# ── 序列分配（自动）──
TRAIN_IDS = list(range(N_TRAIN_SERIES))
TEST_IDS  = list(range(N_TRAIN_SERIES, N_SERIES))

# GRPO
EPISODE_LEN = 50
G_SAMPLES   = 64
LAMBDA_DD   = 2.0
EPS_CLIP    = 0.2
BETA_KL     = 0.04
N_EPISODES  = 5000
SAVE_EVERY  = 500
LR          = 1
WEIGHT_DECAY = 0.1
BATCH_SIZE  = 16

# Output
OUTPUT_DIR = "output"
