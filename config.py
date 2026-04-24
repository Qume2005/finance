"""Constants and hyperparameters."""

SEED = 2048
WINDOWS = [5, 10, 20, 30, 50, 100, 200]

# ── 序列数量 ──
N_TRAIN_SERIES = 3500       # 训练序列数
N_TEST_SERIES  = 10        # 测试序列数
N_SERIES = N_TRAIN_SERIES + N_TEST_SERIES

# ── 每条序列的样本长度（含 warm-up buffer）──
N_TRAIN_DAYS = 1000      # 每条训练序列的原始天数
N_TEST_DAYS  = 350       # 每条测试序列的原始天数
WARMUP = 250              # rolling window 的 warm-up 天数

# ── 序列分配（自动）──
TRAIN_IDS = list(range(N_TRAIN_SERIES))
TEST_IDS  = list(range(N_TRAIN_SERIES, N_SERIES))

# GRPO
EPISODE_LEN = 200
G_SAMPLES   = 24
LAMBDA_DD   = 2.0
EPS_CLIP    = 0.2
BETA_KL     = 0.04
N_EPISODES  = 2000
LR          = 1e-3

# Output
OUTPUT_DIR = "output"
