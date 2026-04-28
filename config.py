"""Constants and hyperparameters."""

SEED = 16
WINDOWS = [5, 10, 20, 30, 50, 100, 200]

# ── 序列数量 ──
N_TRAIN_SERIES = 250000      # 训练序列数
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
LAMBDA_REWARD = 0.5          # 0=纯夏普率差值, 1=纯期末收益差值
REWARD_SCALE = 100
ORTHO_COEFF  = 0.1         # 路由Query向量正交正则系数
EPS_CLIP    = 0.2
BETA_KL     = 0.0
ENTROPY_COEFF = 0.0
N_EPISODES  = 200000
SAVE_EVERY  = 500
LR          = 1e-3
WEIGHT_DECAY = 0
BATCH_SIZE  = 256

# Output
OUTPUT_DIR = "output"

# Dynamic loop
MAX_ITERATIONS = 24
MIN_ITERATIONS = 1
ITER_REWARD_START = 1.0     # min_iterations 时的奖励系数
ITER_REWARD_END   = 0.75     # max_iterations 时的奖励系数

# MoA / MoE
N_Q_EXPERTS   = 64         # MoA Q 投影专家数
N_KV_EXPERTS  = 48         # MoA K/V 投影专家数
N_FFN_EXPERTS = 64         # MoE 前馈专家数

# 路由参数 (Q路由, KV路由, FFN路由)
TOP_PROB = (0.8, 0.8, 0.8)
MAX_K    = (8, 6, 8)
