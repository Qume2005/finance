"""Constants and hyperparameters."""

SEED = 16
WINDOWS = [5, 10, 20, 30, 50, 100, 200]

# ── 序列数量 ──
N_TRAIN_SERIES = 100000      # 训练序列数
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
REWARD_SCALE = 10
ORTHO_COEFF  = 0.5         # 路由Query向量正交正则系数
EPS_CLIP    = 0.2
BETA_KL     = 0.0
ENTROPY_COEFF = 0.0
N_EPISODES  = 200000
SAVE_EVERY  = 500
LR          = 1e-3
WEIGHT_DECAY = 0
BATCH_SIZE  = 32
# Output
OUTPUT_DIR = "output"

# Dynamic loop
MAX_ITERATIONS = 24
MIN_ITERATIONS = 1
ITER_REWARD_START = 1.0     # min_iterations 时的奖励系数
ITER_REWARD_END   = 0.75     # max_iterations 时的奖励系数

# MoA Attention
N_ATTN_HEADS = 3               # 注意力 Head 数
N_Q_EXPERTS_PER_HEAD = 4      # 每 Head Q 专家数（含零专家）
N_KV_EXPERTS_PER_HEAD = 4     # 每 Head KV 专家数（含零专家）

# MoE FFN
N_FFN_EXPERTS = 48            # MoE 前馈专家数
FFN_TOP_PROB  = 0.8
FFN_MAX_K     = 3
