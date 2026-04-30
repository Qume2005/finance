"""Constants and hyperparameters."""

SEED = 16
WINDOWS = [5, 10, 20, 30, 50, 100, 200]

# ── 序列数量 ──
N_TRAIN_SERIES = 200000      # 训练序列数
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
BETA_KL     = 0.00
ENTROPY_COEFF = 0.00
N_EPISODES  = 10000
INNER_STEPS = 12             # 模型内循环固定步数（每个 halting 点内的 MoA+MoE 重复次数）
SAVE_EVERY  = 500
LR          = 1e-3
WEIGHT_DECAY = 0
BATCH_SIZE  = 2
# Output
OUTPUT_DIR = "output"

# Dynamic loop
MAX_ITERATIONS = 6
MIN_ITERATIONS = 1
ITER_REWARD_START = 2.0     # min_iterations 时的奖励系数
ITER_REWARD_END   = 0.5     # max_iterations 时的奖励系数
DEPTH_PENALTY_COEFF = 0.1   # expected_depth 惩罚系数（对抗 backprop "永远继续" 的偏向）

# ── 探索噪声 ──
ROUTE_TEMP    = 2.0         # Q/KV 专家路由温度（>1 更随机，<1 更贪心）
FFN_GUMBEL_TAU = 1.0        # FFN Gumbel 噪声强度（0=确定性，越大越随机）
HALT_TEMP     = 2.0         # 停机决策温度（>1 更随机，<1 更贪心）

# MoA Attention
D_HIDDEN     = 48               # 模型隐藏维度
D_KEY        = 16               # 每 Head 的 Key/Query 维度
D_ATTN       = 48               # 注意力层输出维度（所有 Head concat 后）
N_ATTN_HEADS = 3               # 注意力 Head 数
N_EXPERTS_PER_HEAD = 8        # 每 Head 统一专家数（含零专家，expert 0 对 Q 为零）

# MoE FFN
N_FFN_EXPERTS = 48            # MoE 前馈专家数
FFN_TOP_PROB  = 0.8
FFN_MAX_K     = 6
