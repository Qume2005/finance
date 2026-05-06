"""Constants and hyperparameters."""

SEED = 16
# MMn 特征的滚动窗口长度（交易日），即用最近 N 日的价格做 min/max 归一化
WINDOWS = [5, 10, 20, 30, 50, 100, 200]

# ── 序列数量 ──

N_TRAIN_SERIES = 400000      # 训练序列数
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
G_SAMPLES   = 48
LAMBDA_REWARD = 0.5          # 0=纯夏普率差值, 1=纯期末收益差值
REWARD_SCALE = 10
ORTHO_COEFF  = 0.5         # 路由Query向量正交正则系数
EPS_CLIP    = 0.2
BETA_KL     = 0.01
ENTROPY_COEFF = 0.01
N_EPISODES  = 10000
INNER_STEPS = 32             # 模型内循环固定步数（每个 halting 点内的 MoA+MoE 重复次数）
SAVE_EVERY  = 500
LR          = 1e-3
WEIGHT_DECAY = 0.01
BATCH_SIZE  = 1

# Output

OUTPUT_DIR = "output"

# Dynamic loop

MAX_ITERATIONS = 12
MIN_ITERATIONS = 1
ITER_REWARD_START = 2.0     # min_iterations 时的奖励系数
ITER_REWARD_END   = 1.0     # max_iterations 时的奖励系数
DEPTH_PENALTY_COEFF = 0.01   # expected_depth 惩罚系数（对抗 backprop "永远继续" 的偏向）

# ── 探索噪声 ──

ROUTE_TEMP    = 2.0         # Q/KV 专家路由温度（>1 更随机，<1 更贪心）
FFN_GUMBEL_TAU = 1.0        # FFN Gumbel 噪声强度（已弃用，保留兼容）
HALT_TEMP     = 2.0         # 停机决策温度（>1 更随机，<1 更贪心）

# MoA Attention

D_HIDDEN     = 42               # 模型隐藏维度
D_KEY        = 16               # 每 Head 的 Key/Query 维度
D_ATTN_HEAD  = 112               # 每 Attention Head 输出维度
N_ATTN_HEADS = 12              # 注意力 Head 数
N_EXPERTS_PER_HEAD = 12        # 每 Head 统一专家数（含零专家，expert 0 对 Q 为零）

# MoE FFN

D_FFN_HEAD   = 112               # 每 FFN Head 输出维度
N_FFN_HEADS = 12              # MoE FFN 路由头数
N_EXPERTS_PER_FFN_HEAD = 12   # 每头专家数（含零专家）

# Head-wise expert parallelism

N_ATTN_HEADS_PER_CARD = 0     # 每张卡有梯度的 attention head 数（0 = 全部）
N_FFN_HEADS_PER_CARD  = 0     # 每张卡有梯度的 FFN head 数（0 = 全部）
HEAD_ROTATION_INTERVAL = 0  # 每隔 N 步轮换 head 分配（0 = 固定不轮换）

# Gradient Checkpointing + Expert Dropout

GRADIENT_CHECKPOINTING = True     # 梯度检查点（用计算换显存）
HEAD_DROP_PROB = 0.5              # Head Dropout 概率（0 = 不丢弃）