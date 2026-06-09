# config.py —— 全局配置参数
# 注意：在终端中运行 Python 脚本时请使用 py -3.10 命令

import os

# ======================== 路径配置 ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")
RESULT_DIR = os.path.join(BASE_DIR, "result")
FIGURES_DIR = os.path.join(RESULT_DIR, "figures")

# 确保输出目录存在
os.makedirs(FIGURES_DIR, exist_ok=True)

# 数据文件路径
LOG_RETURN_PATH = os.path.join(PREPROCESSED_DIR, "log_return.csv")

# ======================== 数据切分配置 ========================
# 论文选用的总样本：2010年1月 - 2025年6月
START_DATE = "2010-01-01"
END_DATE = "2025-06-30"

# 训练集：2010年1月 - 2023年12月
TRAIN_START = "2010-01-01"
TRAIN_END = "2023-12-31"

# 测试集：2024年1月 - 2025年6月
TEST_START = "2024-01-01"
TEST_END = "2025-06-30"

# ======================== 模型阶数配置 ========================
# ARIMA 阶数（论文使用的是 ARIMA(2,0,2)）
ARIMA_ORDER = (2, 0, 2)

# GARCH 阶数
GARCH_P = 1
GARCH_Q = 1

# EGARCH 阶数
EGARCH_P = 1
EGARCH_Q = 1

# ARCH-LM 检验滞后阶数
ARCH_LM_LAGS = 12

# ADF 检验最大差分阶数
ADF_MAX_DIFF = 2

# ======================== 图表字体配置 ========================
# 统一的 matplotlib 字体设置
FONT_FAMILY = ['Times New Roman', 'SimSun']
MATH_FONTSET = 'stix'