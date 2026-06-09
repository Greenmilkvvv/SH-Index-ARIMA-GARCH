# %%
import numpy as np
import pandas as pd

PATH = r"data/raw/SH_index.csv"

df = pd.read_csv(PATH)

# df.head()


# %%
# 获取上证指数日线
# 选择需要的列
df = df[ ['日期', '收盘'] ]
# 更规范的命名
df.rename(columns={'日期': 'date', '收盘': 'close',}, inplace=True)
# 日期格式化
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

PATH_index = r"data/preprocessed/index.csv"
df.to_csv( PATH_index, index=True)


# %%
# 获取对数收益率
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

log_return = df['log_return'].dropna()

# log_return.head()

PATH_log_return = r"data/preprocessed/log_return.csv"
log_return.to_csv( PATH_log_return, index=True)

