# data_loader.py —— 数据加载与训练/测试集划分

import pandas as pd
from config import (
    LOG_RETURN_PATH,
    START_DATE,
    END_DATE,
    TRAIN_START,
    TRAIN_END,
    TEST_START,
    TEST_END,
)


def load_log_return(path: str = LOG_RETURN_PATH) -> pd.Series:
    """加载对数收益率数据，返回按日期索引的 Series."""
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df["log_return"].dropna()


def filter_by_window(series: pd.Series) -> pd.Series:
    """按论文时间窗口（2010-01 至 2025-06）筛选数据."""
    return series.loc[START_DATE:END_DATE]


def split_train_test(
    series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """划分训练集和测试集.

    Returns
    -------
    train : pd.Series
        训练集 (2010-01 – 2023-12)
    test : pd.Series
        测试集 (2024-01 – 2025-06)
    """
    train = series.loc[TRAIN_START:TRAIN_END]
    test = series.loc[TEST_START:TEST_END]
    return train, test


def load_data():
    """一站式加载：从文件读取 → 时间窗口筛选 → 划分训练/测试集.

    Returns
    -------
    full_series : pd.Series
        全量对数收益率 (2010-2025)
    train : pd.Series
        训练集
    test : pd.Series
        测试集
    """
    raw = load_log_return()
    full_series = filter_by_window(raw)
    train, test = split_train_test(full_series)
    return full_series, train, test