# visualization.py —— 绑图模块（Figure 1 & Figure 2）

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from config import FIGURES_DIR, FONT_FAMILY, MATH_FONTSET


def _setup_plt():
    """统一设置 matplotlib 中文字体和样式."""
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["mathtext.fontset"] = MATH_FONTSET
    plt.rcParams["axes.unicode_minus"] = False


def plot_log_return_series(
    series: pd.Series,
    save: bool = True,
    filename: str = "figure1_log_return.pdf",
) -> plt.Figure:
    """Figure 1: 对数收益率时间序列图.

    Parameters
    ----------
    series : pd.Series
        全量对数收益率 (2010-2025)
    save : bool
        是否保存图片
    filename : str
        输出文件名

    Returns
    -------
    plt.Figure
    """
    _setup_plt()
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(series.index, series.values, linewidth=0.5, color="black")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.7)
    ax.set_title("图1：上证综指对数收益率时间序列 (2010.01 – 2025.06)")
    ax.set_xlabel("日期")
    ax.set_ylabel("对数收益率")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig


def plot_acf_pacf(
    series: pd.Series,
    lags: int = 40,
    save: bool = True,
    filename: str = "figure2_acf_pacf.pdf",
) -> plt.Figure:
    """Figure 2: ACF 和 PACF 图（两子图）.

    Parameters
    ----------
    series : pd.Series
        对数收益率序列
    lags : int
        显示的滞后阶数
    save : bool
        是否保存
    filename : str
        输出文件名

    Returns
    -------
    plt.Figure
    """
    _setup_plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    plot_acf(series.dropna(), lags=lags, ax=ax1)
    ax1.set_title("(a) ACF 图")
    ax1.set_xlabel("滞后阶数")
    ax1.set_ylabel("自相关系数")

    plot_pacf(series.dropna(), lags=lags, ax=ax2, method="ywm")
    ax2.set_title("(b) PACF 图")
    ax2.set_xlabel("滞后阶数")
    ax2.set_ylabel("偏自相关系数")

    fig.suptitle("图2：对数收益率 ACF / PACF 图", y=1.01)
    fig.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig