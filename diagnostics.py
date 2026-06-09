# diagnostics.py —— 模型诊断与检验函数

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch
from config import ARCH_LM_LAGS, ADF_MAX_DIFF


def adf_test(series: pd.Series, max_diff: int = ADF_MAX_DIFF) -> pd.DataFrame:
    """对序列及其各阶差分执行 ADF 单位根检验.

    Parameters
    ----------
    series : pd.Series
        待检验时间序列
    max_diff : int
        最大差分阶数

    Returns
    -------
    pd.DataFrame
        列：差分阶数、t 统计量、p 值、AIC、1%/5%/10% 临界值
    """
    results = []
    for d in range(max_diff + 1):
        s = series if d == 0 else series.diff(d).dropna()
        res = adfuller(s, autolag="AIC")
        results.append(
            {
                "差分阶数": d,
                "t": res[0],
                "P": res[1],
                "AIC": res[2],
                "1%临界值": res[4]["1%"],
                "5%临界值": res[4]["5%"],
                "10%临界值": res[4]["10%"],
                "样本量": res[3],
            }
        )
    df = pd.DataFrame(results)
    return df


def ljung_box_test(residuals: pd.Series, lags: list = None) -> pd.DataFrame:
    """对残差序列执行 Ljung-Box Q 检验.

    Parameters
    ----------
    residuals : pd.Series
        残差序列
    lags : list, optional
        滞后阶数列表，默认 [6, 12, 18, 24, 30]

    Returns
    -------
    pd.DataFrame
        列：滞后阶数、Q 统计量、P 值
    """
    if lags is None:
        lags = [6, 12, 18, 24, 30]
    # 如果数据量不够最大滞后阶数，截断
    lags = [l for l in lags if l < len(residuals)]
    res = acorr_ljungbox(residuals.dropna(), lags=lags, return_df=True)
    # 重构 DataFrame
    results = []
    for lag in lags:
        row = res.loc[lag] if lag in res.index else {"lb_stat": np.nan, "lb_pvalue": np.nan}
        results.append(
            {
                "滞后阶数": lag,
                "Q统计量": row["lb_stat"],
                "P值": row["lb_pvalue"],
            }
        )
    return pd.DataFrame(results)


def arch_lm_test(residuals: pd.Series, nlags: int = ARCH_LM_LAGS) -> dict:
    """执行 ARCH-LM 检验，检测残差是否存在条件异方差.

    Parameters
    ----------
    residuals : pd.Series
        残差序列
    nlags : int
        检验滞后阶数，论文使用 12

    Returns
    -------
    dict
        包含 LM 统计量、LM p 值、F 统计量、F p 值
    """
    resid = residuals.dropna()
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(resid, nlags=nlags)
    return {
        "LM统计量": lm_stat,
        "LM_P值": lm_pvalue,
        "F统计量": f_stat,
        "F_P值": f_pvalue,
        "滞后阶数": nlags,
    }